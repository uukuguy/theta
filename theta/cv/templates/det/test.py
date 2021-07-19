# Copyright 2021 jwsu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os, json
import os.path as osp
import time
import warnings
from tqdm import tqdm
from loguru import logger
from collections import defaultdict, Counter
import numpy as np
import mmcv
import math
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

from mmdet_transforms import MixUp


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--predict',
                        action='store_true',
                        help='predict results')
    parser.add_argument('--submit',
                        action='store_true',
                        help='generate submission')
    parser.add_argument('--task_name',
                        default="unnamed_task",
                        type=str,
                        help='Task name')
    parser.add_argument('--submission_file',
                        default=None,
                        type=str,
                        help='Submission file')
    parser.add_argument('--test_json_file',
                        default=None,
                        type=str,
                        help='Test json file')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr',
                        type=float,
                        default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--min_size',
                        type=float,
                        default=2,
                        help='Min size (default: 2)')
    parser.add_argument('--prob_threshold',
                        type=float,
                        default=0.1,
                        help='prob threshold (default: 0.1)')

    parser.add_argument('--gpu-collect',
                        action='store_true',
                        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


#%%


def is_intersect_by_area(obj0, obj1, threshold=0.5, debug=False):
    x00, y00, x01, y01, prob0 = obj0
    w0, h0 = x01 - x00, y01 - y00
    x10, y10, x11, y11, prob1 = obj1
    w1, h1 = x11 - x10, y11 - y10

    if x00 > x11 or x01 < x10 or y00 > y11 or y01 < y10:
        return False, 0.0
    if x10 > x01 or x11 < x00 or y10 > y01 or y11 < y00:
        return False, 0.0
    _, x0, x1, _ = sorted([x00, x01, x10, x11])
    _, y0, y1, _ = sorted([y00, y01, y10, y11])
    area = (x1 - x0) * (y1 - y0)
    area0 = w0 * h0
    area1 = w1 * h1
    if debug:
        logger.warning(
            f"x0: {x0:.4f}, x1: {x1:.4f}, y0: {y0:.4f}, y1: {y1:.4f}")
        logger.warning(
            f"area: {area:.4f}, area0: {area0:.4f}, area1: {area1:.4f}")
    is_intersect = False
    if area >= min(area0, area1) * threshold:
        is_intersect = True

    if debug:
        logger.warning(f"is_intersect: {is_intersect}")
    return is_intersect, area


def is_intersect_by_distance(obj0, obj1, dist_ratio=1.0, debug=False):
    x00, y00, x01, y01, prob0 = obj0
    w0, h0 = abs(x01 - x00), abs(y01 - y00)
    x10, y10, x11, y11, prob1 = obj1
    w1, h1 = abs(x11 - x10), abs(y11 - y10)

    cx0 = (x00 + x01) / 2
    cy0 = (y00 + y01) / 2
    cx1 = (x10 + x11) / 2
    cy1 = (y10 + y11) / 2

    cx = cx1 - cx0
    cy = cy1 - cy0

    dist = math.sqrt(cx * cx + cy * cy)
    radius = max(w1, h1)
    if debug:
        logger.warning(f"radius: {radius:0.4f}, dist: {dist:.4f}")

    is_intersect = False
    if radius != 0:
        if dist / radius <= dist_ratio:
            is_intersect = True
    if debug:
        logger.warning(f"is_intersect: {is_intersect}")
    return is_intersect


#  test_img_files = open(
#      "./data/test/annotations/1_testa_user.csv").readlines()[1:]
#  test_img_files = {
#      line.strip().split('/')[1]: i
#      for i, line in enumerate(test_img_files)
#  }
#


def fix_overlap(objects, threshold=0.5, debug=False):
    """
    # 同一图片中每个类别重叠的bbox保留prob最大的一个。
    """
    keep_objects = []
    for i, obj in enumerate(objects):
        found = False
        for j, obj1 in enumerate(keep_objects):
            is_intersect, area = is_intersect_by_area(obj,
                                                      obj1,
                                                      threshold=threshold,
                                                      debug=debug)
            if is_intersect:
                prob0 = obj[4]
                prob1 = obj1[4]
                if prob0 > prob1:
                    keep_objects[j] = obj
                found = True
                break
        if not found:
            keep_objects.append(obj)

    return keep_objects


#%%
def generate_submission(args, outputs, data_loader):

    # classes = ('background', 'badge', 'person', 'glove', 'wrongglove', 'operatingbar',
    #        'powerchecker')
    import importlib
    module_full_name = args.config.split('.')[0].replace('/', '.')
    config_module = importlib.import_module(module_full_name)
    classes = config_module.classes
    num_classes = len(classes)
    id2label = {i + 1: x for i, x in enumerate(classes)}

    test_json_file = args.test_json_file
    assert os.path.exists(test_json_file), f"{test_json_file} does not exist."
    test_json_data = json.load(open(test_json_file, 'r'))
    image_files = [(x['id'], x['file_name'])
                   for x in tqdm(test_json_data['images'], desc="images")]

    assert len(image_files) == len(outputs)

    os.makedirs("submissions", exist_ok=True)
    submission_file = args.submission_file
    if submission_file is None:
        submission_file = f"submissions/submission_{args.task_name}.json"
    prediction_file = f"submissions/prediction_{args.task_name}.json"
    final_results = []
    final_predictions = []
    for _, (
            pice_boxes,
        (orig_img_id, img_filename),
    ) in enumerate(tqdm(zip(outputs, image_files), desc="submissions")):
        #  img_id = test_img_files[img_filename]

        #  img_key, scale, pice_left, pice_top = img_filename.split('.')[0].split(
        #      '__')
        #  assert img_key in img_filename_to_id, f"{img_key} not in img_filename_to_id"
        #  img_id = img_filename_to_id[img_key]

        #  pice_left = int(pice_left)
        #  pice_top = int(pice_top)
        img_id = img_filename.split('.')[0]

        # logger.info(f"pice_boxes: {pice_boxes}")

        cat_objects = defaultdict(list)
        for cat_id in range(1, num_classes + 1):
            for pice_box in pice_boxes[cat_id - 1]:
                x0, y0, x1, y1, prob = pice_box

                w = x1 - x0
                h = y1 - y0
                #  if w < args.min_size or h < args.min_size:
                #      continue

                #  x0 += pice_left
                #  y0 += pice_top

                img_id = img_id
                x0 = round(x0)
                y0 = round(y0)
                x1 = round(x1)
                y1 = round(y1)
                w = round(w)
                h = round(h)
                prob = float(prob)

                if prob < args.prob_threshold:
                    continue
                #  x0 = f"{x0:.3f}"
                #  y0 = f"{y0:.3f}"
                #  w = f"{w:.3f}"
                #  h = f"{h:.3f}"
                #  score = f"{score:.3f}"

                area = w * h
                ann = {
                    'id': img_id,
                    'image_id': orig_img_id,
                    'area': area,
                    'bbox': [x0, y0, w, h],
                    'category_id': cat_id,
                    'score': prob,
                    'iscrowd': 0,
                    'segmentation': []
                }
                final_predictions.append(ann)

                line = f"{id2label[cat_id]},{img_id},{prob},{x0},{y0},{x0+w},{y0+h},{prob:.4f}\n"
                # logger.debug(f"{line}")
                cat_objects[cat_id].append(
                    (int(x0), int(y0), int(x1), int(y1), prob))

        img_result = []
        for k in range(1, num_classes + 1):
            objs = cat_objects.get(k, [])
            img_result.append(objs)
        final_results.append(img_result)

    #  final_results = sorted(final_results, key=lambda x: x['image_id'])
    json.dump(final_results,
              open(submission_file, 'w'),
              ensure_ascii=False,
              indent=4,
              cls=NpEncoder)
    logger.warning(f"Saved {len(final_results)} results in {submission_file}")

    final_predictions = sorted(final_predictions, key=lambda x: x['id'])
    test_json_data['annotations'] = final_predictions
    json.dump(test_json_data,
              open(prediction_file, 'w'),
              ensure_ascii=False,
              indent=4,
              cls=NpEncoder)
    logger.warning(
        f"Saved {len(final_predictions)} annotations in {prediction_file}")

    return final_results


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=distributed,
                                   shuffle=False)

    if args.predict:
        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model,
                                     args.checkpoint,
                                     map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show,
                                      args.show_dir, args.show_score_thr)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.predict:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
        if args.submit:
            if args.out:
                outputs = mmcv.load(args.out)
                generate_submission(args, outputs, data_loader)
                #  logger.info(
                #      f"Saved {len(final_results)} bboxes in {submission_file}")

        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
