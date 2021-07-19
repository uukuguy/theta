#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, re
from glob import glob
from tqdm import tqdm
from loguru import logger
from PIL import Image
from coco_data import CocoData

cat2id = {
    'echinus': 1,
    'holothurian': 2,
    'scallop': 3,
    'starfish': 4,
    'waterweeds': 5
}

categories = []
for c, c_id in cat2id.items():
    categories.append({"id": c_id, "name": c, "supercategory": "none"})


def generate_det_coco_data():
    images = []
    annotations = []
    ann_id = 0
    img_dirs = ['./rawdata/train/image']
    for img_dir in img_dirs:
        img_files = glob(f"{img_dir}/*.jpg")

        for img_file in img_files:
            img = Image.open(img_file)
            img_width, img_height = img.size

            img_file = os.path.basename(img_file)
            img_id = img_file.split('.')[0][-6:]
            assert len(img_id) == 6

            xml_file = re.sub("\.jpg", ".xml", img_file)
            xml_file = f"{img_dir}/../box/{xml_file}"

            if not os.path.exists(xml_file):
                logger.warning(f"{xml_file} does not exists. SKIP.")
                continue

            anns = []

            from lxml import etree
            xml_data = etree.parse(xml_file)
            cats = xml_data.xpath("//object/name/text()")
            x0s = xml_data.xpath("//object/bndbox/xmin/text()")
            y0s = xml_data.xpath("//object/bndbox/ymin/text()")
            x1s = xml_data.xpath("//object/bndbox/xmax/text()")
            y1s = xml_data.xpath("//object/bndbox/ymax/text()")

            assert len(cats) == len(x0s) == len(y0s) == len(x1s) == len(
                y1s), f"{xml_file}, {cats}, {x0s}, {y0s}, {x1s}, {y1s}"

            for c, x0, y0, x1, y1 in zip(cats, x0s, y0s, x1s, y1s):
                cat_id = cat2id[c]
                x0 = int(float(x0))
                y0 = int(float(y0))
                x1 = int(float(x1))
                y1 = int(float(y1))
                w = x1 - x0
                h = y1 - y0
                area = w * h

                anns.append({
                    "id": ann_id,
                    "image_id": int(img_id),
                    "category_id": cat_id,
                    "area": area,
                    "bbox": [x0, y0, w, h],
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1

            if anns:
                annotations.extend(anns)

                images.append({
                    "id": int(img_id),
                    "width": img_width,
                    "height": img_height,
                    "file_name": os.path.basename(img_file)
                })

    coco_json_data = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    logger.warning(f"{len(images)} images, {len(annotations)} annotations")

    return coco_json_data


def generate_test_coco_data():
    img_files = glob(f"rawdata/test-A-image/*.jpg")
    img_files = sorted(img_files)
    images = []
    for img_file in tqdm(img_files):
        img = Image.open(img_file)
        img_width, img_height = img.size

        img_file = os.path.basename(img_file)
        img_id = img_file.split('.')[0][-6:]
        assert len(img_id) == 6

        images.append({
            "id": int(img_id),
            "width": img_width,
            "height": img_height,
            "file_name": os.path.basename(img_file)
        })

    annotations = []
    coco_json_data = {
        "categories": categories,
        "images": images,
        "annotations": annotations
    }
    logger.warning(f"{len(images)} images, {len(annotations)} annotations")

    return coco_json_data


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--generate_train_coco", action="store_true")
    parser.add_argument("--generate_test_coco", action="store_true")

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    if args.generate_train_coco:
        det_coco_json_data = generate_det_coco_data()
        det_coco_file = "./det_coco.json"
        json.dump(det_coco_json_data,
                  open(det_coco_file, 'w'),
                  ensure_ascii=False,
                  indent=2)
        logger.warning(f"Saved {det_coco_file}")

        coco_data = CocoData()
        coco_data.from_file(det_coco_file)
        train_coco_data, val_coco_data = coco_data.split(train_ratio=0.9)

        os.makedirs("train/annotations", exist_ok=True)
        train_coco_data.save(f"train/annotations/train_coco.json")
        val_coco_data.save(f"train/annotations/val_coco.json")

    if args.generate_test_coco:
        os.makedirs("test/annotations", exist_ok=True)
        test_coco_json_data = generate_test_coco_data()
        test_coco_file = "./test/annotations/test_coco.json"
        json.dump(test_coco_json_data,
                  open(test_coco_file, 'w'),
                  ensure_ascii=False,
                  indent=2)
        logger.warning(f"Saved {test_coco_file}")


if __name__ == '__main__':
    main()
