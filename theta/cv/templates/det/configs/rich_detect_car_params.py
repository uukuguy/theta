num_gpus = 1
#  samples_per_gpu = 1
samples_per_gpu = 2
#  samples_per_gpu = 4
#  samples_per_gpu = 8
workers_per_gpu = 4

max_epochs = 12
warmup_steps = [8, 11]
#  max_epochs = 20
#  warmup_steps = [16, 19]
#  max_epochs = 20
#  warmup_steps = [8, 12, 18]
#  max_epochs = 24
#  warmup_steps = [16, 22]
#  max_epochs = 36
#  warmup_steps = [24, 33]

depth = 50
num_train_samples = int(2000 * 0.9)

img_train_scale_list = (1280, 720)
img_test_scale_list = (1280, 720)

# 多尺度预测
#  img_train_scale_list = [(640, 480), (960, 720)]
#  img_test_scale_list = [(640, 480), (800, 600), (960, 720)]

#  img_train_scale_list = [(960, 640), (1333, 800)]
#  img_test_scale_list = [(960, 640), (1100, 700), (1333, 800)]

#  img_train_scale_list = [(1280, 720), (1920, 1080)]
#  img_test_scale_list = [(1280, 720), (1600, 960), (1920, 1080)]

#  img_train_scale_list = [(1333, 800), (2666, 1600)]
#  img_test_scale_list = [(1333, 800), (2000, 1200), (2666, 1600)]

data_root = './data/'

classes = ('phone', 'pad', 'laptop', 'wallet', 'packsack')
num_classes = len(classes)

_base_ = ['./cascade_rcnn_fpn_coco.py']

#  fp16 = dict(loss_scale=8.)

weights_dir = "../weights"
load_from = f"{weights_dir}/cascade_rcnn_r50_fpn_1x_coco_classes_{num_classes+1}.pth"
# load_from = f"{weights_dir}/cascade_rcnn_r101_fpn_20e_coco_classes_{num_classes+1}.pth"
# load_from = f"{weights_dir}/cascade_rcnn_x101_64x4d_fpn_20e_coco_classes_{num_classes+1}.pth"
