#  num_gpus = 1
#  samples_per_gpu = 8
#  workers_per_gpu = 4
#  max_epochs = 20
#  warmup_steps = [16, 19]
#  num_train_samples = int(8000 * 0.9)
#  data_root = './data/'
#  img_train_scale_list = [(640, 480), (960, 720)]
#  img_test_scale_list = [(640, 480), (800, 600), (960, 720)]
#  #  fp16 = dict(loss_scale=8.)
# depth = 50
#  load_from = "../weights/cascade_rcnn_r50_fpn_1x_coco_classes_13.pth"
#  #  load_from = "../weights/cascade_rcnn_r101_fpn_20e_coco_classes_7.pth"
#  # load_from = "../weights/cascade_rcnn_x101_64x4d_fpn_20e_coco_classes_7.pth"

gpu_assign_thr = 100
train_ann_file = data_root + 'train/annotations/train_coco.json'
train_img_prefix = data_root + "train/images/"
val_ann_file = data_root + 'train/annotations/val_coco.json'
val_img_prefix = data_root + "train/images/"
test_ann_file = data_root + 'test/annotations/test_coco.json'
test_img_prefix = data_root + "test/images/"

#  classes = ('knife', 'scissors', 'sharpTools', 'expandableBaton',
#             'smallGlassBottle', 'electricBaton', 'plasticBeverageBottle',
#             'plasticBottleWithaNozzle', 'electronicEquipment', 'battery',
#             'seal', 'umbrella')
num_classes = len(classes)

# -------------------- model settings --------------------
# mmdetection/configs/_base_/models/cascade_rcnn_r50_fpn.py
model = dict(
    type='CascadeRCNN',
    pretrained=None,  # 'torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=depth,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # 在最后三个block加入可变形卷积
        # dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        # stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(type='FPN',
              in_channels=[256, 512, 1024, 2048],
              out_channels=256,
              num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            #   ratios=[0.5, 1.0, 2.0],
            ratios=[0.2, 0.5, 1.0, 2.0, 5.0],  # 添加了0.2, 5.0
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                        target_means=[.0, .0, .0, .0],
                        target_stds=[1.0, 1.0, 1.0, 1.0]),
        #   loss_cls=dict(type='CrossEntropyLoss',
        #                 use_sigmoid=True,
        #                 loss_weight=1.0),
        # 修改了loss，为了调控难易样本与正负样本比例
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        # reg_decoded_bbox=True,      # 使用GIoUI时注意添加
        # loss_bbox=dict(type='GIoULoss', loss_weight=5.0)
    ),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(type='SingleRoIExtractor',
                                roi_layer=dict(type='RoIAlign',
                                               output_size=7,
                                               sampling_ratio=0),
                                out_channels=256,
                                featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                target_means=[0., 0., 0., 0.],
                                target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0),
                #  loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                reg_decoded_bbox=True,  # 使用GIoUI时注意添加
                loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                target_means=[0., 0., 0., 0.],
                                target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0),
                #  loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                reg_decoded_bbox=True,  # 使用GIoUI时注意添加
                loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
            ),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                                target_means=[0., 0., 0., 0.],
                                target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss',
                              use_sigmoid=False,
                              loss_weight=1.0),
                #  loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                reg_decoded_bbox=True,  # 使用GIoUI时注意添加
                loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
            )
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner',
                               pos_iou_thr=0.7,
                               neg_iou_thr=0.3,
                               min_pos_iou=0.3,
                               gpu_assign_thr=gpu_assign_thr,
                               match_low_quality=True,
                               ignore_iof_thr=-1),
                 sampler=dict(type='RandomSampler',
                              num=256,
                              pos_fraction=0.5,
                              neg_pos_ub=-1,
                              add_gt_as_proposals=False),
                 allowed_border=0,
                 pos_weight=-1,
                 debug=False),
        rpn_proposal=dict(nms_pre=2000,
                          max_per_img=2000,
                          nms=dict(type='nms', iou_threshold=0.7),
                          min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(type='MaxIoUAssigner',
                              pos_iou_thr=0.4,
                              neg_iou_thr=0.4,
                              min_pos_iou=0.4,
                              gpu_assign_thr=gpu_assign_thr,
                              match_low_quality=False,
                              ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',  # 解决难易样本，也解决了正负样本比例问题。
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(assigner=dict(type='MaxIoUAssigner',
                               pos_iou_thr=0.5,
                               neg_iou_thr=0.5,
                               min_pos_iou=0.5,
                               gpu_assign_thr=gpu_assign_thr,
                               match_low_quality=False,
                               ignore_iof_thr=-1),
                 sampler=dict(type='OHEMSampler',
                              num=512,
                              pos_fraction=0.25,
                              neg_pos_ub=-1,
                              add_gt_as_proposals=True),
                 pos_weight=-1,
                 debug=False),
            dict(assigner=dict(type='MaxIoUAssigner',
                               pos_iou_thr=0.6,
                               neg_iou_thr=0.6,
                               min_pos_iou=0.6,
                               gpu_assign_thr=gpu_assign_thr,
                               match_low_quality=False,
                               ignore_iof_thr=-1),
                 sampler=dict(type='OHEMSampler',
                              num=512,
                              pos_fraction=0.25,
                              neg_pos_ub=-1,
                              add_gt_as_proposals=True),
                 pos_weight=-1,
                 debug=False)
        ],
        stage_loss_weights=[1, 0.5, 0.25]),
    test_cfg=dict(
        rpn=dict(nms_pre=1000,
                 max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7),
                 min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            # nms=dict(type='nms', iou_threshold=0.5),
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100)))

# -------------------- dataset settings --------------------
# mmdetection/configs/_base_/datasets/coco_detection.py
dataset_type = 'CocoDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

mixup_train_transforms = dict(type='MixUp', p=0.5, lambd=0.5)

#  img_scale_list = [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                                 (736, 1333), (768, 1333), (800, 1333)]

#  img_scale_list = [(1333, 480), (1333, 512), (1333, 544), (1333, 576),
#                    (1333, 608), (1333, 640), (1333, 672), (1333, 704),
#                    (1333, 736), (1333, 768), (1333, 800)]
#  img_scale_list = [(640, 480), (960, 720)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    mixup_train_transforms,  # 采用MixUp
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    #  dict(type='Resize', img_scale=[(1333, 800), (2666, 1600)],
    #  dict(type='Resize', img_scale=[(960, 640), (1333, 800)], keep_ratio=True),
    #  dict(type='Resize', img_scale=[(1280, 720), (1920, 1080)],
    #       keep_ratio=True),

    #  dict(type='Resize', img_scale=[(640, 480), (960, 720)], keep_ratio=True),
    #  dict(type='RandomCrop',
    #       crop_type='absolute_range',
    #       crop_size=(360, 360),
    #       allow_negative_crop=True),
    #  dict(type='Resize', img_scale=[(640, 480), (960, 720)], keep_ratio=True),
    dict(type='Resize', img_scale=img_train_scale_list, keep_ratio=True),
    #  dict(
    #      type='AutoAugment',
    #      policies=[
    #          [
    #              dict(
    #                  type='Resize',
    #                  #  img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #                  #             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
    #                  #             (736, 1333), (768, 1333), (800, 1333)],
    #                  img_scale=img_scale_list,
    #                  multiscale_mode='value',
    #                  keep_ratio=True)
    #          ],
    #          [
    #              dict(
    #                  type='Resize',
    #                  #  img_scale=[(400, 1333), (500, 1333), (600, 1333)],
    #                  #  img_scale=[(1333, 400, 1333), (1333, 500), (1333, 600)],
    #                  img_scale=img_scale_list,
    #                  multiscale_mode='value',
    #                  keep_ratio=True),
    #              dict(
    #                  type='RandomCrop',
    #                  crop_type='absolute_range',
    #                  crop_size=(360, 360),
    #                  #  crop_size=(384, 600),
    #                  #  crop_size=(600, 384),
    #                  allow_negative_crop=True),
    #              dict(
    #                  type='Resize',
    #                  #  img_scale=[(480, 1333), (512, 1333), (544, 1333),
    #                  #             (576, 1333), (608, 1333), (640, 1333),
    #                  #             (672, 1333), (704, 1333), (736, 1333),
    #                  #             (768, 1333), (800, 1333)],
    #                  img_scale=img_scale_list,
    #                  multiscale_mode='value',
    #                  override=True,
    #                  keep_ratio=True)
    #          ]
    #      ]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        #  img_scale=[(1333, 800), (2000, 1200), (2666, 1600)],  # 多尺度预测
        #  img_scale=[(960, 640), (1100, 700), (1333, 800)],  # 多尺度预测
        #  img_scale=[(1280, 720), (1600, 960), (1920, 1080)],  # 多尺度预测
        #  img_scale=[(640, 480), (800, 600), (960, 720)],  # 多尺度预测
        img_scale=img_test_scale_list,  # 多尺度预测
        #  img_scale=img_scale_list,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(samples_per_gpu=samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            train=dict(type=dataset_type,
                       classes=classes,
                       ann_file=train_ann_file,
                       img_prefix=train_img_prefix,
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     classes=classes,
                     ann_file=val_ann_file,
                     img_prefix=val_img_prefix,
                     pipeline=test_pipeline),
            test=dict(type=dataset_type,
                      classes=classes,
                      test_mode=True,
                      ann_file=test_ann_file,
                      img_prefix=test_img_prefix,
                      pipeline=test_pipeline))

# -------------------- scheduler settings --------------------
# mmdetection/configs/_base_/schedules/schedule_1x.py

# optimizer
#  fp16 = dict(loss_scale=8.)
# fp16_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
optimizer = dict(type='SGD',
                 lr=0.00125 * num_gpus * samples_per_gpu,
                 momentum=0.9,
                 weight_decay=0.0001)

#  optimizer = dict(
#      type='AdamW',
#      lr=0.00125 * num_gpus * samples_per_gpu,
#      betas=(0.9, 0.999),
#      weight_decay=0.05,
#      paramwise_cfg=dict(
#          custom_keys={
#              'absolute_pos_embed': dict(decay_mult=0.),
#              'relative_position_bias_table': dict(decay_mult=0.),
#              'norm': dict(decay_mult=0.)
#          }))

# optimizer_config = dict(grad_clip=None)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=int(
                     (num_train_samples / samples_per_gpu) * max_epochs / 10),
                 warmup_ratio=0.001,
                 step=warmup_steps)
#  step=[8, 11])
#  step=[16, 19])
#  step=[24, 33])
#  step=[16, 22])
#  step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

# -------------------- scheduler settings --------------------
# mmdetection/configs/_base_/default_runtime.py
checkpoint_config = dict(interval=1, by_epoch=True, max_keep_ckpts=3)
evaluation = dict(interval=1, save_best='auto')
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
#  load_from = "../weights/cascade_rcnn_r50_fpn_1x_coco_classes_13.pth"
#  load_from = "../weights/cascade_rcnn_r101_fpn_20e_coco_classes_7.pth"
# load_from = "../weights/cascade_rcnn_x101_64x4d_fpn_20e_coco_classes_7.pth"
resume_from = None
workflow = [('train', 1)]
