default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(
        type='PPYOLOEParamSchedulerHook',
        warmup_min_iter=1000,
        start_factor=0.0,
        warmup_epochs=5,
        min_lr_ratio=0.0,
        total_epochs=288),
    checkpoint=dict(
        type='CheckpointHook', interval=5, save_best='auto', max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'models/ppyoloe_plus_s_fast_8xb8-240e_ms_sar/epoch_165.pth'
resume = False
file_client_args = dict(backend='disk')
_file_client_args = dict(backend='disk')
tta_model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300))
img_scales = [(480, 480), (640, 640), (960, 960), (1024, 1024)]
_multiscale_resize_transforms = [
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(480, 480)),
            dict(
                type='LetterResize',
                scale=(480, 480),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(640, 640)),
            dict(
                type='LetterResize',
                scale=(640, 640),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(960, 960)),
            dict(
                type='LetterResize',
                scale=(960, 960),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ]),
    dict(
        type='Compose',
        transforms=[
            dict(type='YOLOv5KeepRatioResize', scale=(1024, 1024)),
            dict(
                type='LetterResize',
                scale=(1024, 1024),
                allow_scale_up=False,
                pad_val=dict(img=114))
        ])
]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (480, 480)
            }, {
                'type': 'LetterResize',
                'scale': (480, 480),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (640, 640)
            }, {
                'type': 'LetterResize',
                'scale': (640, 640),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (960, 960)
            }, {
                'type': 'LetterResize',
                'scale': (960, 960),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }, {
            'type':
            'Compose',
            'transforms': [{
                'type': 'YOLOv5KeepRatioResize',
                'scale': (1024, 1024)
            }, {
                'type': 'LetterResize',
                'scale': (1024, 1024),
                'allow_scale_up': False,
                'pad_val': {
                    'img': 114
                }
            }]
        }],
                    [{
                        'type': 'mmdet.RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'mmdet.RandomFlip',
                        'prob': 0.0
                    }], [{
                        'type': 'mmdet.LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'mmdet.PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'pad_param', 'flip', 'flip_direction')
                    }]])
]
data_root = 'data/sar/'
dataset_type = 'YOLOv5CocoDataset'
img_scale = (640, 640)
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 240
num_classes = 7
save_epoch_intervals = 5
train_batch_size_per_gpu = 4
train_num_workers = 8
val_batch_size_per_gpu = 4
val_num_workers = 8
persistent_workers = True
base_lr = 0.001
strides = [8, 16, 32]
model = dict(
    type='mmdet.DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=300),
    module=dict(
        type='YOLODetector',
        data_preprocessor=dict(
            type='PPYOLOEDetDataPreprocessor',
            pad_size_divisor=32,
            batch_augments=[
                dict(
                    type='PPYOLOEBatchRandomResize',
                    random_size_range=(640, 1024),
                    interval=1,
                    size_divisor=32,
                    random_interp=True,
                    keep_ratio=False)
            ],
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True),
        backbone=dict(
            type='PPYOLOECSPResNet',
            deepen_factor=0.33,
            widen_factor=0.5,
            block_cfg=dict(
                type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-05),
            act_cfg=dict(type='SiLU', inplace=True),
            attention_cfg=dict(
                type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
            use_large_stem=True),
        neck=dict(
            type='PPYOLOECSPPAFPN',
            in_channels=[256, 512, 1024],
            out_channels=[192, 384, 768],
            deepen_factor=0.33,
            widen_factor=0.5,
            num_csplayer=1,
            num_blocks_per_layer=3,
            block_cfg=dict(
                type='PPYOLOEBasicBlock', shortcut=False, use_alpha=False),
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-05),
            act_cfg=dict(type='SiLU', inplace=True),
            drop_block_cfg=None,
            use_spp=True),
        bbox_head=dict(
            type='PPYOLOEHead',
            head_module=dict(
                type='PPYOLOEHeadModule',
                num_classes=7,
                in_channels=[192, 384, 768],
                widen_factor=0.5,
                featmap_strides=[8, 16, 32],
                reg_max=16,
                norm_cfg=dict(type='BN', momentum=0.1, eps=1e-05),
                act_cfg=dict(type='SiLU', inplace=True),
                num_base_priors=1),
            prior_generator=dict(
                type='mmdet.MlvlPointGenerator',
                offset=0.5,
                strides=[8, 16, 32]),
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            loss_cls=dict(
                type='mmdet.VarifocalLoss',
                use_sigmoid=True,
                alpha=0.75,
                gamma=2.0,
                iou_weighted=True,
                reduction='sum',
                loss_weight=1.0),
            loss_bbox=dict(
                type='IoULoss',
                iou_mode='giou',
                bbox_format='xyxy',
                reduction='mean',
                loss_weight=2.5,
                return_iou=False),
            loss_dfl=dict(
                type='mmdet.DistributionFocalLoss',
                reduction='mean',
                loss_weight=0.125)),
        train_cfg=dict(
            initial_epoch=30,
            initial_assigner=dict(
                type='BatchATSSAssigner',
                num_classes=7,
                topk=9,
                iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=7,
                topk=13,
                alpha=1,
                beta=6,
                eps=1e-09)),
        test_cfg=dict(
            multi_label=True,
            nms_pre=1000,
            score_thr=0.01,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300)))
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PPYOLOERandomDistort'),
    dict(type='mmdet.Expand', mean=(103.53, 116.28, 123.675)),
    dict(type='PPYOLOERandomCrop'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate', use_ms_training=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/sar/',
        ann_file='train.json',
        data_prefix=dict(img='train_img/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PPYOLOERandomDistort'),
            dict(type='mmdet.Expand', mean=(103.53, 116.28, 123.675)),
            dict(type='PPYOLOERandomCrop'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ]))
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='mmdet.FixShapeResize',
        width=640,
        height=640,
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/sar/',
        test_mode=True,
        data_prefix=dict(img='test_img/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file='test.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(
                type='mmdet.FixShapeResize',
                width=640,
                height=640,
                keep_ratio=False,
                interpolation='bicubic'),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
test_dataloader = dict(
    batch_size=20,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='',
        test_mode=True,
        data_prefix=dict(img='/test_img'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file='test.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(
                type='TestTimeAug',
                transforms=[[{
                    'type':
                    'Compose',
                    'transforms': [{
                        'type': 'YOLOv5KeepRatioResize',
                        'scale': (480, 480)
                    }, {
                        'type': 'LetterResize',
                        'scale': (480, 480),
                        'allow_scale_up': False,
                        'pad_val': {
                            'img': 114
                        }
                    }]
                }, {
                    'type':
                    'Compose',
                    'transforms': [{
                        'type': 'YOLOv5KeepRatioResize',
                        'scale': (640, 640)
                    }, {
                        'type': 'LetterResize',
                        'scale': (640, 640),
                        'allow_scale_up': False,
                        'pad_val': {
                            'img': 114
                        }
                    }]
                }, {
                    'type':
                    'Compose',
                    'transforms': [{
                        'type': 'YOLOv5KeepRatioResize',
                        'scale': (960, 960)
                    }, {
                        'type': 'LetterResize',
                        'scale': (960, 960),
                        'allow_scale_up': False,
                        'pad_val': {
                            'img': 114
                        }
                    }]
                }, {
                    'type':
                    'Compose',
                    'transforms': [{
                        'type': 'YOLOv5KeepRatioResize',
                        'scale': (1024, 1024)
                    }, {
                        'type': 'LetterResize',
                        'scale': (1024, 1024),
                        'allow_scale_up': False,
                        'pad_val': {
                            'img': 114
                        }
                    }]
                }],
                            [{
                                'type': 'mmdet.RandomFlip',
                                'prob': 1.0
                            }, {
                                'type': 'mmdet.RandomFlip',
                                'prob': 0.0
                            }],
                            [{
                                'type': 'mmdet.LoadAnnotations',
                                'with_bbox': True
                            }],
                            [{
                                'type':
                                'mmdet.PackDetInputs',
                                'meta_keys':
                                ('img_id', 'img_path', 'ori_shape',
                                 'img_shape', 'scale_factor', 'pad_param',
                                 'flip', 'flip_direction')
                            }]])
        ]))
param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=False),
    paramwise_cfg=dict(norm_decay_mult=0.0))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='data/sar/test.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='test.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='output.json')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=240, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
launcher = 'none'
work_dir = './work_dirs/ppyoloe_plus_s_fast_8xb8-240e_no-val_ms_sar'
