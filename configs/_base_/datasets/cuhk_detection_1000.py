dataset_type = 'CuhkDataset'
# change to you own path
data_root = '/home/cvlab3/Downloads/AlignPS/demo/anno/kist/'
# /home/cvlab3/Downloads/AlignPS/demo/anno
# '/home/yy1/2021/data/cuhk/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_ids']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        img_scale=(1000, 600),
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
data_name = 'images_x1_cam/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_pid_new.json', # change to you own path
        img_prefix='/home/cvlab3/Downloads/WRCAN-PyTorch/src/'+data_name,
        # /home/cvlab3/Downloads/WRCAN-PyTorch/src/images data_root + 'Image/SSM/'
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test_new.json', # change to you own path
        img_prefix='/home/cvlab3/Downloads/WRCAN-PyTorch/src/'+data_name,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_new.json', # change to you own path
        img_prefix='/home/cvlab3/Downloads/WRCAN-PyTorch/src/'+data_name,
        proposal_file=data_root+'TestG50.mat',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
