# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False,
                 warmup='linear', warmup_iters=4000, warmup_ratio=0.01)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=84000)
checkpoint_config = dict(by_epoch=False, interval=21000)
evaluation = dict(interval=4000, metric='mIoU')
