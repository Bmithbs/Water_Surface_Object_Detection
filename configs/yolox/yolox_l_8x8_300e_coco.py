_base_ = './yolox_s_8x8_300e_coco.py'

# model settings
model = dict(
    random_size_range = (5, 15),
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))

img_scale = (320, 320)
max_epochs = 200
num_last_epochs = 15
interval = 10
load_from = '/root/HSK/Water_Surface_Object_detection/work_dirs/yolox_config/latest.pth'
data = dict(
    samples_per_gpu=8,
)
resume_from = None
