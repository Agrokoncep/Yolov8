import os

MODEL_SIZE = dict(
    NANO = "yolov8n.pt",
    SMALL = "yolov8s.pt",
    MEDIUM = "yolov8m.pt",
    LARGE = "yolov8l.pt",
    XLARGE = "yolov8x.pt",
)

wandb = dict(
    api_key = '0a172795af06b84dd255b5192d3722a9d098af32',
    project_name = 'batch1',
)

data = dict(
    batch_datasets_folder = "C:\\Users\\gblai\\Documents\\github\\AgroKoncep\\Dataset",
    dataset_name = "baseimg-batch1-augmented1",
    dataset_config_name = "data.yaml",
)

dataset_abs_path = os.path.join(data["batch_datasets_folder"], data["dataset_name"], data["dataset_config_name"])

model_size = MODEL_SIZE["MEDIUM"]  # Choose from MODEL_SIZE,

# https://docs.ultralytics.com/fr/usage/cfg/#augmentation
data_augmentation_kwargs = dict(
    hsv_h = 0.0,
    hsv_s = 0.0,
    hsv_v = 0.0,
    degrees = 0.0,
    translate = 0.0,
    scale = 0.0,
    shear = 0.0,
    perspective = 0.0,
    flipud = 0.0,
    fliplr = 0.0,
    mosaic = 0.0,
    mixup = 0.0,
    copy_paste = 0.0,
)

train_kwargs = dict(
    data = dataset_abs_path,
    name = f"{data['dataset_name']}_",
    epochs = 300,
    imgsz = 640,
    batch = -1,
    workers = 10,
    lr0 = 0.02,
    device = "0",
    patience = 75,
    **data_augmentation_kwargs
)
