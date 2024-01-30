from ultralytics import YOLO
import wandb

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 10
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{num_freeze} layers are freezed.")

if __name__ == '__main__':
    wandb.login(key="17db354b20a284ceacef610b63e16effc00b0ff3")
    wandb.init(project="agrokoncep-tomato")
    # Load a model
    model = YOLO('yolov8m.pt')
    model.add_callback("on_train_start", freeze_layer)
    # Load custom datasets :
    dataset_path = "E:\\Batch#1\\data.yaml"

    # Train the model
    result_train = model.train(data=dataset_path, epochs=300, imgsz=1280, batch=8, workers=2,
                               name="test_tomatoes", lr0=0.05,device="0", patience=75)
    # Evaluate the model's performance on the validation set
    results_val = model.val()