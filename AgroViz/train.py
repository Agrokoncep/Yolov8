from ultralytics import YOLO
import wandb
import os
import config

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
    wandb.login(key=config.wandb.get("api_key"))
    wandb.init(project=config.wandb.get("project_name"))
    # Load a model
    model = YOLO(config.model_size)
    model.add_callback("on_train_start", freeze_layer)

    print(f"Dataset path: {config.dataset_abs_path}")
    # Train the model
    result_train = model.train(**config.train_kwargs)
    # Evaluate the model's performance on the validation set
    results_val = model.val()