from ultralytics import YOLO
import wandb
import os
import config

def freeze_layer(trainer):
    model = trainer.model
    config.nb_freezed_layers = 5
    print(f"Freezing {config.nb_freezed_layer} layers")
    freeze = [f'model.{x}.' for x in range(config.nb_freezed_layer)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False
    print(f"{config.nb_freezed_layer} layers are freezed.")

if __name__ == '__main__':
    wandb.login(key=config.wandb.get("api_key"))
    wandb.init(project=config.wandb.get("project_name"), 
               notes=f"Dataset={config.data.get('dataset_name')} // model_size={config.model_size} // lr0={config.train_kwargs.get('lr0')} // batch={config.train_kwargs.get('batch')} // epochs={config.train_kwargs.get('epochs')} // workers={config.train_kwargs.get('workers')} // n_freeze_layers={config.nb_freezed_layers}",
    )
    # Load a model
    model = YOLO(config.model_size)
    if config.nb_freezed_layers > 0:
        model.add_callback("on_train_start", freeze_layer)

    print(f"Dataset path: {config.dataset_abs_path}")
    # Train the model
    result_train = model.train(**config.train_kwargs)
    # Evaluate the model's performance on the validation set
    results_val = model.val()