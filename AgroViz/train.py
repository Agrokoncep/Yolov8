from ultralytics import YOLO
import wandb

wandb.login(key="17db354b20a284ceacef610b63e16effc00b0ff3")
wandb.init(project="agrokoncept-tomates")
# Load a model
model = YOLO('yolov8n.pt')

# Load custom datasets :
dataset_path = "D:\\Datasets\\Agrokoncept\\TomatoesMergedYolov8\\tomatoes.yaml"


if __name__ == '__main__':
    # Train the model
    result_train = model.train(data=dataset_path, epochs=100, imgsz=640, batch=16, workers=2, name="test_tomatoes")
    # Evaluate the model's performance on the validation set
    results_val = model.val()