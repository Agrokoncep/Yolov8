from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

model = YOLO('resources/Agrokoncept_model_cluster.pt')
dataset = 'resources/Benchmark/data.yaml'

results = model.predict(source="resources/20231218_103237.mp4", save=True, save_txt=False, iou=0.5, conf=0.6,agnostic_nms=True)
# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks