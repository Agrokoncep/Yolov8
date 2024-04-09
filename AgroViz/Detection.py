from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO('resources/Agrokoncept_model_no_cluster.pt')  # load an official model
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from ndarray
vid = cv2.imread("resources/20231218_103237.mp4")
results = model.predict(source="resources/20231218_103237.mp4", save=True, save_txt=False, iou=0.5, conf=0.6,agnostic_nms=True)

print(results)