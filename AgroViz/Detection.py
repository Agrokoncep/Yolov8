from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("D:\\Datasets\\Agrokoncept\\Training\\exalted-durian-8\\best.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

# from ndarray
im2 = cv2.imread("depositphotos_336989494-stock-photo-tomatoes-growing-in-a-greenhouse.jpg")
results = model.predict(source=im2, save=True, save_txt=True, )  # save predictions as labels
