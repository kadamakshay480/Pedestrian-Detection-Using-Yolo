import numpy as np
import cv2
import os
import imutils
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

def pedestrian_detection(frame, model, layer_names, labels):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                  swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_names)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == labels.index("person") and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)

    return results

def select_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        detect_button.config(state=tk.NORMAL)
        process_image(image_path)

def process_image(image_path):
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=700)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    panel.config(image=photo)
    panel.image = photo

def detect_pedestrians():
    if image_path:
        image = cv2.imread(image_path)
        image = imutils.resize(image, width=700)
        results = pedestrian_detection(image, model, layer_names, LABELS)

        for res in results:
            (x, y, x_plus_w, y_plus_h) = res[1]
            cv2.rectangle(image, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image)
        panel.config(image=photo)
        panel.image = photo

def select_video():
    global video_path
    video_path = filedialog.askopenfilename()
    if video_path:
        detect_button.config(state=tk.NORMAL)
        process_video()

def process_video():
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=700)
        results = pedestrian_detection(frame, model, layer_names, LABELS)

        for res in results:
            (x, y, x_plus_w, y_plus_h) = res[1]
            cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)

        cv2.imshow("Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()



root = tk.Tk()
root.title("Pedestrian Detection")

image_path = ""
video_path = ""

select_image_button = tk.Button(root, text="Select Image", command=select_image)
select_image_button.pack(pady=5)

select_video_button = tk.Button(root, text="Select Video", command=select_video)
select_video_button.pack(pady=5)

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

detect_button = tk.Button(root, text="Detect Pedestrians", command=detect_pedestrians, state=tk.DISABLED)
detect_button.pack(pady=5)

# Define paths to YOLO configuration, weights, and labels
weights_path = "C:/Age Gender Detection/Pedestrian/yolov4-tiny.weights"
config_path = "C:/Age Gender Detection/Pedestrian/yolov4-tiny.cfg"
labelsPath = "C:/Age Gender Detection/Pedestrian/coco.names"

# Load YOLO model
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_names = model.getLayerNames()
layer_indices = model.getUnconnectedOutLayers()

# Ensure layer_indices is a list
if not isinstance(layer_indices, list):
    layer_indices = [layer_indices]

layer_names = [layer_names[i[0] - 1] for i in layer_indices]

# Load labels
LABELS = open(labelsPath).read().strip().split("\n")

root.mainloop()
