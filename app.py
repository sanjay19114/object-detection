import cv2
from ultralytics import YOLO

model = YOLO('yolov5s.pt')
class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
    33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

print("Model classes:", class_names)

webcamera = cv2.VideoCapture(0)
if not webcamera.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Optionally set the resolution of the webcam
# webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Capture frame-by-frame from the webcam
    success, frame = webcamera.read()
    if not success:
        print("Failed to capture image")
        break

    
    results = model(frame, classes=None, conf=0.8, imgsz=480)  # Set classes=None to detect all classes

    
    num_boxes = len(results[0].boxes) if results else 0
    cv2.putText(frame, f"Total: {num_boxes}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

   
    for box in results[0].boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Get the class id and confidence
        class_id = int(box.cls)
        confidence = float(box.conf)  # Convert Tensor to float

        class_name = class_names[class_id]

        # Draw the bounding box and label on the frame
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the annotated frame in a window
    cv2.imshow("Live Camera", frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break


webcamera.release()
cv2.destroyAllWindows()  
