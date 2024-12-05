import torch
import cv2
import time  # For FPS calculation

def load_model(weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load the YOLO model from the specified weights file.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True, trust_repo=True)
    model.to(device)
    return model

def run_inference(model, image, conf_threshold=0.25):
    """
    Run inference on a single image using the YOLO model.
    """
    results = model(image)
    results = results.pandas().xyxy[0]  # Pandas DataFrame of results
    filtered_results = results[results['confidence'] >= conf_threshold]  # Apply confidence threshold
    return filtered_results

def visualize_results(image, results, fps):
    """
    Visualize the results on the image, including the FPS display.
    """
    for _, row in results.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Display FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

weights_path = '/home/jetson-gmi/Downloads/osomdataset/best.pt'

# Load model
print("Loading model...")
model = load_model(weights_path)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("Starting real-time detection... Press 'q' to quit.")

while True:
    start_time = time.time()  # Start time for FPS calculation
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference
    results = run_inference(model, frame)
    
    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # Visualize results on the frame
    annotated_frame = visualize_results(frame, results, fps)

    # Display the annotated frame
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

