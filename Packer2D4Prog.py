import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *
import datetime

def left_click_detect(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"\tClick on {x}, {y}")

# Load YOLO model and tracker
model = YOLO('./best.pt')
tracker = Tracker()

count = 0
cy1 = 560
cy2 = 695
offset = 5
ids = []

# Open the video file
cap = cv2.VideoCapture('Packer2D4.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the VideoWriter to save the output
output_file = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    result = results[0]
    output_image = frame

    # Extract bounding boxes
    detections = result.boxes.data
    detections = detections.detach().cpu().numpy()
    px = pd.DataFrame(detections).astype("float")

    list = []
    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)
        cv2.circle(output_image, (cx, cy), 4, (0, 0, 255), -1)  # Draw center points of bounding box
        cv2.putText(output_image, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if cy1 <= cy <= cy2:
            cv2.rectangle(output_image, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
        if (cy == cy1 and cy < cy2) or (cy1 < (cy + offset) and cy1 > (cy - offset)):
            if id not in ids:
                count += 1
                ids.append(id)
                current_time = datetime.datetime.now().replace(microsecond=0)
                print("Count:", count, " at:", current_time)

        elif (cy == cy2) or (cy2 < (cy + offset) and cy2 > (cy - offset)):
            if id not in ids:
                count += 1
                ids.append(id)
                current_time = datetime.datetime.now().replace(microsecond=0)
                print("Count:", count, " at:", current_time)

    # Draw the counting lines and text
    text_color = (255, 0, 0)
    red_color = (0, 0, 255)
    count_color = (0, 0, 255)

    cv2.line(output_image, (540, cy1), (900, cy1), red_color, 2)
    cv2.putText(output_image, 'Counter Line 1', (520, cy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(output_image, (400, cy2), (900, cy2), red_color, 2)
    cv2.putText(output_image, 'Counter Line 2', (380, cy2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.putText(output_image, f'Count: {count}', (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, count_color, 4, cv2.LINE_AA)

    # Save the frame to the output video
    out.write(output_image)

    # Show the frame
    cv2.imshow("RESULT", output_image)
    cv2.setMouseCallback('RESULT', left_click_detect)

    # Exit on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
out.release()  # Save the video file
cv2.destroyAllWindows()
