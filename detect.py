import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("test_no_helmet.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (960, 540))
    results = model(frame, verbose=False)
    annotated = results[0].plot()
    cv2.imshow("Helmet Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()