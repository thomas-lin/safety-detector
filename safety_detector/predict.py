import os

import cv2
from ultralytics import YOLO


def predict():
    app_path = os.path.join(os.path.dirname(__file__), '..')
    model_path = os.path.join(app_path, 'runs/detect/train3/weights/best.pt')
    stream_path = [
        "https://reac.taipei:8443/hls/S11CC-B8DF6B001533/index.m3u8",
        # "https://reac2.taipei:20443/p/109f-cecf1c/live/index.m3u8",
        # "https://reac2.taipei:20443/p/109e-1ab86b/live/index.m3u8",
        # "https://reac.taipei:8443/hls/S11CC-B8DF6B0013D5/index.m3u8",
        # 'https://reac.taipei:8443/hls/S11CC-B8DF6B0013CD/index.m3u8',
        # 'https://reac.taipei:8443/hls/S11CC-B8DF6B0014E6/index.m3u8',
        # 'https://reac2.taipei:20443/p/109d-b204dd/live/index.m3u8',
    ]

    cap = cv2.VideoCapture(stream_path[0])
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO(model_path)

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Perform object detection on an image
            result, *tail = model(frame)

            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB output

            annotated_frame = result.plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
