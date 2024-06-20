import os
import time
from threading import Event

import cv2
from ultralytics import YOLO

from .strapi_client import SafetyEvent, StrapiClient


def predict_direct():
    stream_path = [
        "https://reac.taipei:8443/hls/S11CC-B8DF6B0015ED/index.m3u8",
        # "https://reac.taipei:8443/hls/S11CC-B8DF6B001543/index.m3u8",
        # "https://reac2.taipei:10443/p/1007-e8eb3e/live/index.m3u8",
        # "https://reac2.taipei:20443/p/105d-99f106/live/index.m3u8",
        # "https://reac2.taipei:10443/p/1068-cf591f/live/index.m3u8",
        # "https://reac.taipei:8443/hls/S11CC-B8DF6B0015C7/index.m3u8",
        "https://reac2.taipei:10443/p/109f-cecf1c/live/index.m3u8",
        "https://reac.taipei:8443/hls/S11CC-B8DF6B00157D/index.m3u8",
    ]
    stop_event = Event()
    predict("test", stream_path[0], stop_event, True)


def predict(ac_no: str, videoUrl: str, stop_event: Event, is_show: bool = False):
    strapi_client = StrapiClient()
    app_path = os.path.join(os.path.dirname(__file__), '..')
    model_path = os.path.join(app_path, 'runs/detect/train3/weights/best.pt')
    cap = cv2.VideoCapture(videoUrl)
    model = YOLO(model_path)

    while cap.isOpened():
        if stop_event.is_set():
            break

        success, frame = cap.read()
        if success:
            # Perform object detection on an image
            result, *tail = model.track(frame, is_show)

            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                cls_name = result.names[cls_id]
                print(f"[{ac_no}] cls_id:{cls_id}, cls_name:{cls_name}, conf:{conf}")

                event: SafetyEvent = {
                    "AC_NO": ac_no,
                    "className": cls_name,
                    "trace_id": "h2ll",
                    "conf": conf,
                }
                strapi_client.createSafetyEvent(event)

            if is_show is True:
                annotated_frame = result.plot()
                cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
        time.sleep(1)

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
