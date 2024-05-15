import os

from ultralytics import YOLO


def train():
    app_path = os.path.join(os.path.dirname(__file__), '..')
    dataset_path = os.path.join(app_path, 'dataset')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=10, imgsz=64)

    # # Evaluate the model's performance on the validation set
    model.val()

    # # Perform object detection on an image using the model
    # results = model('https://ultralytics.com/images/bus.jpg')

    # # Export the model to ONNX format
    model.export(format='onnx')
