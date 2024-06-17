import os

import torch
from ultralytics import YOLO


def train():
    mpsEnabled = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    app_path = os.path.join(os.path.dirname(__file__), '..')
    dataset_path = os.path.join(app_path, 'dataset')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    if mpsEnabled:
        print(f"mpsEnabled: {mpsEnabled}")
        model = model.to('mps')

    # Train the model
    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=10, imgsz=320)

    # # Evaluate the model's performance on the validation set
    model.val()

    # # Export the model to ONNX format
    # model.export(format='onnx')
