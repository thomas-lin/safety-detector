# make dataset for yolo8 from label-stadio

[導出數據集](https://richelf.tech/posts/2023-09/0f87cd80-8c23-c2dd-40f3-e1d2df83c2f7.html#%E5%AF%BC%E5%87%BA%E6%95%B0%E6%8D%AE%E9%9B%86)

# Train with Python

## folder structure

- dataset: 訓練驗證資料集，每個資料夾下包含 test、train 及 val 三個資料夾，分別用於測試、訓練及驗證
  |- images: 圖片
  |- labels: 標籤
- run: 訓練結果

## train

```
poetry run aia-yolo-train
```

# [OPTIONS] Use yolo8 to train with docker

```
sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container
```

## train and test

測試失敗，Macbook 無法使用 GPU 來進行訓練

```
docker run -it --ipc=host -v ${PWD}/yolo8/data:/usr/src/datasets ultralytics/ultralytics:8.1.42-arm64
cd /usr/src/datasets
yolo train data=data.yaml model=yolov8n.pt epochs=50
```
