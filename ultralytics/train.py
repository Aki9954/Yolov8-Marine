from ultralytics import YOLO
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def train_model():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 强制设置PyTorch使用的CUDA设备为索引0的GPU

    # 加载model
    # model = YOLO('yolov8n.yaml')  # build a new model from YAML
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('runs/detect/train10/weights/last.pt')  # build from YAML and transfer weights
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    # model = YOLO('runs/detect/train7/weights/last.pt')  # load a partially trained model
    device = "cuda:0"  # 使用第一个CUDA设备


    # 训练
    results = model.train(data='datasets/marine/data.yaml', epochs=2400, batch=64, lr0=0.01, resume=True,patience=5000, device=0)
    """ yolo detect train data=ultralytics/datasets/marine/data.yaml model=yolov8s.yaml pretrained=ultralytics/yolov8s.pt epochs=300 batch=16 lr0=0.01 resume=True device=0
    yolo train resume model=ultralytics/runs/detect/train2/weights/last.pt
    """
    # 继续训练
    # results = model.train(resume=True)
if __name__ == '__main__':
    train_model()
