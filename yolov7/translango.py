import time
from pathlib import Path

import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages, TransformImage
from utils.general import check_img_size, non_max_suppression, \
    scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized
import numpy as np

from typing import List, Dict

def translango_detect(img_array: np.ndarray) -> List[Dict]:
    # source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    weights = '/home/ubuntu/yolov7-flask/yolov7/yolov7.pt'
    imgsz = 640
    # device = select_device('0')
    device = select_device('cpu')
    augment = False
    conf_thres = 0.25
    iou_thres = 0.45
    predictions: List[Dict] = []

    # Initialize
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride=stride)
    image_transformer = TransformImage(img=img_array)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    img = image_transformer.get()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=augment)[0]

    # Inference
    t1 = time_synchronized()
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    t3 = time_synchronized()

    # Process detections
    for det in pred:  # detections per image
        im0 = img_array
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # line = Prediction(cls, *xywh, conf)  # label format
                x, y, w, h = xywh
                predictions.append({
                    "id": int(cls.item()),
                    "name": names[int(cls.item())],
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "conf": conf.item()
                })
    t4 = time_synchronized()

    print(f"{t4-t3}")
    return predictions

if __name__ == '__main__':
    print(translango_detect())
