import os
import sys
sys.path.append('/content/gdrive/MyDrive/yolov7')


import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from tracker import *

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

classes_to_filter = None  #You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]


opt  = {
    
    "weights": "best.pt", # Path to weights file default weights are for nano model
    "yaml"   : "data/coco.yaml",
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres" : 0.45, # NMS IoU threshold for inference.
    "device" : '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes" : classes_to_filter  # list of classes to filter or None

}
def video_detection(path_x='' ,conf_=0.25):
  import time
  start_time = time.time()
  # total_detections = 0

  video_path = path_x

  video = cv2.VideoCapture(video_path)


  #Video information
  fps = video.get(cv2.CAP_PROP_FPS)
  w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  # Initialzing object for writing video output
  # output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'DIVX'),fps , (w,h))
  torch.cuda.empty_cache()
  # Initializing model and setting it for inference
  with torch.no_grad():
    weights, imgsz = opt['weights'], opt['img-size']
    set_logging()
    device = select_device(opt['device'])
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
      model.half()
    tracker = EuclideanDistTracker()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
      model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # classes = [46,47,49]
    classes = None
    if opt['classes']:
      classes = []
      for class_name in opt['classes']:
        classes.append(opt['classes'].index(class_name))

    for j in range(nframes):
        

        ret, img0 = video.read()
        if ret:
          labels = []
          coordinates = []
          img = letterbox(img0, imgsz, stride=stride)[0]
          img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
          img = np.ascontiguousarray(img)
          img = torch.from_numpy(img).to(device)
          img = img.half() if half else img.float()  # uint8 to fp16/32
          img /= 255.0  # 0 - 255 to 0.0 - 1.0
          if img.ndimension() == 3:
            img = img.unsqueeze(0)

          # Inference
          t1 = time_synchronized()
          pred = model(img, augment= False)[0]

          # conf = 0.5
          total_detections = 0
          pred = non_max_suppression(pred, conf_, opt['iou-thres'], classes= classes, agnostic= False)
          detectionTracker = []
          t2 = time_synchronized()
          for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
              det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

              for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                total_detections += int(n)
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
      
              for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                detectionTracker.append([xyxy[0],xyxy[1],xyxy[2],xyxy[3],label,img0])
                
                # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
          boxes_ids = tracker.update(detectionTracker)
          for box_id in boxes_ids:
            x, y, w, h, id, label_,text_label = box_id
            coordinates.append([x,y,w,h , text_label,id])
            # labels.append(text_label)
            # plot_one_box([x,y,w,h], img0, label=str(id)+" "+text_label, color=colors[int(0)], line_thickness=3)

          yield img0, coordinates,colors[int(0)],3

        else:
          break
      

  # output.release()
  video.release()
# cv2.imshow("image",img0)
# cv2.waitKey(0) & 0xFF == ord("q")