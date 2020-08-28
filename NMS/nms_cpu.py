from torchvision.ops import nms
import torch
import numpy as np
from datetime import datetime


def nms_cpu(dets, thresh):
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return torch.IntTensor(keep)


FILE = open('./data/input.txt', 'r')

s = FILE.readline()[:-1].split(' ')
n, th = int(s[0]), float(s[1])

boxes = []
scores = []

for i in range(n):
    s = FILE.readline()[:-1]
    s = s.split(' ')
    s = [float(t) for t in s]
    box, score = s[:-1], s[-1]
    boxes.append(box)
    scores.append(score)

boxes = torch.Tensor(boxes)
scores = torch.Tensor(scores)

t1 = datetime.now()
keep = nms(boxes, scores, th)
t2 = datetime.now()

print(keep.shape)
print('Time consuming (PyTorch): %fms' % ((t2 - t1).microseconds/1000))

t1 = datetime.now()
keep = nms_cpu(torch.cat([boxes, scores.unsqueeze(1)], dim=1), th)
t2 = datetime.now()

print(keep.shape)
print('Time consuming: %fms' % ((t2 - t1).microseconds/1000))
