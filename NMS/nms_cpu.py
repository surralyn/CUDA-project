from torchvision.ops import nms
import torch
import numpy as np
from datetime import datetime


FILE = open('./data/input.txt', 'r')

s = FILE.readline()[:-1].split(' ')
n, th = int(s[0]), float(s[1])
s = FILE.readline()[:-1].split(' ')
h, w = int(s[0]), int(s[1])

boxes = []
scores = []

for i in range(n):
    s = FILE.readline()[:-1]
    s = s.split(' ')
    s = [float(t) for t in s]
    box, score = s[:-1], s[-1]
    boxes.append(box)
    scores.append(score)

FILE.close()

boxes = torch.Tensor(boxes)
scores = torch.Tensor(scores)

t1 = datetime.now()
keep = nms(boxes, scores, th)
t2 = datetime.now()

print(keep.shape)
print('Time consuming (PyTorch): %fms' % ((t2 - t1).microseconds/1000))

boxes = torch.index_select(boxes, 0, keep)
scores = torch.index_select(scores, 0, keep)

FILE = open('./data/output_cpu.txt', 'w')
FILE.write('%d\n' % keep.shape[0])
for i in range(keep.shape[0]):
    FILE.write('%9.4f    %9.4f    %9.4f    %9.4f    %9.4f\n' 
        % (boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i]))
