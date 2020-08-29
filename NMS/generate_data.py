import argparse
from random import random
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("-e", "--expo", type=int, default=16)
parser.add_argument("-b", "--bias", type=int, default=0)
parser.add_argument("-th", "--threshold", type=float, default=0.1)
parser.add_argument("--height", type=float, default=1080)
parser.add_argument("--width", type=float, default=1920)
args = parser.parse_args()

n = (1 << args.expo) + args.bias
FILE = open('./data/input.txt', 'w')
FILE.write('%d %s\n' % (n, args.threshold))

h = args.height
w = args.width
FILE.write('%d %d\n' % (h, w))

scores = np.random.rand(n)
scores = np.sort(scores)[::-1]

for score in scores:
    x1 ,y1, x2, y2 = w*random(), h*random(), w*random(), h*random()
    if x1 > x2:
        t = x1
        x1 = x2
        x2 = t
    if y1 > y2:
        t = y1
        y1 = y2
        y2 = t
    FILE.write('%s %s %s %s %s\n' % (x1, y1, x2, y2, score))

FILE.close()