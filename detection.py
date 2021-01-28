from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from neuralNet import NeuralNet
import pickle as pkl
import pandas as pd
import random
import pickle as pkl
import sys
import os


colors = pkl.load(open("data/pallete", "rb"))
outputPath="/home/shubham/Documents/yolo/yolo-project/output/{}"
def write(x, img):

    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    return img



num_classes = 80
classes = loadClasses("data/coco.names")
confidence=0.5
nms_thresh=0.4
dimension=416

print("loading network")
model=NeuralNet("yolov3.cfg")
model.loadWeights("yolov3.weights")
print("network loaded")


model.networkInfo["height"]=416
model.cuda()
model.eval()

if(len(sys.argv)>1):
    folderName=sys.argv[1]
else:
    folderName="images"

imageList=[]
if(os.path.isdir(folderName)):
    imageList= [os.path.join(folderName, f) for f in os.listdir(folderName)]
else:
    imageList.append(folderName)


for imageName in imageList:
  
    frame=cv2.imread(imageName)
    image, orig_im=preprocess(frame, 416)

    image=image.cuda()


    start=time.time()
    output=model(Variable(image))

    output = write_results(output, confidence, num_classes, nms_conf=nms_thresh)


    if isinstance(output, int) == False:
        output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(dimension)) / dimension
        output[:, [1, 3]] *= orig_im.shape[1]
        output[:, [2, 4]] *= orig_im.shape[0]

        list(map(lambda x: write(x, orig_im), output))

    print("image predicted in {:2.3f} seconds".format(time.time()- start))
    cv2.imwrite(outputPath.format(imageName),orig_im)
