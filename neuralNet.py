from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *
import json
from pathlib import Path 



def parse_cfg(configurationFile):
    configuration = Path(configurationFile).read_text()
    blocks = json.loads(configuration)
    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors



def createModules(blocks):
    net = blocks[0]
    modules = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for i, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if(x['type'] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size,
                            stride, pad, bias=bias)
            module.add_module("conv_{}".format(i), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(i), bn)

            if activation == "leaky":
                relu = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(i), relu)



        if(x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(i), upsample)
                
             
        if (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            start = int(x["layers"][0])
            end=0
            if(len(x["layers"])>1):
                end = int(x["layers"][1])
                
            
            if start > 0: 
                start = start - i
            if end > 0:
                end = end - i
            route = EmptyLayer()
            module.add_module("route_{0}".format(i), route)
            if end < 0:
                filters = output_filters[i + start] + output_filters[i + end]
            else:
                filters= output_filters[i + start]
                

        if x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(i), shortcut)
            
        
        if x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
    
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
    
            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(i), detection)
            
        
        modules.append(module)
        prev_filters = filters
        output_filters.append(filters)
        
    return (net, modules)


class NeuralNet(nn.Module):
    def __init__(self, config):
        super(NeuralNet, self).__init__()
        self.blocks=parse_cfg(config)
        self.networkInfo, self.moduleList=createModules(self.blocks)
    
    
    def forward(self,x):
        modules=self.blocks[1:]
        outputs={}
        detections=None
        
        for index, module in  enumerate(modules):
            type=module["type"]
            if(type=="convolutional" or type=="upsample"):
                x=self.moduleList[index](x)
                
            if(type=="route"):
                layers=module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - index
    
                if len(layers) == 1:
                    x = outputs[index + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - index
    
                    map1 = outputs[index + layers[0]]
                    map2 = outputs[index + layers[1]]
                    x = torch.cat((map1, map2), 1)
            
            if type=="shortcut":
                f=int(module["from"])
                x=outputs[index-1]+outputs[index+f]
                

            if type=="yolo":
                anchors=self.moduleList[index][0].anchors
                dimension=int(self.networkInfo["height"])
                classes=int(module["classes"])
                x=x.data
                x=transformPrediction(x, dimension, anchors,classes)
                if detections is None:
                    detections=x
                else:
                    detections=torch.cat((detections,x),1)
            
            outputs[index]=x
            
        return detections
    
    
    
    def loadWeights(self, file):
        file=open(file,"r")
        
        header = np.fromfile(file, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(file, dtype = np.float32)
        
        ptr = 0

        
        for i in range(len(self.moduleList)):
            type=self.blocks[i+1]["type"]

            if(type=="convolutional"):
                module=self.moduleList[i]
                if("batch_normalize" in self.blocks[i+1]):
                    batchNormalize=self.blocks[i+1]["batch_normalize"]
                else:
                    batchNormalize=0
                
                conv=module[0]
                
                if (batchNormalize):
                    bn = module[1]
        
                    num_bn_biases = bn.bias.numel()
        
                   
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                   
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    
                    num_biases = conv.bias.numel()
                
                   
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                   
                    conv.bias.data.copy_(conv_biases)
                    
                
                num_weights = conv.weight.numel()
                
               
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)