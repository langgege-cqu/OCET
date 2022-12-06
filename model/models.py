from torchtools import *
from collections import OrderedDict
import math
#import seaborn as sns

import matplotlib.pyplot as plt

import torch.nn.functional as F
#import glog as log

import glog as log

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np



def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp
    
def feature_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    feature_pyramid = []
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2
        #maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
        maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(int(h_pad), int(w_pad)))
        x = maxpool(previous_conv)
        feature_pyramid.append(x)
    return feature_pyramid
    
                                        

class LinearRegression(nn.Module):
    def __init__(self, in_features, class_num):
        super(LinearRegression, self).__init__()
        # set size
        self.embedding_net = nn.Linear(in_features=in_features, out_features=class_num, bias=True)

    def forward(self, x):
        y = self.embedding_net(x)
        return y 

class MLP(nn.Module):
    def __init__(self, in_features, class_num):
        super(MLP, self).__init__()
        # set size
        self.layer_1 = nn.Sequential(nn.Linear(in_features=in_features, out_features=512, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_2 = nn.Sequential(nn.Linear(in_features=512, out_features=1024, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_3 = nn.Sequential(nn.Linear(in_features=1024, out_features=1024, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_4 = nn.Sequential(nn.Linear(in_features=1024, out_features=64, bias=True),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=64, out_features=class_num, bias=True))

    def forward(self, x):
        y = self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x)))))
        return y 



class ConvNet(nn.Module):
    def __init__(self, in_features, class_num):
        super(ConvNet, self).__init__()
        # set size
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=[4,124],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=[1,1],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=[4,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=[3,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))
        self.layer_5 = nn.Sequential(nn.Linear(in_features=512, out_features=class_num, bias=True)) 

    def forward(self, x):
        conv_feat = self.conv_1(x)
        #print(conv_feat.shape)            
        conv_feat = self.conv_2(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_3(conv_feat)
        #print(conv_feat.shape) 
        conv_feat = self.conv_4(conv_feat)
        #print(conv_feat.shape)  
        conv_feat = F.avg_pool2d(conv_feat, kernel_size=conv_feat.size()[2:])
        conv_feat = conv_feat.squeeze(-1).squeeze(-1)
        #print(conv_feat.shape) 
        #assert 0==1  
        y = self.layer_5(conv_feat)
        return y


class LstmNet(nn.Module):
    def __init__(self, in_features, seq_len, class_num):
        super(LstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len 
        self.class_num = class_num         
        self.rnn = nn.LSTM(in_features, in_features, num_layers=1)
        self.relu= nn.ReLU(inplace=False) 
        self.classifier = nn.Sequential(nn.Linear(in_features=in_features*seq_len, out_features=class_num, bias=True)) 

    def forward(self, x):
        x_reshaped = x.permute(1,0,2)
        batch_size = x_reshaped.size()[1]          
        h0 = torch.zeros(1, batch_size, self.in_features).to(x.device)
        c0 = torch.zeros(1, batch_size, self.in_features).to(x.device)
        #print(x_reshaped.shape,x_reshaped)        
        output, (hn,cn) = self.rnn(x_reshaped,(h0,c0))
        output = output.permute(1,0,2).contiguous()
        b,l,c = output.shape
        output = output.view(b,l*c)        
        y = self.classifier(self.relu(output))
        return y


class ConvLstmNet(nn.Module):
    def __init__(self, in_features, seq_len, class_num):
        super(ConvLstmNet, self).__init__()
        # set size
        self.in_features = in_features
        self.seq_len     = seq_len 
        self.class_num = class_num 

        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                              out_channels=64,
                                              kernel_size=[4,124],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=64),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                              out_channels=128,
                                              kernel_size=[1,1],
                                              padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=128),
                                    nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                              out_channels=256,
                                              kernel_size=[4,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=256),
                                    nn.ReLU(inplace=True))
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                              out_channels=512,
                                              kernel_size=[3,1],
                                              padding=0,
                                              bias=False),
                                    nn.MaxPool2d(kernel_size=[2,1]),
                                    nn.BatchNorm2d(num_features=512),
                                    nn.ReLU(inplace=True))
        self.rnn = nn.LSTM(512, 64, num_layers=1)
        self.relu= nn.ReLU(inplace=False)         
        self.layer_5 = nn.Sequential(nn.Linear(in_features=640, out_features=class_num, bias=True)) 

    def forward(self, x):
        # print(x.shape) 
        conv_feat = self.conv_1(x)
        # print(conv_feat.shape)            
        conv_feat = self.conv_2(conv_feat)
        # print(conv_feat.shape) 
        conv_feat = self.conv_3(conv_feat)
        # print(conv_feat.shape) 
        conv_feat = self.conv_4(conv_feat)
        # print(conv_feat.shape)  
        conv_feat = conv_feat.squeeze(-1).permute(2,0,1)
        # print(conv_feat.shape) 
        rnn_feat,_  = self.rnn(conv_feat)
        # print(rnn_feat.shape) 
        rnn_feat  = rnn_feat.permute(1,0,2).contiguous()
        # print(rnn_feat.shape) 
        b,l,d     = rnn_feat.shape
        rnn_feat  = rnn_feat.view(b,l*d)
        # print(rnn_feat.shape) 
        rnn_feat  = self.relu(rnn_feat)
        # print(rnn_feat.shape)    
        y = self.layer_5(rnn_feat)
        #print(y.shape)
        #assert 0==1  
        return y






class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x





class OCET(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, dim_head, mlp_dim, dropout=0.,):
        super().__init__()


        self.conv_positional_embedding = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(4, 1), padding='same'),
            nn.Sigmoid()
        )


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)


        self.fc = nn.Linear(dim, num_classes)
        

    def forward(self, x):
        x = x + self.conv_positional_embedding(x)
        
        

        x = torch.squeeze(x)

        x = self.transformer(x)
        
        x = x[:, 0]
        x = self.fc(x)
        
        return x




