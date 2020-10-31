import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats


from torchstat import stat


import datetime

GPU = args.gpu
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())




feature_encoder = CNNEncoder()
relation_network = RelationNetwork(64,10)

feature_encoder.apply(weights_init)
relation_network.apply(weights_init)

feature_encoder.cuda(GPU)
relation_network.cuda(GPU)
num_params = 0
for param in feature_encoder.parameters():
    num_params += param.numel()
print(num_params / 1e6)
#
# stat(feature_encoder, Variable(torch.rand(3,84,84)))
#
#             sample_features = feature_encoder(Variable(samples).cuda(GPU))
#             #torch.Size([5, 64, 19, 19])
#             batch_features = feature_encoder(Variable(batches).cuda(GPU))
#             #75 64 19 19
#             # calculate relations
#             # each batch sample link to every samples to calculate relations
#             # to form a 100x128 matrix for relation network
#             sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
#             #torch.Size([75, 5, 64, 19, 19])
#             batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
#             #5 75 64 19 19
#             batch_features_ext = torch.transpose(batch_features_ext,0,1)
#             #75 5 64 19 19
#             relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
#             #print(relation_pairs.shape)
#             relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)
#             #print(relations.shape)
#             mse = nn.MSELoss().cuda(GPU)
#             one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
#             loss = mse(relations,one_hot_labels)
#
#
#             # training
#
#             feature_encoder.zero_grad()
#             relation_network.zero_grad()
#
#             loss.backward()
#
#             torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
#             torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)
#
#             feature_encoder_optim.step()
#             relation_network_optim.step()
#
#
#             if (episode+1)%100 == 0:
#                 print("episode:",episode+1,"loss",loss.item())
#                 newcontext = "episode:    " + str(episode + 1) + "  loss    " + str(loss.item()) + '\n'
#                 f.writelines(newcontext)
#
#             if episode%5000 == 0:
#
#                 # test
#                 print("Testing...")
#                 accuracies = []
#                 for i in range(TEST_EPISODE):
#                     total_rewards = 0
#                     counter = 0
#                     task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
#                     sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)
#
#                     num_per_class = 3
#                     test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
#                     sample_images,sample_labels = sample_dataloader.__iter__().next()
#                     for test_images,test_labels in test_dataloader:
#                         batch_size = test_labels.shape[0]
#                         # calculate features
#                         sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
#                         test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64
#
#                         # calculate relations
#                         # each batch sample link to every samples to calculate relations
#                         # to form a 100x128 matrix for relation network
#                         sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
#                         test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
#                         test_features_ext = torch.transpose(test_features_ext,0,1)
#                         relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
#                         relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
#
#                         _,predict_labels = torch.max(relations.data,1)
#
#                         rewards = [1 if predict_labels[j].cuda()==test_labels[j].cuda() else 0 for j in range(batch_size)]