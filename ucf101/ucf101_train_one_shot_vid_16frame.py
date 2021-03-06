import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_vid_16 as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats


import datetime
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default=50000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '4,7'

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

class CNNEncoder(nn.Module):
    #C3d
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv3d(3,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        self.layer2 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.layer3 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)))
        self.layer4 = nn.Sequential(
                        nn.Conv3d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm3d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        x=x.transpose(1,2)
        # print("input")
        # print(x.shape)
        out = self.layer1(x)
        # print("after layer 1 ")
        # print(out.shape)
        out = self.layer2(out)
        # print("after layer 2")
        # print(out.shape)
        out = self.layer3(out)
        # print("after layer 3")
        # print(out.shape)
        out = self.layer4(out)
        # print("afterape) layer 4")
        #         # print(out.sh
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

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.ucf101_folders()

    # Step 2: init neural networks
    print("init neural networks")


    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
    feature_encoder = nn.DataParallel(feature_encoder)
    relation_network = nn.DataParallel(relation_network)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str("./model/ucf_feature_encoder_c3d_16frame" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./model/ucf_feature_encoder_c3d_16frame" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./model/ucf_relation_network_c3d_16frame"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./model/ucf_relation_network_c3d_16frame"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    ########################################################################################
    year = datetime.datetime.now().year
    month = datetime.datetime.now().month
    day = datetime.datetime.now().day
    filename = "ucf_train_oneshot_c3d_16frame_" + str(year) + '_' + str(month) + '_' + str(day) + ".txt"
    with open("models/" + filename, "w") as f:

        for episode in range(EPISODE):

            feature_encoder_scheduler.step(episode)
            relation_network_scheduler.step(episode)

            # init dataset
            # sample_dataloader is to obtain previous samples for compare
            # batch_dataloader is to batch samples for training
            task = tg.Ucf101Task(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
            #class num=5
            sample_dataloader = tg.get_ucf101_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
            batch_dataloader = tg.get_ucf101_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

            # sample datas
            samples,sample_labels = sample_dataloader.__iter__().next()
            batches,batch_labels = batch_dataloader.__iter__().next()
            sample_features = feature_encoder(Variable(samples).cuda(GPU))
            batch_features = feature_encoder(Variable(batches).cuda(GPU))
            # calculate relations   each batch sample link to every samples to calculate relations
            # to form a 100x128 matrix for relation network
            # print(sample_features.shape)[5,64,1,19,19]
            sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1,1)
            sample_features_ext = torch.squeeze(sample_features_ext)
            # print(sample_features_ext.shape) 75,5,64,1,19,19
            batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1,1)
            # print(batch_features_ext.shape) [5, 75, 64, 1, 19, 19]
            batch_features_ext = torch.transpose(batch_features_ext,0,1)
            # print(batch_features_ext.shape) 75 5 64 1 19 19
            batch_features_ext = torch.squeeze(batch_features_ext)
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
            #print(relation_pairs.shape)
            relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)
            #print(relations.shape)
            mse = nn.MSELoss().cuda(GPU)
            one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
            loss = mse(relations,one_hot_labels)


            # training

            feature_encoder.zero_grad()
            relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
            torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

            feature_encoder_optim.step()
            relation_network_optim.step()


            if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.item())
                newcontext = "episode:    " + str(episode + 1) + "  loss    " + str(loss.item()) + '\n'
                f.writelines(newcontext)

            if episode%1000 == 0:
                # test
                print("Testing...")
                accuracies = []
                for i in range(TEST_EPISODE):
                    total_rewards = 0
                    counter = 0
                    task = tg.Ucf101Task(metatest_folders,CLASS_NUM,1,15)
                    sample_dataloader = tg.get_ucf101_data_loader(task,num_per_class=1,split="train",shuffle=False)
                    num_per_class = 3
                    test_dataloader = tg.get_ucf101_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                    sample_images,sample_labels = sample_dataloader.__iter__().next()
                    for test_images,test_labels in test_dataloader:
                        batch_size = test_labels.shape[0]
                        # calculate features
                        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                        test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                        # calculate relations
                        # each batch sample link to every samples to calculate relations
                        # to form a 100x128 matrix for relation network
                        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1,1)
                        sample_features_ext = torch.squeeze(sample_features_ext)
                        test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1,1)
                        test_features_ext = torch.transpose(test_features_ext,0,1)
                        test_features_ext = torch.squeeze(test_features_ext)
                        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                        _,predict_labels = torch.max(relations.data,1)

                        rewards = [1 if predict_labels[j].cuda()==test_labels[j].cuda() else 0 for j in range(batch_size)]

                        total_rewards += np.sum(rewards)
                        counter += batch_size
                    accuracy = total_rewards/1.0/counter
                    accuracies.append(accuracy)

                test_accuracy,h = mean_confidence_interval(accuracies)

                print("test accuracy:",test_accuracy,"h:",h)
                newcontext ="episode:    "+ str(episode + 1) +"test accuracy:    " + str(test_accuracy)+ '\n'
                f.writelines(newcontext)

                if test_accuracy > last_accuracy:

                    # save networks
                    torch.save(feature_encoder.state_dict(),str("./model/ucf_feature_encoder_c3d_16frame" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save(relation_network.state_dict(),str("./model/ucf_relation_network_c3d_16frame"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                    print("save networks for episode:",episode)

                    last_accuracy = test_accuracy





if __name__ == '__main__':
    main()