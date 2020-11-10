# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def ucf101_folders():
    train_folder = '../datas/ucf_data/trainsplit1'
    test_folder = '../datas/ucf_data/testsplit1'

    metatrain_folders = [os.path.join(train_folder, label) \
                for label in os.listdir(train_folder) \
                if os.path.isdir(os.path.join(train_folder, label)) \
                ]
    metatest_folders = [os.path.join(test_folder, label) \
                for label in os.listdir(test_folder) \
                if os.path.isdir(os.path.join(test_folder, label)) \
                ]

    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders,metatest_folders

class Ucf101Task(object):

    def __init__(self, character_folders, num_classes, train_num,test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders,self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()
        class_ucf_support_folders = []
        class_ucf_query_folders = []
        self.train_roots=[]
        self.test_roots=[]
        self.train_labels=[]
        self.test_labels=[]
        # print(num_classes)5
        # print(train_num)1
        # print(test_num)15
        for c in class_folders:
            #c apply
            filelist = os.listdir(c)
            #file list apply/fefew/
            train_file=random.sample(filelist,train_num)
            for d in train_file:
                class_ucf_support_folders.append(os.path.join(c,d))
            #从每类一堆视频中选一个视频 一共选出了五个文件 5way 1shot

        for c in class_folders:
            filelist = os.listdir(c)
            filelist=random.sample(filelist, test_num)
            for filefolder in filelist :
                class_ucf_query_folders.append(os.path.join(c, filefolder))

            # 从每类一堆视频中选3个视频 一共选出了15个文件 5way 1shot
        for c in class_ucf_support_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            temp.sort()
            samples=temp[len(temp)//2]
            #取中间的帧作为输入
            self.train_labels.append(labels[self.get_class(samples[0])])
            self.train_roots.append(samples)

        for c in class_ucf_query_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples=temp[1:len(temp):len(temp)//16]
            samples = samples[:16]
            self.test_labels.append(labels[self.get_class(samples[0])])
            self.test_roots.append(samples)

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-2])

class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class Ucf101(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Ucf101, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        imagelist=[]
        for i in image_root:
            i = Image.open(i)
            i = i.convert('RGB')
            if self.transform is not None:
                i = self.transform(i)
            label = self.labels[idx]
            if self.target_transform is not None:
                label = self.target_transform(label)
            imagelist.append(i)
        image=torch.stack(imagelist,dim=0)
        return image, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst,shuffle=True):

        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i+j*self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                   random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1

class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_ucf101_data_loader(task, num_per_class=1, split='train',shuffle = False):
    normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])

    dataset = Ucf101(task,split=split,transform=transforms.Compose([transforms.ToTensor(),normalize]))
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.train_num,shuffle=shuffle)

    else:
        sampler = ClassBalancedSampler(task.num_classes, task.test_num,shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    return loader
