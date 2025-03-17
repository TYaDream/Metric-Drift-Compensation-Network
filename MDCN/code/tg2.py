import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler


meta_train_folder_name = r'..\data\drift_dataset(5classes)'
meta_test_folder_name = r'..\data\drift_dataset(5classes)'

def mini_imagenet_folders():
    train_folder = meta_train_folder_name
    test_folder = meta_test_folder_name

    return train_folder,test_folder

class MiniImagenetTask(object):
    def __init__(self, character_folders, num_classes, mouths, train_num, test_num):
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        class_folders = os.listdir(character_folders)
        all_data_list = []
        for i in mouths:
            arr = np.load(character_folders + '\\' + class_folders[i])
            all_data_list.append(arr)
        all_data = np.concatenate(all_data_list, axis=0)

        train_indices = []
        test_indices = []
        all_labels = all_data[:, 0]
        unique_labels = np.unique(all_labels)
        label_indices  = {label: np.where(all_labels == label)[0] for label in unique_labels}
        for label in unique_labels:
            label_indices_array = label_indices[label]
            random_indices = np.random.choice(label_indices_array, size=train_num + test_num, replace=False)
            train_indices.append(random_indices[0:train_num])
            test_indices.append(random_indices[train_num:train_num+test_num])
        selected_train_indices = np.concatenate(train_indices, axis=0)
        selected_train_data = all_data[selected_train_indices]
        self.train_x = selected_train_data[:,1:].reshape(-1,16,8).transpose((0, 2, 1))
        #self.train_x = np.delete(self.train_x, [4,5,12,13], axis=2)
        self.train_y = selected_train_data[:,0]
        selected_test_indices = np.concatenate(test_indices, axis=0)
        selected_test_data = all_data[selected_test_indices]
        self.test_x = selected_test_data[:,1:].reshape(-1,16,8).transpose((0, 2, 1))
        #self.test_x = np.delete(self.test_x, [4,5,12,13], axis=2)
        self.test_y = selected_test_data[:,0]

class FewShotDataset(Dataset):

    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.data_x = self.task.train_x if self.split == 'train' else self.task.test_x
        self.data_y = self.task.train_y if self.split == 'train' else self.task.test_y

    def __len__(self):
        return len(self.data_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class MiniImagenet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(MiniImagenet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        temp_data = self.data_x[idx]
        label = self.data_y[idx]
        label = label.astype(np.int64)
        return temp_data, label

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


def get_mini_imagenet_data_loader(task, num_per_class=1, split='train',shuffle = False):
    dataset = MiniImagenet(task,split=split)
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class,task.num_classes, task.train_num,shuffle=shuffle)

    else:
        sampler = ClassBalancedSampler(task.num_classes, task.test_num,shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    return loader
