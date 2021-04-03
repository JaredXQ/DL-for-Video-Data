import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaitDataset(Dataset):
    """Generates an iterable dataset containing the samples of all 
    PIDs that performed the selected task (scenario)"""
    
    def __init__(self, scenario, labels_file_path, data_dir, concatenate=5, jump=2):
        self.scenario = scenario
        self.labels_file_path = labels_file_path
        self.labels_df = pd.read_csv(labels_file_path)
        self.data_dir = data_dir
        self.pids = self.select_task(self.scenario)
        self.concatenate = concatenate
        self.jump = jump
        self.mean, self.std = self.find_mean_std_across_dataset(self.labels_df)
        self.samples, self.frames_count, self.labels, self.pids_list = self.generate_data()
        
    def __getitem__(self, idx):
#         """Returns a dict of {sample, frames_count, label}"""
#         return {'sample_data': self.samples[idx], 'frame_count': self.frames_count[idx], 'label': self.labels[idx]}
        return self.samples[idx], self.frames_count[idx], self.labels[idx]
        
    def __len__(self):
        return len(self.samples)
    
    
    def generate_data(self):
        df = self.labels_df
        samples_list = []
        frames_count_list = []
        pids_list = []
        labels_list = []
        for pid in self.pids:
            sub_df = df[(df['PID'] == pid) & (df['scenario'] == self.scenario)]
            keys = sub_df['key'].values
            for i in range(0, len(keys) - self.concatenate + 1, self.jump):
                #Normalize data_sample tensor
                data_sample = torch.tensor(self._multiple_strides_array(keys[i:i+self.concatenate])).float()
                data_sample = self.normalize(data_sample).to(device)
                samples_list.append(data_sample)
                #Normalize frames_count
                frames_count = torch.tensor(sub_df['frame_count'].values[i:i+self.concatenate]).float()
                frames_count = self.normalize(frames_count, frame_count=True).to(device)
                frames_count_list.append(frames_count)
                labels_list.append(torch.tensor([sub_df['label'].values[i]]).to(device))
                pids_list.append(sub_df['PID'].values[i])
                
#         return (torch.tensor(samples_list).to(device), torch.tensor(frames_count_list).to(device),
#                 torch.tensor(labels_list).to(device), torch.tensor(pids_list).to(device))
        return samples_list, frames_count_list, labels_list, pids_list

            
    
    
    def _multiple_strides_array(self, keys):
        """Returns a non-normalized numpy.ndarray of the strides in a single sample"""
        multiple_strides = []
        for key in keys:
            stride_file = self.data_dir + key + '.csv'
            key_df = pd.read_csv(stride_file)
            ndarray_from_key = key_df.loc[:, 'right hip-x':'right heel-z'].to_numpy()
            multiple_strides.append(ndarray_from_key)
        return np.concatenate(list(stride for stride in multiple_strides))    
    
    
    def normalize(self, tensor, frame_count = False, eps=1e-6):
        if frame_count == True:
            tensor_mean = self.mean
            tensor_std = self.std
        else:
            tensor_mean = tensor.mean(dim=0, keepdim=True)
            tensor_std = tensor.std(dim=0, keepdim=True)
        
        tensor_normal = (tensor-tensor_mean)/(tensor_std + eps)
        return tensor_normal

    def find_mean_std_across_dataset(self, df):
        frame_count_df = df[df['scenario'] == self.scenario]['frame_count']
        mean = frame_count_df.mean()
        std = frame_count_df.std()
        return mean, std

    
    
    def select_task(self, scenario):
        """
        Args:
            scenario (string): task to be chosen from ['W', 'WT', 'both']. 
        """
        df = self.labels_df
        if scenario == 'both':
            W_pids = list(df[(df['scenario'] == 'W')]['PID'].unique())
            WT_pids = list(df[(df['scenario'] == 'WT')]['PID'].unique())
            both = [pid for pid in W_pids if (pid in WT_pids)]
            return both
        return df[df['scenario'] == scenario]['PID'].unique()

    def statistics(self):
        label_data = self.labels_df
        # num of groups for each cohort
        label_col = self.labels
        group_num = {'HOA' : 0, "MS" : 0, "PD" : 0}
        for i in label_col:
            if i == 0:
                group_num['HOA'] += 1
            elif i == 1:
                group_num['MS'] += 1
            else:
                group_num['PD'] += 1

        # num of actual person for each cohort
        person_num = {'HOA' : 0, "MS" : 0, "PD" : 0}
        for i in self.pids:
            if label_data[label_data['PID'] == i]['label'].unique() == 0:
                person_num['HOA'] += 1
            elif label_data[label_data['PID'] == i]['label'].unique() == 1:
                person_num['MS'] += 1
            else:
                person_num['PD'] += 1

        # Imbalance Ratio
        IR = {"hoa" : 1, "ms" : group_num['MS'] / group_num['HOA'], "pd" : group_num['PD'] / group_num['HOA']}
        return group_num, person_num, IR


def get_loaders(train_task, test_task, labels_file_path, data_dir):
    
    train_dataset = GaitDataset(train_task, labels_file_path, data_dir)  
    train_loader = DataLoader(train_dataset)
    test_dataset = GaitDataset(test_task, labels_file_path, data_dir)  
    test_loader = DataLoader(test_dataset)
    return train_loader, test_loader














"""
# test

train_task = 'W'
test_task = 'WT'
labels_file_path = '/Users/jared_/Desktop/Data/labels.csv'
data_dir = '/Users/jared_/Desktop/Data/downsampled_strides/'
train_dataset = GaitDataset(train_task, labels_file_path, data_dir)  
train_loader = DataLoader(train_dataset)
test_dataset = GaitDataset(test_task, labels_file_path, data_dir)  
test_loader = DataLoader(test_dataset)


# data (100, 36)
train_dataset[0][0]

# frame count torch.Size([5])
train_dataset[0][1]

# label
train_dataset[0][2]
"""
