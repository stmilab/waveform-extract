import os
from tqdm import tqdm

import pandas as pd
from torch.utils.data import Dataset

def frame_count(frame_size, window_size, stride):
    return int(((frame_size - (window_size - 1) - 1) / stride) + 1)

class WaveForm(Dataset):
    """
    Simple waveform sliding window dataset for PyTorch
    Arguments:
        root (str): path to data
        window_size (int): size of sliding window
        stride (int): stride size of sliding window
        preprocess (function): preprocessing function
        transform (torch.transform): transform function
        train (bool): if true, returns training split
                      if false, return test split
    """
    def __init__(
        self, root, window_size, stride=1, 
        preprocess=None, transform=None, train=True):
        
        super().__init__()
        self.data = []
        self.frame_lengths = []
        self.window_size = window_size
        self.stride = stride
        self.transform = transform

        print('Loading Data')
        for file in tqdm(os.listdir(root)):
            path = os.path.join(root, file, 'waveforms.csv')
            data_i = pd.read_csv(path).values
            
            if train:
                data_i = data_i[:int(len(data_i) * 0.8)]
            else:
                data_i = data_i[int(len(data_i) * 0.8):]
            
            if preprocess:
                data_i = preprocess(data_i)
            
            file_size = len(data_i)
            
            if file_size >= window_size:
                self.data.append(data_i)
                self.frame_lengths.append(
                    frame_count(file_size, window_size, stride))
            
    def __len__(self):
        return sum(self.frame_lengths)

    def __getitem__(self, idx):
        frame_idx = 0
        while idx >= self.frame_lengths[frame_idx]:
            idx -= self.frame_lengths[frame_idx]
            frame_idx += 1
            
            if frame_idx >= len(self.frame_lengths):
                print('out of range', frame_idx, idx)
                raise
                
        idx *= self.stride
        data = self.data[frame_idx][idx:idx+self.window_size]
        
        if self.transform:
            return self.transform(data)
        else:
            return data