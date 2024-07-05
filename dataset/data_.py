
import os
import copy
import random
from dataset.tenhou import TenhouData
from torch.utils.data import IterableDataset

class TenhouDataset(IterableDataset):
    def __init__(self, data_dir, mode='discard', target_length=1):
        self.data_dir = data_dir
        self.data_files = os.listdir(data_dir)
        self.target = slice(0, target_length)
        self.func = f'parse_{mode}_data'
        self.data_buffer = []
        self.used_data = []

    def __iter__(self):
        if len(self.data_buffer) == 0:
            self.update_buffer()
        for data in self.data_buffer:
            yield (data[0], data[1] // 4)
        self.data_buffer.clear()

    def reset(self):
        self.data_files = copy.copy(self.used_data)
        random.shuffle(self.data_files)
        self.used_data.clear()

    def update_buffer(self):
        data_file = self.data_files.pop()
        self.used_data.append(data_file)
        playback = TenhouData(os.path.join(self.data_dir, data_file))
        targets = playback.get_rank()[self.target]
        for target in targets:
            features, labels = playback.__getattribute__(self.func)(target=target)
            if isinstance(features, list):
                data = list(zip(features, labels))
                random.shuffle(data)
                self.data_buffer.extend(data)
            else:
                self.data_buffer.append((features, labels))