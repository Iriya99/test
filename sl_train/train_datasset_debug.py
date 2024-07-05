import os
import copy
import torch
from torch.optim import Adam
import argparse
import wandb
from torch.nn import CrossEntropyLoss
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.models import DiscardModel
from dataset.data import process_data, TenhouData

mode = 'discard'
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', '-n', default=50, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
args = parser.parse_args()

# experiment = wandb.init(project='Mahjong', resume='allow', anonymous='must', name=f'train-{mode}-sl')

import random
from torch.utils.data import IterableDataset, DataLoader

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

train_set = TenhouDataset(data_dir='data', mode=mode, target_length=2)
train_loader = DataLoader(train_set, batch_size=512)

for i, data in enumerate(train_loader):    
    print(data[0].dtype, data[1].dtype)
    exit()
# train_set.data_files, test_set.data_files = train_set.data_files[:len_train], train_set.data_files[len_train:]

# num_layers = args.num_layers
# in_channels = 291
# model = DiscardModel(num_layers=num_layers, in_channels=in_channels)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# optim = Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
# loss_fcn = CrossEntropyLoss()
# epochs = args.epochs

# os.makedirs(f'output/{mode}-model/checkpoints', exist_ok=True)
# max_acc = 0
# global_step = 0
# for epoch in range(epochs):
#     while len(train_set) > 0:
#         data = train_set()
#         if len(data) == 0:
#             break
#         features, labels = process_data(data, label_trans=lambda x: x // 4)
#         features, labels = features.to(device), labels.to(device)
#         output = model(features)
#         loss = loss_fcn(output, labels)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         global_step += 1
#         print(f"Epoch-{epoch + 1}: {len_train - len(train_set)} / {len_train} loss={loss.item():.3f}".center(50, '-'), end='\r')
#         experiment.log({
#             'train loss': loss.item(),
#             'epoch': epoch + 1
#         })

#     train_set.reset()

#     torch.save({"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels}, f'output/{mode}-model/checkpoints/epoch_{epoch + 1}.pt')
#     model.eval()
#     acc = model_test(model, test_set)
#     if acc > max_acc:
#         max_acc = acc
#         torch.save({"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels}, f'output/{mode}-model/checkpoints/best.pt')
#     model.train()

#     experiment.log({
#         'epoch': epoch + 1,
#         'test_acc': acc,
#         'lr': optim.param_groups[0]['lr']
#     })
#     scheduler.step(acc)

