import os
import torch
from torch.optim import Adam
import argparse
import wandb
from torch.nn import CrossEntropyLoss
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.models import DiscardModel
from dataset.data import TenhouDataset, process_data


@torch.no_grad()
def model_test(model, test_loader):
    acc = 0
    total = 0
    for i, data in enumerate(test_loader):
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        available = features[:, :4].sum(1) != 0
        pred = (output * available).argmax(1)
        correct = (pred == labels).sum()
        acc += correct
        total += len(labels)
        print(f"Testing - acc: {correct.item() / len(labels):.3f}".center(50, '-'), end='\r')
    test_set.reset()
    return acc / total

mode = 'discard'
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', '-n', default=50, type=int)
parser.add_argument('--epochs', '-e', default=10, type=int)
args = parser.parse_args()

experiment = wandb.init(project='Mahjong', resume='allow', anonymous='must', name=f'train-{mode}-sl')
# data_.py中需要限制data_files长度
train_set = TenhouDataset(data_dir='data/train', mode=mode, target_length=2)
test_set = TenhouDataset(data_dir='data/test', mode=mode, target_length=2)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2048)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2048)

num_layers = args.num_layers
in_channels = 291
model = DiscardModel(num_layers=num_layers, in_channels=in_channels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optim = Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=1)
loss_fcn = CrossEntropyLoss()
epochs = args.epochs

os.makedirs(f'output/{mode}-model/checkpoints', exist_ok=True)
max_acc = 0
global_step = 0
for epoch in range(epochs):
    for i, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        output = model(features)
        loss = loss_fcn(output, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        global_step += 1
        experiment.log({
            'train loss': loss.item(),
            'epoch': epoch + 1
        })
    train_set.reset()

    torch.save({"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels}, f'output/{mode}-model/checkpoints/epoch_{epoch + 1}.pt')
    model.eval()
    acc = model_test(model, test_set)
    if acc > max_acc:
        max_acc = acc
        torch.save({"state_dict": model.state_dict(), "num_layers": num_layers, "in_channels": in_channels}, f'output/{mode}-model/checkpoints/best.pt')
    model.train()

    experiment.log({
        'epoch': epoch + 1,
        'test_acc': acc,
        'lr': optim.param_groups[0]['lr']
    })
    scheduler.step(acc)

