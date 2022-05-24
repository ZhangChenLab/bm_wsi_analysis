from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
import argparse
from pytorchtools import EarlyStopping
from torch.autograd import Variable
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
from my_dataset import MyDataSet
from torchvision.models import resnext101_32x8d,resnet152,resnet101
#from model import *
from utils import read_split_data, train_one_epoch, evaluate,epoch_test
import time
import numpy as np
import torch.nn as nn

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_name = args.task
    save_route = os.path.join(args.save_route, model_name)
    if not os.path.exists(save_route):
        os.makedirs(save_route)

    train_images_path, train_images_label, class_indices = read_split_data(os.path.join(args.data_path, "train"),
                                                                           modes='train')
    img_size = int(args.img_size)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = 0
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    if args.model == 'resnext101_32x8d':
        model = resnext101_32x8d()
    elif args.model == 'resnet101':
        model = resnet101()
    elif args.model == 'resnet152':
        model = resnet152()
    else:
        print('model error')
        raise NotImplementedError
    model_weight_path = args.weights
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in model.parameters():
    # param.requires_grad = False
    in_channel = model.fc.in_features
    class_num = int(args.num_classes)
    model.fc = nn.Linear(in_channel, class_num)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr
    optimizer = optim.Adam(params, lr=lr, weight_decay=0.0005)
    init_value = 1e-8
    final_value = 10.
    beta = 0.98
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for data in train_loader:
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        # outputs = net(inputs.to(device))
        # loss = criterion(outputs, labels)
        loss_function = nn.CrossEntropyLoss()
        logits = model(inputs.to(device))
        loss = loss_function(logits, labels.to(device))
        # loss = loss_function(logits, labels.to(device))
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    plt.plot(log_lrs[10:-5], losses[10:-5])
    save_path_pic = os.path.join(save_route,args.model+'_find_lr.png')
    plt.savefig(save_path_pic)
    plt.show()
    df = pd.DataFrame({'logs':log_lrs,'losses':losses})
    df.to_excel(os.path.join(save_route,args.model+'_find_lr.xlsx'))
    #print(type(log_lrs), type(losses))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--data-path', type=str,
                        default="/media/wagnchogn/data_2tb/transformer_bm_cells/dataset/0820_SEG_aug")
    parser.add_argument('--weights', type=str,
                        default='/media/wagnchogn/data_2tb/transformer_bm_cells/resnet/pretrained_pth/resnet152.pth',
                        help='initial weights path')
    parser.add_argument('--model', type=str, default='resnet152',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--task', default='resnet152_20220411', help='save_name')
    parser.add_argument('--img_size', default='400', help='image_size')
    parser.add_argument('--save_route', default='/media/wagnchogn/data_2tb/transformer_bm_cells/resnet/records', help='image_size')

    opt = parser.parse_args()
    main(opt)
