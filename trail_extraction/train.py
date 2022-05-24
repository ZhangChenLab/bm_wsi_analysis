import os
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
import argparse
from my_dataloader import MyDataSet
from torchvision import transforms
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torchvision.models import resnet18

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_txt_path = os.path.join(args.txt_path,'train.txt')
    val_txt_path = os.path.join(args.txt_path, 'val.txt')
    train_dataset = MyDataSet(txt=train_txt_path,
                              transform=data_transform["train"])
    val_dataset = MyDataSet(txt=val_txt_path,
                              transform=data_transform["val"])
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet18()
    model_weight_path = r"H:\hemopathy_diagnosis_code\region_select\train\pre_pth\resnet18.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, args.class_num)
    net.to(device)
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    epochs = args.epoch
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    save_route = os.path.join(args.save_dir,args.task)
    os.makedirs((save_route), exist_ok=True)
    train_epochs = []
    train_losses = []
    val_accs = []
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        train_epochs.append(epoch)
        train_losses.append(running_loss)
        val_accs.append(val_accurate)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), os.path.join(save_route,'resnet18.pth'))
    setting = {'class_num':args.class_num,
               'max_epoch':args.epoch,
               'batch_size':args.batch_size,
               'task':args.task,
               'best_acc':best_acc}
    with open(save_route + '/experiment_{}.txt'.format(args.task), 'w') as f:
        print(setting, file=f)
    f.close()
    train_detail = pd.DataFrame({'epoch':train_epochs,'train_loss':train_losses,'val_acc':val_accs})
    train_detail.to_excel(os.path.join(save_route,'train_detail.xlsx'))
    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trail Extraction')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--txt_path', type=str,
                        default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    parser.add_argument('--custom_downsample', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, default=-1)
    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()


    main(args)




