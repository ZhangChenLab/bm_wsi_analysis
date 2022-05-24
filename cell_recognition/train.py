import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision
from tqdm import tqdm
import math
#from model import resnet34,resnext101_32x8d
from torchvision.models import resnext101_32x8d,resnet152,resnet101
from pytorchtools import EarlyStopping
from torchsummary import summary
from focal_Loss import focal_loss,FocalLoss
import pandas as pd
from PIL import Image

import argparse
from utils import get_weights,get_acc,get_roc


def main(args):
    task = args.task
    save_route =args.save_route
    save_path = os.path.join(save_route,task)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_size = args.img_size
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    patientce = args.patientce
    alpha = 0.25
    gamma = 2


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    if args.test == False:
        data_transform = {
            "train": transforms.Compose([transforms.Resize(img_size),
                                         transforms.CenterCrop(img_size),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(img_size),
                                       transforms.CenterCrop(img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

        image_path = args.data_path
        assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
        train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                             transform=data_transform["train"])

        per_cls_weights = get_weights(os.path.join(image_path,'train'))
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        setting = {'img_size': img_size,
                   'batch_size': batch_size,
                   'weights': per_cls_weights.cpu().numpy().tolist(),
                   'gamma':gamma,
                   'lr': lr,
                   'patientce':patientce,
                   'task': task}
        with open(save_path + '/experiment.txt', 'w') as f:
            print(setting, file=f)
        f.close()

        flower_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in flower_list.items())

        json_str = json.dumps(cla_dict, indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
        #nw = 0
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=nw)
        train_num = len(train_dataset)
        validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                                transform=data_transform["val"])
        val_num = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=batch_size, shuffle=False,
                                                      num_workers=nw)

        print("using {} images for training, {} images for validation.".format(train_num,
                                                                               val_num))
        net = resnext101_32x8d()
        model_weight_path = '/media/wagnchogn/data_2tb/transformer_bm_cells/resnet_101_cell_cls/pre_pth/resnext101_32x8d-8ba56ff5.pth'
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        net.load_state_dict(torch.load(model_weight_path, map_location=device))
        # for param in net.parameters():
        #     param.requires_grad = False
        # change fc layer structure

        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, 15)
        net.to(device)

        # define loss function
        #loss_function = nn.CrossEntropyLoss()

        loss_function = FocalLoss(weight=per_cls_weights,gamma=gamma)

        # construct an optimizer
        params = [p for p in net.parameters() if p.requires_grad]
        #optimizer = optim.Adam(params, lr=0.0001)

        optimizer = optim.Adam(params, lr=lr, weight_decay=0.0005)

        early_stopping = EarlyStopping(patientce,verbose=True)
        # learning rate scheduler
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=2,
                                                       gamma=0.9)



        best_acc = 0.0
        save_weights_path = os.path.join(save_path,'best_model.pth')
        train_steps = len(train_loader)

        train_accs = []
        val_losss = []
        val_acc = []
        train_epochs = []
        lrs = []

        for epoch in range(epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader)
            train_acc = 0.0
            vaild_losses = []
            avg_valid_losses = []
            for step, data in enumerate(train_bar):

                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                predict_y = torch.max(logits, dim=1)[1]
                train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            train_accurate = train_acc / train_num
            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    loss = loss_function(outputs,val_labels.to(device)).cpu().item()

                    vaild_losses.append(loss)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            valid_loss = np.average(vaild_losses)
            val_accurate = acc / val_num

            print('[epoch %d] lr: %.6f train_loss: %.3f train_acc: %.3f val_accuracy: %.3f' %
                  (epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'],running_loss / train_steps,train_accurate, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_weights_path)

            lr_scheduler.step()
            train_accs.append(train_accurate)
            val_losss.append(valid_loss)
            val_acc.append(val_accurate)
            train_epochs.append(epoch)
            lrs.append(lr)
            valid_losses = []
            early_stopping(valid_loss,net)
            if early_stopping.early_stop:
                print("early stopping")
                break

        df = pd.DataFrame({'train_epochs':train_epochs,'train_accs':train_accs,'val_losss':val_losss,'val_acc':val_acc,'lrs':lrs})
        df.to_csv(os.path.join(save_path,'records.csv'))
        print('Finished Training')
    else:
        print('test start')
        model = resnext101_32x8d(num_classes=15).to(device)
        weights_path = os.path.join(save_path,'best_model.pth')
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        df = pd.DataFrame(columns=('img', 'predict_scoremax', 'label', 'pred', 'pred_score'))

        data_path = os.path.join(args.data_path,'test_patient')
        result_path = os.path.join(save_path,'predict')
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        patients = os.listdir(data_path)

        data_transform = transforms.Compose([transforms.Resize(img_size),
                                            transforms.CenterCrop(img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


        json_file = open('class_indices.json', "r")
        class_indict = json.load(json_file)
        cls_dict = dict((v, k) for k, v in class_indict.items())
        bar = tqdm(patients)
        for patient in bar:
            df_patient = pd.DataFrame(columns=('img', 'predict_scoremax', 'label', 'pred', 'pred_score'))
            patient_path = os.path.join(data_path, patient)
            labels = os.listdir(patient_path)
            output_file = os.path.join(result_path, patient)
            if not os.path.exists(output_file):
                os.makedirs(output_file)
            for label in labels:
                img_path = os.path.join(data_path, patient, label)
                img_files = os.listdir(img_path)
                for img_file in img_files:
                    pic = os.path.join(img_path, img_file)
                    img = Image.open(pic)
                    img = data_transform(img)
                    img = torch.unsqueeze(img, dim=0)
                    model.eval()
                    with torch.no_grad():
                        # predict class
                        output = torch.squeeze(model(img.to(device))).cpu()
                        predict = torch.softmax(output, dim=0)
                        predict_score = predict.numpy()
                        # predict_scoremax = predict.numpy()
                        predict_cla = torch.argmax(predict).numpy()
                        predict_score_max = predict_score[np.argmax(predict_score)]
                        pred_label = class_indict[str(predict_cla)]
                        if label in cls_dict.keys():
                            true_label_idx = cls_dict[label]
                        if pred_label in cls_dict.keys():
                            pred_label_idx = cls_dict[pred_label]

                        df = df.append([{'img': img_file, 'predict_scoremax': predict_score_max, 'label': label,
                                         'pred': class_indict[str(predict_cla)], 'predict_scores': predict_score,
                                         'true_label_idxs': true_label_idx, 'pred_label_idxs': pred_label_idx}],
                                       ignore_index=True)
                        df_patient = df_patient.append([{'img': img_file, 'predict_scoremax': predict_score_max,
                                                         'label': label, 'pred': class_indict[str(predict_cla)],
                                                         'predict_scores': predict_score,
                                                         'true_label_idxs': true_label_idx,
                                                         'pred_label_idxs': pred_label_idx}],
                                                       ignore_index=True)
            bar.set_description("Processing %s" % patient)

            df_patient.to_csv(os.path.join(output_file, patient + '.csv'))
        df.to_csv(os.path.join(save_path, 'predict_test.csv'))

        classes,cls_accs = get_acc(df,class_indict)
        cla_accs_df = pd.DataFrame({'class': classes, 'acc': cls_accs})
        save_acc_path = os.path.join(save_path, 'class_acc.csv')
        cla_accs_df.to_csv(save_acc_path)

        fpr,tpr,roc_auc,clses = get_roc(df, class_indict, save_path)
        df_auc = pd.DataFrame(roc_auc, index=[0])
        df_auc.to_excel(os.path.join(save_path, str(roc_auc["micro_auc"]) + '_auc.xlsx'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patientce', type=int, default=30)
    parser.add_argument('--img_size', default=224, type=int,help='image_size')


    parser.add_argument('--data_path', type=str,
                        default="",help='data_root')
    parser.add_argument('--weights', type=str,
                        default='',
                        help='initial weights path')
    parser.add_argument('--model', type=str, default='resnext101',
                        help='model')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--task', default='resnext', help='save_name')
    parser.add_argument('--test', default=False, help='test_mode')
    parser.add_argument('--save_route', default='',
                        help='save path')
    opt = parser.parse_args()
    main(opt)

