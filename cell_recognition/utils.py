import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import json
import torch
import tqdm

@torch.no_grad()
def epoch_test(model, data_loader, device, epoch,num_classes):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss
        labels_np = labels.numpy()
        c = (pred_classes.cpu()==labels).squeeze().numpy()
        for i in range(c.shape[0]):
            label = labels_np[i]
            class_correct[label] += c[i]
            class_total[label] += 1
        data_loader.desc = "[test epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,class_correct,class_total


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def get_acc(df,class_indict):

    class_correct = list(0. for i in range(15))
    class_total = list(0. for i in range(15))
    for i in range(df.shape[0]):
        true_label = df['true_label_idxs'].loc[i]
        pred_label = df['pred_label_idxs'].loc[i]
        if true_label == pred_label:
            class_correct[true_label] += 1
        class_total[true_label] += 1
    cls_accs = []

    for i in range(15):
        cls_acc = class_correct[i] / class_total[i]
        cls_accs.append(cls_acc)
    classes = list(class_indict.keys())
    return classes,cls_accs



def get_roc(df,class_indict,save_path):
    labels = df["true_label_idxs"]
    predict_score = df['predict_scores']
    y_one_hot = labels.values.astype(int)
    y_one_hot = label_binarize(y_one_hot, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    n_classes = y_one_hot.shape[1]
    y_score = predict_score.values
    all_score = []
    for i in range(y_score.shape[0]):
        score = y_score[i].strip('[').strip(']')
        score = score.replace('\n', '').replace('\r', '')
        score = score.split(' ')
        score = [i for i in score if i != '']
        score1 = np.array(score).astype(float)
        all_score.append(score1)
    # macro
    all_score = np.array(all_score)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    clses = []
    a = class_indict.keys()
    for i in range(n_classes):
        if str(i) in class_indict.keys():
            cls = class_indict[str(i)]
            clses.append(cls)
        fpr[cls + '_fpr'], tpr[cls + '_tpr'], _ = roc_curve(y_one_hot[:, i], all_score[:, i])
        roc_auc[cls + '_auc'] = auc(fpr[cls + '_fpr'], tpr[cls + '_tpr'])
    # y_one_hot = label_binarize(a, classes=[1,2])
    all_fpr = np.unique(np.concatenate([fpr[i + '_fpr'] for i in clses]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in clses:
        mean_tpr += interp(all_fpr, fpr[i + '_fpr'], tpr[i + '_tpr'])
    mean_tpr /= n_classes
    fpr["macro_fpr"] = all_fpr
    tpr["macro_tpr"] = mean_tpr
    roc_auc["macro_auc"] = auc(fpr["macro_fpr"], tpr["macro_tpr"])
    # micro
    fpr["micro_fpr"], tpr["micro_tpr"], _ = roc_curve(y_one_hot.ravel(), all_score.ravel())
    roc_auc["micro_auc"] = auc(fpr["micro_fpr"], tpr["micro_tpr"])

    for i in tpr.keys():
        cla_name = i.split('_')[0]
        tpr_cla = tpr[cla_name + '_tpr'].tolist()
        fpr_cla = fpr[cla_name + '_fpr'].tolist()
        dict_cla = {'tpr_cla': tpr_cla, 'fpr_cla': fpr_cla}
        df_cla = pd.DataFrame(dict_cla)
        df_cla.to_excel(os.path.join(save_path, str(cla_name) + '.xlsx'))
    return fpr,tpr,roc_auc,clses

def roc_plot(fpr,tpr,roc_auc,clses):
    lw = 2
    plt.figure()
    plt.plot(fpr["micro_fpr"], tpr["micro_tpr"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro_auc"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro_fpr"], tpr["macro_tpr"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro_auc"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'beige', 'bisque', 'blanchedalmond', 'brown',
                    'chocolate', 'crimson', 'cyan', 'darkkhaki', 'darkred', 'darksalmon', 'darkviolet', 'gold'])
    for i, color in zip(clses, colors):
        plt.plot(fpr[i + '_fpr'], tpr[i + '_tpr'], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i + '_auc']))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def get_weights(data_root):
    def get_group_weights(cls_num_list):
        beta = 0.999
        effective_num = 1.0 - np.power(beta,cls_num_list)
        per_cls_weights = (1.0-beta)/np.array(effective_num)
        per_cls_weights = per_cls_weights/np.sum(per_cls_weights) * len(cls_num_list)
        return per_cls_weights

    labels = os.listdir(data_root)
    labels.sort()
    label_num_dict = dict()
    for label in labels:
        num = len(os.listdir(os.path.join(data_root,label)))
        label_num_dict[label] = num

    group_dict = {'group_1': ['DH','WYL','ZHYL','ZLF','ZLG'],
                  'group_2': ['BLAST','ZYL'],
                  'group_3': ['ZWH','YZH'],
                  'group_4': ['YSLB','CSLB','JXB'],
                  'group_5':['SJL','SSL']}
    cla_weights = dict()
    for i in range(len(group_dict)):
        new_list = group_dict['group_'+str(i+1)]
        cls_nums = []
        for j in range(len(new_list)):
            label = new_list[j]
            num = label_num_dict[label]
            cls_nums.append(num)
        group_weight = get_group_weights(cls_nums).tolist()
        for j in range(len(new_list)):
            label = new_list[j]
            weight = group_weight[j]
            cla_weights[label] = weight
    weights = []
    for label in labels:
        keys = list(cla_weights.keys())
        if label not in keys:
            cla_weights[label] = 1.0
            weight = 1.0
        else:
            weight = cla_weights[label]
        weights.append(weight)
    weights = np.array(weights)
    return weights







