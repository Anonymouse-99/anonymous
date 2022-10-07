import random
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

'''
Some utility functions
'''

def set_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y), \
            'The number of inputs(%d) and targets(%d) does not match.' % (len(x), len(y))
        self.x = x
        self.y = y
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train_epoch(model, train_loader, params , my_loss, optimizer, epoch, threshold=1.0):
    model.train()
    acc = 0
    total = 0
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()

        optimizer.zero_grad()
        outputs = model(x)

        loss_kl1 = params['lamada1'] * outputs['kl_g']
        loss_kl2 = params['lamada2'] * outputs['kl_b']
        loss = my_loss(outputs['y_hat'], y) + loss_kl1 + loss_kl2

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), threshold)
        optimizer.step()

        pred = torch.argmax(outputs['y_hat'].data, 1)
        acc += ((pred == y).sum()).cpu().numpy()
        total += len(y)
        loss_list.append(loss.item())
        loss_kl1_list.append(loss_kl1.item())
        loss_kl2_list.append(loss_kl2.item())
        # print("[TR]epoch:%d, step:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
        #       (epoch + 1, batch_idx, loss, result_dic['kl_g'], result_dic['kl_b'], (acc / total)))
    loss_mean = np.mean(loss_list)
    loss_kl1_mean = np.mean(loss_kl1_list)
    loss_kl2_mean = np.mean(loss_kl2_list)
    print("[TR]epoch:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, (acc / total)))
    return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean


def val(model, val_loader, learning, my_loss, epoch, output=True):
    model.eval()
    acc = 0
    total = 0
    pred_list = []
    true_list = []
    loss_list = []
    loss_kl1_list = []
    loss_kl2_list = []
    summ_graph_list = []
    spec_graph_list = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            # If you run this code on CPU, please remove the '.cuda()'
            x = x.cuda()
            y = y.cuda()

            outputs = model(x)

            loss_kl1 = learning['lamada1'] * outputs['kl_g']
            loss_kl2 = learning['lamada2'] * outputs['kl_b']
            loss = my_loss(outputs['y_hat'], y) + loss_kl1 + loss_kl2

            pred = torch.argmax(outputs['y_hat'].data, 1)
            pred_list.append(pred.cpu().numpy())
            true_list.append(y.cpu().numpy())
            acc += ((pred == y).sum()).cpu().numpy()
            total += len(y)

            loss_list.append(loss.item())
            loss_kl1_list.append(loss_kl1.item())
            loss_kl2_list.append(loss_kl2.item())
            if output:
                summ_graph_list.append(outputs['summ_graph'].detach().cpu().numpy())
                spec_graph_list.append(outputs['spec_graph'].detach().cpu().numpy())

    loss_mean = np.mean(loss_list)
    loss_kl1_mean = np.mean(loss_kl1_list)
    loss_kl2_mean = np.mean(loss_kl2_list)
    print("[VA]epoch:%d, loss:%f, l1:%f, l2:%f, acc:%f" %
          (epoch + 1, loss_mean, loss_kl1_mean, loss_kl2_mean, (acc / total)), end='')
    
    if output:
        pred_list = np.concatenate(pred_list)
        true_list = np.concatenate(true_list)
        summ_graph_list = np.squeeze(np.concatenate(summ_graph_list))
        spec_graph_list = np.squeeze(np.concatenate(spec_graph_list))
        return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean, \
            pred_list, true_list, summ_graph_list, spec_graph_list
    else:
        return acc / total, loss_mean, loss_kl1_mean, loss_kl2_mean



def PrintScore(true, pred, savePath=None, average='macro',
               classes=['Wake', 'N1', 'N2', 'N3', 'REM']):
    if savePath == None:
        saveFile = None
    else:
        saveFile = open(savePath + "Result.txt", 'a+')
    # Main scores
    F1 = metrics.f1_score(true, pred, average=None)
    print("Main scores:", file=saveFile)
    print('Acc\tF1S\tKappa\tF1_W\tF1_N1\tF1_N2\tF1_N3\tF1_R', file=saveFile)
    print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' %
          (metrics.accuracy_score(true, pred),
           metrics.f1_score(true, pred, average=average),
           metrics.cohen_kappa_score(true, pred),
           F1[0], F1[1], F1[2], F1[3], F1[4]),
          file=saveFile)
    # Classification report
    print()
    print("Classification report:", file=saveFile)
    print(metrics.classification_report(true, pred, target_names=classes, digits=4),
          file=saveFile)
    # Confusion matrix
    print('Confusion matrix:', file=saveFile)
    print(metrics.confusion_matrix(true, pred), file=saveFile)
    # Overall scores
    print()
    print('    Accuracy\t', metrics.accuracy_score(true, pred), file=saveFile)
    print(' Cohen Kappa\t', metrics.cohen_kappa_score(true, pred), file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=average),
          '\tAverage =', average, file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=average),
          '\tAverage =', average, file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=average),
          '\tAverage =', average, file=saveFile)
    # Results of each class
    print('\nResults of each class:', file=saveFile)
    print('    F1-Score\t', metrics.f1_score(true, pred, average=None), file=saveFile)
    print('   Precision\t', metrics.precision_score(true, pred, average=None), file=saveFile)
    print('      Recall\t', metrics.recall_score(true, pred, average=None), file=saveFile)
    if savePath != None:
        saveFile.close()
    return


def ConfusionMatrix(y_true, y_pred, savePath=None, title=None, cmap=plt.cm.Blues,
                    classes=['Wake', 'N1', 'N2', 'N3', 'REM'], print=False):
    if not title:
        title = 'Confusion matrix'
    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm_n = cm
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if print:
        print("Confusion matrix")
        print(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel='True label',
        xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i,
                    format(cm[i, j] * 100, '.2f') + '%\n' +
                    format(cm_n[i, j], 'd'),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if savePath is not None:
        plt.savefig(savePath + title + ".png")
    plt.show()
    return ax

def row2matrix(x, side):
    result = np.empty([side,side])
    edge_idx = 0
    for i in range(side):
        for j in range(i):
            result[i,j] = x[edge_idx]
            result[j,i] = x[edge_idx]
            edge_idx += 1
        result[i,i] = np.nan
    return result
