import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from os import path
import numpy as np
import time
import gc
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn as nn
import torch
from model.BayesEEGNet import BayesEEGNet
from utils import *
torch.set_num_threads(4)


print('=====Start=====', time.asctime(time.localtime(time.time())))
print('PyTorch:', torch.__version__)

####################### Learning config ########################
params = {
    'optimizer': 'Adam',
    'lr': 0.001,
    'lr_decay': 0.0,
    'weight_decay': 0.0,
    'batchSize': 256,
    'minEpoch': 100,
    'hiddenDim': 512,
    'num_nodes': 6,
    'graph_dim': 512,
    'lamada1': 1e-7,
    'lamada2': 1e-7,
    'targetDim': 5,
    'dense': 64,
    'seed': 0,
    'loss_score': [1,1.5,1,1,1.5]
}
othercfg = {
    'fold': 5,
    'data_path': './data/ISRUC_S3/',
    'out_dir': './run_ISRUC/',
    'workers': 0,
}
print('[Info] Config:')
print('params:', params)
print('othercfg:', othercfg)
if not os.path.exists(othercfg['out_dir']):
    os.makedirs(othercfg['out_dir'])
    print('[Info] Make out dir:', othercfg['out_dir'])
np.savez(path.join(othercfg['out_dir'], 'params.npz'),
         params = params,
         othercfg = othercfg)

####################### Set random seed ########################
set_random_seed(params['seed'])

########################### Training ###########################
pred_list = []
true_list = []
summ_graph_list=[]
spec_graph_list=[]
tr_acc_list = []
tr_loss_list = []
tr_kl1_list = []
tr_kl2_list = []
val_acc_list = []
val_loss_list = []
val_kl1_list = []
val_kl2_list = []

print('[Info]', time.asctime(time.localtime(time.time())), 'Start training:')
for i in range(othercfg['fold']):
    print()
    print(28 * "=", 'Fold #', i, 'Train', 28 * '=')
    print(time.asctime(time.localtime(time.time())))
    # Load data from .npz files
    trainX = []
    trainY = []
    for fid in range(othercfg['fold']):
        fold = np.load(path.join(othercfg['data_path'], '%d.npz' % (fid+1)))
        if fid!=i:
            trainX.append(np.float32(fold['data']))
            trainY.append(np.int64(fold['label']))
        else:
            valX = np.float32(fold['data'])
            valY = np.int64(fold['label'])
    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)
    print('[Data] Train:', trainY.shape, 'Val:', valY.shape)

    # Organize data to Torch
    trDataset = SimpleDataset(trainX, trainY)
    cvDataset = SimpleDataset(valX, valY)
    trGen = DataLoader(trDataset,
                       batch_size = params['batchSize'],
                       shuffle = True,
                       num_workers = othercfg['workers'])
    cvGen = DataLoader(cvDataset,
                       batch_size = params['batchSize'],
                       shuffle = False,
                       num_workers = othercfg['workers'])

    # Define Model\Loss\Optimizer
    model = BayesEEGNet(hidden_size = params['hiddenDim'],
                        output_size = params['targetDim'],
                        graph_node_dim = params['graph_dim'],
                        num_nodes = params['num_nodes'],
                        last_dense = params['dense']
                       ).cuda()
    loss_func = nn.CrossEntropyLoss(weight = torch.FloatTensor(params['loss_score']).cuda())
    
    if params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    if params['lr_decay'] != 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=params['lr_decay'])

    val_loss_before = np.inf
    best_acc = 0
    count_epoch = 0
    tr_acc_list_e = []
    tr_loss_list_e = []
    tr_kl1_list_e = []
    tr_kl2_list_e = []
    val_acc_list_e = []
    val_loss_list_e = []
    val_kl1_list_e = []
    val_kl2_list_e = []
    for epoch in range(params['minEpoch']):
        time_start = time.time()

        tr_acc, tr_loss, tr_kl1, tr_kl2 = train_epoch(model, trGen, params, loss_func, optimizer, epoch)
        va_acc, va_loss, va_kl1, va_kl2 = val(model, cvGen, params, loss_func, epoch, False)

        if params['lr_decay']>0:
            lr_epoch = scheduler.get_last_lr()
        else:
            lr_epoch = params['lr']
        tr_acc_list_e.append(tr_acc)
        tr_loss_list_e.append(tr_loss)
        tr_kl1_list_e.append(tr_kl1)
        tr_kl2_list_e.append(tr_kl2)
        val_acc_list_e.append(va_acc)
        val_loss_list_e.append(va_loss)
        val_kl1_list_e.append(va_kl1)
        val_kl2_list_e.append(va_kl2)

        if val_loss_before < va_loss:
            if count_epoch >= 2:
                if params['lr_decay']>0:
                    scheduler.step()
                val_loss_before = np.inf
                count_epoch = 0
            else:
                val_loss_before = va_loss
                count_epoch += 1
        else:
            val_loss_before = va_loss

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), othercfg['out_dir'] + 'best_model_' + str(i) + '.nnet.pth')
            print(" U ", end='')
        time_end = time.time()
        time_cost = time_end - time_start
        print(" Time:%.3f" % (time_cost), 'lr:', lr_epoch)

    print(28 * "=", 'Fold #', i, 'Final', 28 * '=')
    print(time.asctime(time.localtime(time.time())))
    model.eval()
    model.load_state_dict(torch.load(othercfg['out_dir'] + 'best_model_' + str(i) + '.nnet.pth'))
    _, _, _, _, pred, true, summ, spec = val(model, cvGen, params, loss_func, epoch, True)
    pred_list.append(pred)
    true_list.append(true)
    summ_graph_list.append(summ)
    spec_graph_list.append(spec)
    tr_acc_list.append(tr_acc_list_e)
    tr_loss_list.append(tr_loss_list_e)
    tr_kl1_list.append(tr_kl1_list_e)
    tr_kl2_list.append(tr_kl2_list_e)
    val_acc_list.append(val_acc_list_e)
    val_loss_list.append(val_loss_list_e)
    val_kl1_list.append(val_kl1_list_e)
    val_kl2_list.append(val_kl2_list_e)

    torch.save(model, othercfg['out_dir'] + 'model_' + str(i) + '.nnet.pth')
    torch.cuda.empty_cache()
    del trainX, valX, trainY, valY, trGen, cvGen, trDataset, cvDataset, loss_func, optimizer
    gc.collect()

true_list = np.concatenate(true_list)
pred_list = np.concatenate(pred_list)
summ_graph_list = np.concatenate(summ_graph_list)
spec_graph_list = np.concatenate(spec_graph_list)

##################### Accuracy\Loss curve ######################
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
for i in range(5):
    plt.plot(tr_acc_list[i], label='Fold %d' % (i))
plt.title('Training Accuracy')
plt.legend()
plt.subplot(2, 2, 2)
for i in range(5):
    plt.plot(tr_loss_list[i], label='Fold %d' % (i))
plt.title('Training Loss')
plt.legend()
plt.subplot(2, 2, 3)
for i in range(5):
    plt.plot(val_acc_list[i], label='Fold %d' % (i))
plt.title('Validation Accuracy')
plt.legend()
plt.subplot(2, 2, 4)
for i in range(5):
    plt.plot(val_loss_list[i], label='Fold %d' % (i))
plt.title('Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(othercfg['out_dir'] + 'Curve.png')
plt.draw()
print()

######################## Model metrics #########################
print('\n')
print(10 * '=', 'End Of Training', 10 * '=')
print(time.asctime(time.localtime(time.time())))
PrintScore(true_list, pred_list)
PrintScore(true_list, pred_list, savePath=othercfg['out_dir'])
ConfusionMatrix(true_list, pred_list, savePath=othercfg['out_dir'])
print('===== Succeed =====', time.asctime(time.localtime(time.time())))
