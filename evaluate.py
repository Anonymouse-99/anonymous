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


print('=====Start(Eval)=====', time.asctime(time.localtime(time.time())))
print('PyTorch:', torch.__version__)

##################### load learning config #####################
config = np.load(path.join('./run_ISRUC/', 'params.npz'), allow_pickle=True)
params = config['params'].item()
othercfg = config['othercfg'].item()
print('[Info] Config:')
print('params:', params)
print('othercfg:', othercfg)

####################### Set random seed ########################
set_random_seed(params['seed'])

########################### Training ###########################
pred_list = []
true_list = []
summ_graph_list=[]
spec_graph_list=[]

print('[Info]', time.asctime(time.localtime(time.time())), 'Start training:')
for i in range(othercfg['fold']):
    print()
    print(28 * "=", 'Fold #', i, 'Eval', 28 * '=')
    print(time.asctime(time.localtime(time.time())))
    # Load data from .npz files
    fold = np.load(path.join(othercfg['data_path'], '%d.npz' % (i+1)))
    valX = np.float32(fold['data'])
    valY = np.int64(fold['label'])
    print('[Data] Val:', valY.shape)

    # Organize data to Torch
    cvDataset = SimpleDataset(valX, valY)
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

    model.eval()
    model.load_state_dict(torch.load(othercfg['out_dir'] + 'best_model_' + str(i) + '.nnet.pth'))
    va_acc, va_loss, va_kl1, va_kl2, pred, true, summ, spec = val(model, cvGen, params, loss_func, 0, True)
    pred_list.append(pred)
    true_list.append(true)
    summ_graph_list.append(summ)
    spec_graph_list.append(spec)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Summary graph (Fold #%d)'%i)
    summ_mean = summ.mean(axis=0)
    plt.imshow(row2matrix(summ_mean, params['num_nodes']))
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title('Specific graph (Fold #%d)'%i)
    spec_mean = spec.mean(axis=0)
    plt.imshow(row2matrix(spec_mean, params['num_nodes']))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(othercfg['out_dir'] + 'graph'+str(i)+'.png')
    plt.draw()

    torch.save(model, othercfg['out_dir'] + 'model_' + str(i) + '.nnet.pth')
    torch.cuda.empty_cache()
    del valX, valY, cvGen, cvDataset, loss_func
    gc.collect()

true_list = np.concatenate(true_list)
pred_list = np.concatenate(pred_list)
summ_graph_list = np.concatenate(summ_graph_list)
spec_graph_list = np.concatenate(spec_graph_list)


######################## Model metrics #########################
print('\n')
print(10 * '=', 'End Of Training', 10 * '=')
print(time.asctime(time.localtime(time.time())))
PrintScore(true_list, pred_list)
PrintScore(true_list, pred_list, savePath=othercfg['out_dir'])
ConfusionMatrix(true_list, pred_list, savePath=othercfg['out_dir'])

####################### Save graph edges #######################
np.save(othercfg['out_dir']+'graph_summary.npy',summ_graph_list)
np.save(othercfg['out_dir']+'graph_specific.npy',spec_graph_list)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Summary graph')
summ_mean = summ_graph_list.mean(axis=0)#keepdims=True
plt.imshow(row2matrix(summ_mean, params['num_nodes']))
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title('Specific graph')
spec_mean = spec_graph_list.mean(axis=0)#keepdims=True
plt.imshow(row2matrix(spec_mean, params['num_nodes']))
plt.colorbar()
plt.tight_layout()
plt.savefig(othercfg['out_dir'] + 'graph.png')
plt.draw()

print()
print('===== Succeed =====', time.asctime(time.localtime(time.time())))
