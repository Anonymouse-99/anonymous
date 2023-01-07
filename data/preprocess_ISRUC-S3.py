from os import path
import numpy as np
from scipy import signal
import scipy.io as scio

################################################################################
######## Processing the ISRUC-S3 dataset into 5-fold cross-subject data ########
################################################################################


'''
Path and channel settings
'''
path_Extracted = './ISRUC_S3/ExtractedChannels/'
path_RawData   = './ISRUC_S3/RawData/'
path_output    = './ISRUC_S3/'
channels       = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1']
resample_freq  = 100


'''
Read function
'''
def read_psg(path_Extracted, sub_id, channels, freq=3000):
    psg = scio.loadmat(path.join(path_Extracted, 'subject%d.mat' % (sub_id)))
    psg_use = []
    for c in channels:
        psg_use.append(
            np.expand_dims(signal.resample(psg[c], freq*30, axis=-1), 1))
    psg_use = np.concatenate(psg_use, axis=1)
    return psg_use


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


'''
output:
    save to $path_output/$fold_id.npz:
        Fold_data:  [k-fold] list, each element is [N,C,T]
        Fold_label: [k-fold] list, each element is [N]
'''

fold_label = []
fold_psg = []
fold_len = []

for sub in range(1, 11):
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_Extracted, sub, channels, resample_freq)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in original ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM
    fold_label.append(label)
    fold_psg.append(psg)
    fold_len.append(len(label))
    if sub % 2 == 0:
        data = np.concatenate(fold_psg)
        label = np.concatenate(fold_label)
        print('Fold', sub//2 , data.shape, label.shape)
        print('==>', path.join(path_output, '%d.npz'%(sub//2)))
        np.savez(path.join(path_output, '%d.npz'%(sub//2)),
                 data = data,
                 label = label,
                 len = np.sum(fold_len),
                 channels = channels)
        print('-'*36)
        fold_label = []
        fold_psg = []
        fold_len = []

print('Preprocess over.', 'Saved to', path_output)
