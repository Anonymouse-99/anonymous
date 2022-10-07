import torch.nn as nn

'''
Single channel EEG feature extraction
'''
def FeatureNet(s_freq, filters=128, dropout=0.5):
    model = nn.Sequential(
        nn.Conv1d(1, filters, kernel_size=s_freq//2, stride=s_freq//4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.MaxPool1d(8, 8),
        nn.Dropout(dropout),
        nn.Conv1d(filters, filters, kernel_size=8, stride=1, padding=4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, filters, kernel_size=8, stride=1, padding=4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.Conv1d(filters, filters, kernel_size=8, stride=1, padding=4),
        nn.BatchNorm1d(filters),
        nn.ReLU(),
        nn.MaxPool1d(4, 4),
        nn.Dropout(dropout),
        nn.Flatten()
    )
    return model


'''
Multi-channel EEG feature extraction
'''
class FeatureNet_MC(nn.Module):
    def __init__(self, s_freq, filters=128):
        super(FeatureNet_MC, self).__init__()
        self.tiny_model = FeatureNet(s_freq, filters)

    def forward(self, X):
        B, C, T = X.shape
        X = X.reshape(B * C, 1, T)
        H = self.tiny_model(X)
        return H.reshape(B, C, -1)
