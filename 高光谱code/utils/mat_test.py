import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim


class HybridSN(nn.Module):
    def __init__(self, in_channels=1):
        super(HybridSN, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU()
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 16)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = x.view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], x.size()[4])
        x = self.conv2d_features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_num = 16
net = HybridSN()
net.load_state_dict(torch.load('utils/lenet_state_dict.pth'))
net.to(device)
net.eval()

print("模型加载完毕")


def test_mat(mat1, mat2, result_path):
    # 用于测试样本的比例
    test_ratio = 0.90
    # 每个像素周围提取 patch 的尺寸
    patch_size = 25
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    # load the original image
    X = sio.loadmat(mat1)
    X = X[list(X.keys())[-1]]
    y = sio.loadmat(mat2)
    y = y[list(y.keys())[-1]]

    height = y.shape[0]
    width = y.shape[1]

    X = applyPCA(X, numComponents=pca_components)
    X = padWithZeros(X, patch_size // 2)

    # 逐像素预测类别
    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if int(y[i, j]) == 0:
                continue
            else:
                image_patch = X[i:i + patch_size, j:j + patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                prediction = net(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')

    predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(5, 5))
    plt.savefig(result_path)
    return True
