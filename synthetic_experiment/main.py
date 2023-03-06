import math

from scipy.io import savemat
from sklearn.datasets import make_circles, make_moons, make_multilabel_classification, make_classification, \
    make_gaussian_quantiles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from torch import nn
from torch.utils.data import DataLoader

from CustomTensorDataset import CustomTensorDataset
from Gaussian import Gaussian
from Model.MLP import QMLP, MLP
from train_function import group_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

def train(chosen_model, dataloader, lr, slr, epochs, D, neurons):
    if chosen_model == 'qmlp':
        net = QMLP(D, neurons)
    if chosen_model == 'mlp':
        net = MLP(D, neurons)
    # 如果有GPU，载入到CUDA()
    if use_gpu:
        net.cuda()

    acc_max = 0
    train_loss = []
    train_acc = []
    for e in range(epochs):
        loss = 0
        total = 0
        correct = 0
        loss_total = 0
        net.train()  # 让模型训练

        for step, (x, y) in enumerate(dataloader):

            # x = x.type(torch.float)
            # y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            if chosen_model == 'qmlp':
                group = group_parameters(net)
                optimizer = torch.optim.Adam([
                    {"params": group[0], "lr": lr},  # weight_r
                    {"params": group[1], "lr": lr * slr},  # weight_g
                    {"params": group[2], "lr": lr * slr},  # weight_b
                    {"params": group[3], "lr": lr},  # bias_r
                    {"params": group[4], "lr": lr * slr},  # bias_g
                    {"params": group[5], "lr": lr * slr},  # bias_b
                    {'params': group[6],"lr": lr}, # normal weight
                    {'params': group[7], "lr": lr} # normal bias
                ], lr=lr, weight_decay=1e-4)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                                 eta_min=1e-8)  # goal: maximize Dice score
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器， 梯度下降训练算法
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
            loss_func = nn.CrossEntropyLoss()  # 交叉熵损失
            # loss_func = LabelSmoothingCrossEntropy()
            if use_gpu:
                y_hat = net(x).cuda()
            else:
                y_hat = net(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += loss.item()
            y_predict = y_hat.argmax(dim=1)

            total += y.size(0)
            if use_gpu:
                correct += (y_predict == y).cpu().squeeze().sum().numpy()
            else:
                correct += (y_predict == y).squeeze().sum().numpy()

            if step % 20 == 0:
                print('Epoch:%d, Step [%d/%d], Loss: %.4f' % (e + 1, step + 1, len(dataloader.dataset), loss_total / len(dataloader.dataset)))
        loss_total = loss_total / len(dataloader.dataset)
        acc = correct / total
        print("Current Acc: %f" % (acc) )
        train_loss.append(loss_total)
        train_acc.append(acc)
        if not os.path.exists('models'):
            os.mkdir('models')
        if acc_max <= acc:
            acc_max = acc
            torch.save(net.state_dict(), os.path.join("models", "checkpoint.pth"))

    return net


def inference(dataloader, chosen_model, D, neurons):
    if chosen_model == 'qmlp':
        net = QMLP(D, neurons)
    if chosen_model == 'mlp':
        net = MLP(D, neurons)

    best_model_dict = torch.load(os.path.join("models", 'checkpoint.pth'))
    net.load_state_dict(best_model_dict)
    y_list, y_predict_list = [], []
    if use_gpu:
        net.cuda()
    net.eval()
    # endregion
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            # x = x.type(torch.float)
            # y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())

        acc = accuracy_score(y_list, y_predict_list)
        print("Test Acc:%f" % acc)

def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_decision_boundary(X, y, chosen_model, D, neurons):
    if chosen_model == 'qmlp':
        model = QMLP(D, neurons)
    if chosen_model == 'mlp':
        model = MLP(D, neurons)
    if chosen_model == 'cnn':
        model = LeNet(10)
    if chosen_model == 'qcnn':
        model = QLeNet(10)
    best_model_dict = torch.load(os.path.join("models", 'checkpoint.pth'))
    model.load_state_dict(best_model_dict)

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    Z = model(X_test)
    y_predict = Z.argmax(dim=1)
    y_predict = y_predict.reshape(xx.shape)
    plt.figure(figsize=(0.98, 0.9))
    plt.rc('font', family='Times New Roman')
    plt.contourf(xx, yy, y_predict.detach().numpy(), cmap=plt.cm.Spectral, alpha=0.5)
    # plt.ylabel('x2')
    # plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], s=0.1, c=np.squeeze(y), cmap=plt.cm.rainbow_r)
    plt.axis('off')
    # plt.tick_params(labelsize=1)
    save_path = 'figure/%s_multi4_%d.tif' % (chosen_model, neurons[0])
    plt.savefig(save_path, bbox_inches='tight', dpi=1200, pad_inches=0.05)
    plt.close()
    plt.show()

    # savemat('results/%s_moon_%d.mat' % (chosen_model, neurons[0]), {'X': X,
    #                                                                    'y': y,
    #                                                                    'xx': xx,
    #                                                                    'yy': yy,
    #                                                                    'y_pre': y_predict.detach().numpy()})



    # Get a random point on a unit hypersphere of dimension n
def random_hypersphere_point(n, scale_factor=1.0):
    # fill a list of n uniform random values
    points = [np.random.normal(0, 1) for r in range(n)]
    points = np.array(points)
    # calculate 1 / sqrt of sum of squares
    sqr_red = 1.0 / math.sqrt(sum(i * i for i in points))
    # multiply each point by scale factor 1 / sqrt of sum of squares
    return list(map(lambda x: x * sqr_red * scale_factor, points))

def hyper_concentric_circle(n_sample=2000, n_feature=2, scale_factor=0.7, noise=None, random_state=42):
    generator = check_random_state(random_state)

    out_circle = []
    in_circle = []
    for i in range(n_sample // 2):
        out_circle.append(random_hypersphere_point(n_feature))
    for j in range(n_sample // 2):
        in_circle.append(random_hypersphere_point(n_feature, scale_factor))
    out_y = np.zeros(n_sample // 2, dtype=np.intp)
    in_y = np.ones(n_sample // 2, dtype=np.intp)
    X = np.vstack([out_circle, in_circle])
    y = np.hstack([out_y, in_y])
    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)
    return X, y


if __name__ == '__main__':

    chosen_model = 'qmlp' # qmlp mlp
    chosen_data = 'circle' # moon circle multi
    lr = 0.01
    slr = 2
    bs = 64  # input batch size for training (default: 64)
    epochs = 50
    neurons = [10, 2]
    seed = 42
    random_seed(seed)
    D = 2
    # circles
    if chosen_data == 'circle':
        X, y = make_circles(n_samples=2000, shuffle=True, noise=0.03, factor=0.7, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # multi
    if chosen_data == 'multi':
        X, y = make_classification(n_samples=5000, n_features=D, n_informative=50, n_redundant=0, n_repeated=0,
                                   n_classes=10, n_clusters_per_class=1, weights=None,
                                   hypercube=True, shift=0, scale=1.0, shuffle=True, random_state=seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    if chosen_data == 'hypercircle':
        X, y = hyper_concentric_circle(n_sample=2000, n_feature=D, scale_factor=0.7, noise=0.03, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


    if chosen_model in ('cnn', 'qcnn'):
        X_train = X_train[:, np.newaxis, :]
        X_test = X_test[:, np.newaxis, :]

    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    train_dataset = CustomTensorDataset(torch.tensor(X_train, dtype=torch.float),
                                        torch.tensor(y_train, dtype=torch.long))
    # valid_dataset = CustomTensorDataset(torch.tensor(G.valid_point, dtype=torch.float), torch.tensor(G.valid_label))
    test_dataset = CustomTensorDataset(torch.tensor(X_test, dtype=torch.float),
                                       torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    net = train(chosen_model, train_loader, lr, slr, epochs, D, neurons)
    # plot_decision_boundary(X_test, y_test, chosen_model, D, neurons)
    #
    inference(test_loader, chosen_model, D, neurons)


