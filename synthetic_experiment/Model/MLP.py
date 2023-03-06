import torch
import torch.nn as nn
from fvcore.nn import flop_count_str, FlopCountAnalysis

from Model.QuadraticOperation import QuadraticOperation
import torch.nn.functional as F

def get_activation_by_name(name=str):
    activations={
        "relu":nn.ReLU(),
        "sigmoid":nn.Sigmoid(),
        "tanh":nn.Tanh()
    }

    if name in activations.keys():
        return activations[name]

    else:
        raise ValueError(name,"is not a valid activation function")

class MLP(nn.Module):

    def __init__(self,
                 n_features,
                 hidden_neurons=[2,4,4,2],
                 activation='relu',
                 dropout=False,
                 dropout_rate=0.5,
                 batch_norm=False):
        super(MLP, self).__init__()
        self.n_features=n_features
        self.hidden_neurons=hidden_neurons
        self.activation_name=activation   #函数名称，str
        self.drop_rate=dropout_rate
        self.dropout=dropout
        self.batch_norm=batch_norm

        self.activation = get_activation_by_name(self.activation_name)  #函数

        self.layer_neurons_=[self.n_features,*hidden_neurons]  #每一层的参数
        self.length_layer=len(self.layer_neurons_)-1           #隐藏层长度
        self.mlp=nn.Sequential()


        for idx,layer in enumerate(self.layer_neurons_[:-1]):
            if batch_norm:
                self.mlp.add_module("batch_norm"+str(idx),
                                    nn.BatchNorm1d(self.layer_neurons_[idx]))
            else:
                self.mlp.add_module("layer"+str(idx),
                                    nn.Linear(self.layer_neurons_[idx],self.layer_neurons_[idx+1]))

            if not idx+1 == self.length_layer : #判断是不是输出层，最后一层不加激活函数
                self.mlp.add_module(self.activation_name+str(idx),
                                    self.activation)
                if dropout:
                    self.mlp.add_module("dropout"+str(idx),
                                        nn.Dropout(dropout_rate))
    def forward(self,x):
        y=self.mlp(x)
        y=F.softmax(y, dim=1)
        return y


class QMLP(nn.Module):

    def __init__(self,
                 n_features,
                 hidden_neurons=[2,4,4,2],
                 activation='relu',
                 dropout=False,
                 dropout_rate=0.5,
                 batch_norm=False):
        super(QMLP, self).__init__()
        self.n_features=n_features
        self.hidden_neurons=hidden_neurons
        self.activation_name=activation   #函数名称，str
        self.drop_rate=dropout_rate
        self.dropout=dropout
        self.batch_norm=batch_norm

        self.activation = get_activation_by_name(self.activation_name)  #函数

        self.layer_neurons_=[self.n_features,*hidden_neurons]  #每一层的参数
        self.length_layer=len(self.layer_neurons_)-1           #隐藏层长度
        self.mlp = nn.Sequential()


        for idx,layer in enumerate(self.layer_neurons_[:-1]):
            if batch_norm:
                self.mlp.add_module("batch_norm"+str(idx),
                                    nn.BatchNorm1d(self.layer_neurons_[idx]))
            else:
                self.mlp.add_module("layer"+str(idx),
                                    QuadraticOperation(self.layer_neurons_[idx], self.layer_neurons_[idx+1]))

            if not idx+1 == self.length_layer : #判断是不是输出层，最后一层不加激活函数
                self.mlp.add_module(self.activation_name+str(idx),
                                    self.activation)
                if dropout:
                    self.mlp.add_module("dropout"+str(idx),
                                        nn.Dropout(dropout_rate))
    def forward(self,x):
        y=self.mlp(x)
        y=F.softmax(y, dim=1)
        return y


if __name__=="__main__":


    ##获取Sequential的中间层输出

    net=MLP(500,[120, 10])
    x=torch.rand(1, 500)
    y=net(x)
    print(flop_count_str(FlopCountAnalysis(net, x)))

    # for idx,item in enumerate(net.named_children()):
    #    if idx==1:
    #        print(item)
    # print(net.mlp)
    # print(type(net.mlp))
    # print(list(net.mlp))
    # for i in list(net.mlp): #不要list也可以
    #     #print(i)
    #     x = i(x)
    #     if isinstance(i,nn.ReLU):
    #         print(x)