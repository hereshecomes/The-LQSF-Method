from d2l import torch as d2l
import torch
from torch import nn
def train_from_scratch(net, optimizer, loss, num_epochs, train_iter, device): #打印损失
    net.to(device)
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    for epoch in range(num_epochs):  
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                if (i+1) % 1000 == 0:
                    print(l)

def continue_train(net, optimizer, loss, num_epochs, train_iter, device): ##继续训练，打印损失
    for epoch in range(num_epochs):
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                if (i+1) % 1000 == 0:
                    print(l)
                        
def evalute_results(net, data_iter, threshold_value, device): ##最后输出一个值，模型在验证集上的平均正确率,默认模型在显卡上
    net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            y_pred = torch.where(output>threshold_value,torch.ones_like(output),torch.zeros_like(output))
            cmp = d2l.astype(y_pred, y.dtype) == y
            acc_num = d2l.reduce_sum(d2l.astype(cmp, y.dtype))
            metric.add(acc_num, y.numel())
    return metric[0] / metric[1]

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            output = net(X)
            y_pred = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
            metric.add(d2l.accuracy(y_pred, y), y.numel())
    return metric[0] / metric[1]

def train_with_animator(net, optimizer, loss, num_epochs, train_iter, test_iter, threshold_value, device):
    animator = d2l.Animator(xlabel='epoch', xlim=[0, epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    net.to(device)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                train_acc_num = torch.where(y_hat>threshold_value,torch.ones_like(y_hat),torch.zeros_like(y_hat))
                metric.add(l * X.shape[0], d2l.accuracy(train_acc_num, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')