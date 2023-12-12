import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random
from tqdm import tqdm
import heartrate
import datetime
# from torch.utils.tensorboard import SummaryWriter
import itertools

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser()
    # 模型架构：Options: simple-cnn, vgg, resnet, mlp. Default = mlp.
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    # 数据集：Dataset to use. Options: mnist, cifar10, fmnist, svhn, generated, femnist, a9a, rcv1, covtype. Default = mnist.
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    # 网络配置
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    # 数据集划分方式：The partition way. Options: homo, noniid-labeldir, noniid-#label1 (or 2, 3, ..., which means the fixed number of labels each party owns), real, iid-diff-quantity. Default = homo
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    # Batch size, default = 64
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    # test batch size, default = 32
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 32)')
    # Learning rate for the local models, default = 0.01.
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    # local epochs数目： Default = 10
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    # calibration epochs数目： Default = 5
    parser.add_argument('--calibration_epochs', type=int, default=5, help='number of calibration epochs')
    # calibration reg权重： Default = 1.0
    parser.add_argument('--ccreg_w', type=float, required=False, default=1.0, help="Weight of ccreg. Default=1.0")
    # 参与方数量：Number of parties, default = 2.
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    # percentage of clients to refine.
    parser.add_argument('--cc_p', type=float, default=0.5, help='percentage of clients to refine')
    # FL算法：The training algorithm. Options: fedavg, fedprox, scaffold, fednova, moon. Default = fedavg.
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    # （MOON算法选项）
    parser.add_argument('--use_projection_head', type=bool, default=False,
                        help='whether add an additional header to model or not (see MOON)')
    # projection层的输出维度
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    # （MOON算法选项）
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    # contrastive loss用的温度
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    # 最大的训练轮次：Number of communication rounds to use, default = 50
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    # 是否使用与FedAvg相同的初始化
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    # random seed
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    # 网络中的dropout参数
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    # 数据集的路径
    parser.add_argument('--datadir', type=str, required=False, default="../data/", help="Data directory")
    # L2 正则化
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    # 日志的路径
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    # 定义模型的文件路径
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    # 狄利克雷分布参数
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    # 运行设备：Specify the device to run the program, default = cuda:0.
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    # 日志文件名称
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    # 优化器
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    # cc优化器
    parser.add_argument('--cc_optimizer', type=str, default='adam', help='the cc optimizer')
    # （FedProx参数）
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    # （Classifier Calibration参数）
    parser.add_argument('--mu_cc', type=float, default=1, help='the mu parameter for fedprox')
    # 数据噪声
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    # 噪声类型：Maximum variance of Gaussian noise we add to local party, default = 0.
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')
    # （momentum SGD参数）
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    # 每轮的采样比例：Ratio of parties that participate in each communication round, default = 1.
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    # 模型保存路径
    parser.add_argument('--save_path', type=str, default='./saved_model', help='Path for saving model')
    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    # 根据不同的网络，配置n_classes参数
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2

    # 根据args.model，给各方赋值网络
    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model + add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model + add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16, 8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg":
                    net = vgg11()
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                elif args.model == "vgg-9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                elif args.model == "resnet":
                    net = ResNet50_cifar10()
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    # model_meta_data：记录网络每层的形状
    # layer_type：网络每层的类型
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


# FedAvg算法训练过程
def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # 计算预训练前的训练集和测试集准确率
    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # 根据选择的优化器类型进行优化器的初始化
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            loop = tqdm(enumerate(tmp), total=len(tmp))
            for batch_idx, (x, target) in loop:
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                _, prediction = out.max(1)
                num_correct = (prediction == target).sum()
                running_train_acc = float(num_correct) / float(x.shape[0])

                cnt += 1
                epoch_loss_collector.append(loss.item())

                # 更新loop信息
                loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
                loop.set_postfix(loss=loss.item(), acc=running_train_acc)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # 训练完成后计算训练集和测试集的准确率
    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


# FedProx算法训练过程
def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (x, target) in loop:
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            _, prediction = out.max(1)
            num_correct = (prediction == target).sum()
            running_train_acc = float(num_correct) / float(x.shape[0])

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

            # 更新loop信息
            loop.set_description(f'Epoch [{epoch}/{epochs}]')
            loop.set_postfix(loss=loss.item(), acc=running_train_acc)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


# SCAFFOLD算法训练过程
def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr,
                       args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para


# FedNova算法训练过程
def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                          weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)
    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


# MOON算法训练过程
def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                   args_optimizer, mu, temperature, args,
                   round, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy_class(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2 - pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    train_acc = compute_accuracy_class(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


# classifier calibration算法step1
def train_net_classifier_calibration(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr,
                                     args_optimizer, mu, dataidxs, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    # Step1: local train on private data
    for epoch in range(epochs):
        epoch_loss_collector = []
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0, leave=True)
        for batch_idx, (x, target) in loop:
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            _, prediction = out.max(1)
            num_correct = (prediction == target).sum()
            running_train_acc = float(num_correct) / float(x.shape[0])

            cnt += 1
            epoch_loss_collector.append(loss.item())

            # 更新loop信息
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(loss=loss.item(), acc=running_train_acc)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # Step2: compute feature norm on public data
    loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), position=0, leave=True)

    # 创建一个字典，用于存储每个类别的 feature norms 和样本数量
    class_feature_norms = defaultdict(list)
    class_sample_counts = defaultdict(int)
    class_avg_feature_norms = defaultdict(float)
    for batch_idx, (x, target) in loop:
        x, target = x.to(device), target.to(device)

        optimizer.zero_grad()
        x.requires_grad = True
        target.requires_grad = False
        target = target.long()

        out = net(x)

        # 获取feature，并计算按类别计算feature norm
        features['output'] = features['output'].reshape(x.size(0), features['output'].shape[1])
        feature_norms = torch.norm(features['output'], p=2, dim=1).cpu().detach().numpy()

        # 获取当前 batch 中每个样本的 class
        for i in range(x.size(0)):
            class_label = target[i].item()
            class_feature_norms[class_label].append(feature_norms[i])
            class_sample_counts[class_label] += 1

        cnt += 1
        # 更新loop信息
        loop.set_description(f'Batch [{batch_idx}/{len(test_dataloader)}]')

    # 计算每个类别的平均 feature norm
    for class_label, norms in class_feature_norms.items():
        if class_sample_counts[class_label] > 0:
            avg_feature_norm = sum(norms) / class_sample_counts[class_label]
            class_avg_feature_norms[class_label] = avg_feature_norm
            print(f'Class {class_label}, Avg Feature Norm: {avg_feature_norm}')

    logger.info('class_avg_feature_norms length: %s' % (len(class_avg_feature_norms)))    # 313

    # 记录该client本round的acc
    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')

    return net.state_dict(), train_acc, test_acc, class_avg_feature_norms


# classifier calibration算法step2
def classifier_calibration(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr,
                           args_optimizer, mu, dataidxs, class_norm_diffs, device="cpu"):
    logger.info('Calibrate network %s classifier' % str(net_id))

    net.to(device)
    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    epoch_loss_collector = []
    epoch_acc_collector = []
    global_weight_collector = list(global_net.to(device).parameters())
    for epoch in range(args.calibration_epochs):  # hard code
        loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for batch_idx, (x, target) in loop:
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)

            loss = criterion(out, target)
            # for classifier calibration

            # ---------------------------------cc----------------------------------------------------
            # 计算当前批次类别的平均特征规范差异
            current_class_labels = target.unique()
            class_weights = []  # 用于存储每个类别的权重
            for class_label in current_class_labels:
                class_label = int(class_label.cpu().numpy())
                diff = class_norm_diffs[class_label]

                # 计算类别的权重，以样本数量占比作为权重
                class_mask = (target == class_label)
                class_count = class_mask.sum().item()
                total_samples = len(target)
                class_weight = class_count / total_samples
                loss += args.ccreg_w * class_weight * diff


            _, prediction = out.max(1)
            num_correct = (prediction == target).sum()
            running_train_acc = float(num_correct) / float(x.shape[0])

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_acc_collector.append(running_train_acc)

            # 更新loop信息
            loop.set_description(f'Epoch [{epoch}/{args.calibration_epochs}]')
            loop.set_postfix(loss=loss.item(), acc=running_train_acc)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_acc = sum(epoch_acc_collector) / len(epoch_acc_collector)

        # 记录本epoch信息
        logger.info('Epoch: %d Acc: %f Loss: %f' % (epoch, epoch_acc, epoch_loss))

    train_acc = compute_accuracy_class(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy_class(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Classifier calibration complete **')
    return net.state_dict(), train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


def calculate_feature_norm_diff(feature_norm_all_clients, net_to_train):
    # 创建用于存储特征规范差异的字典
    class_norm_diffs = {net_to_train[i]: {} for i in range(len(net_to_train))}

    # 遍历要训练的客户端
    for i in range(len(net_to_train)):
        current_client = net_to_train[i]

        # 遍历其他客户端
        for j in range(len(feature_norm_all_clients)):
            if j != current_client:
                for class_label in feature_norm_all_clients[current_client].keys():
                    if class_label in feature_norm_all_clients[j]:
                        # 计算特征规范差异
                        diff = feature_norm_all_clients[j][class_label] - feature_norm_all_clients[current_client][class_label]
                        # 累加差异值
                        if class_label in class_norm_diffs[current_client]:
                            class_norm_diffs[current_client][class_label] += diff
                        else:
                            class_norm_diffs[current_client][class_label] = diff


    # 打印结果或进一步处理
    for i in range(len(net_to_train)):
        logger.info("Client %s feature norm difference: %s" % (str(net_to_train[i]), str(class_norm_diffs[net_to_train[i]])))
    return class_norm_diffs


features = dict()


def forward_hook(module, input, output):
    features['input'] = input
    features['output'] = output


def local_train_net(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            # 获取本地训练集和测试集的数据加载器（用于空间噪声）
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            # 获取本地训练集和测试集的数据加载器（用于随机噪声）
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level)

        # 获取全局训练集和测试集的数据加载器
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size)
        n_epoch = args.epochs

        # 进行网络训练
        trainacc, testacc = train_net(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                      device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl=None,
                             device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size)
        n_epoch = args.epochs

        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global,
                                                             train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                                             device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            # print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size)
        n_epoch = args.epochs

        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch,
                                                        args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model=None, prev_model_pool=None,
                         round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size)
        n_epoch = args.epochs

        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch,
                                           args.lr,
                                           args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list


def local_train_net_classifier_calibration(nets, selected, global_model, args, net_dataidx_map, test_dl=None,
                                           device="cpu"):
    avg_acc = 0.0

    # step 1: 各方训练
    # feature_norm_all_clients = [[] for _ in range(len(selected))]
    feature_norm_all_clients = []
    testacc_dict = {}
    train_dl_local_list = []
    test_dl_local_list = []
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        # 设置hook，获取中间层输出以获取feature norm
        net.avgpool.register_forward_hook(forward_hook)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size,
                                                                 dataidxs, noise_level)
        train_dl_local_list.append(train_dl_local)
        test_dl_local_list.append(test_dl_local)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, args.test_batch_size)
        n_epoch = args.epochs

        # test_dl = test_dl_global
        net_train_tmp_para, trainacc, testacc, class_avg_feature_norms = train_net_classifier_calibration(net_id, net, global_model,
                                                                                train_dl_local, test_dl_global, n_epoch,
                                                                                args.lr,
                                                                                args.optimizer, args.mu, dataidxs,
                                                                                device=device)
        # testacc = compute_accuracy_class(net, test_dl_global, device=device)
        # logger.info("Step 1: net %d final test acc %f" % (net_id, testacc))
        feature_norm_all_clients.append(class_avg_feature_norms)

        avg_acc += testacc
        testacc_dict[net_id] = testacc
        nets[net_id].load_state_dict(net_train_tmp_para)

    # ---------------------------------------------------------------------------------------------------
    # 按照数据集大小，计算fedavg聚合时各方权重
    total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

    print(fed_avg_freqs)

    for idx in range(len(selected)):
        net_para = nets[selected[idx]].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                global_para[key] = net_para[key] * fed_avg_freqs[idx]
        else:
            for key in net_para:
                global_para[key] += net_para[key] * fed_avg_freqs[idx]
    global_model.load_state_dict(global_para)

    global_model.to(device)
    train_acc = compute_accuracy_class(global_model, train_dl_global, device=device)
    test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                             device=device)

    logger.info('>> Global Model Train accuracy: %f' % train_acc)
    logger.info('>> Global Model Test accuracy: %f' % test_acc)
    # ---------------------------------------------------------------------------------------------------

    # step 2: 使用feature norm进行校准

    # 计算要选择的客户端数量
    num_clients = len(testacc_dict)
    num_to_select = int(num_clients * args.cc_p)

    # 选择测试精度最低的客户端
    net_to_train = [k for k, v in sorted(testacc_dict.items(), key=lambda item: item[1], reverse=False)[:num_to_select]]

    print("Selected clients:", net_to_train)

    # 计算网络之间的对应类别的平均特征规范差异
    total_diffs = calculate_feature_norm_diff(feature_norm_all_clients, net_to_train)
    logger.info("total_diffs: %s"%str(total_diffs))

    for net_id in net_to_train:
        net_cc_tmp_para, trainacc, testacc = \
            classifier_calibration(net_id, nets[net_id], global_model, train_dl_local_list[net_id], test_dl_global, n_epoch, args.lr,
                                   args.cc_optimizer, args.mu, dataidxs, total_diffs[net_id], device=device)

        avg_acc /= len(selected)
        nets[net_id].load_state_dict(net_cc_tmp_para)

    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())

    # ---------------------------------------------------------------------------------------------------
    # 按照数据集大小，计算fedavg聚合时各方权重
    total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

    print(fed_avg_freqs)
    # logger.info("Testing global model para before classifier calibration: %s" % (str(global_para)))
    # logger.info("Testing global model para before classifier calibration")
    for idx in range(len(selected)):
        net_para = nets[selected[idx]].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                global_para[key] = net_para[key] * fed_avg_freqs[idx]
        else:
            for key in net_para:
                global_para[key] += net_para[key] * fed_avg_freqs[idx]
    global_model.load_state_dict(global_para)

    global_model.to(device)
    train_acc = compute_accuracy_class(global_model, train_dl_global, device=device)
    test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                             device=device)

    logger.info('>> Global Model Train accuracy: %f' % train_acc)
    logger.info('>> Global Model Test accuracy: %f' % test_acc)
    return nets, feature_norm_all_clients


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    '''

    Args:
        dataset: Dataset to use. Options: mnist, cifar10, fmnist, svhn, generated, femnist, a9a, rcv1, covtype.
        partition: Tha partition way. Options: homo, noniid-labeldir, noniid-#label1 (or 2, 3, ..., which means the fixed number of labels each party owns), real, iid-diff-quantity
        n_parties: Number of parties.
        init_seed: The initial seed.
        datadir: The path of the dataset.
        logdir: The path to store the logs.
        beta: The concentration parameter of the Dirichlet distribution for heterogeneous partition.

    Returns:
        net_dataidx_map: a dictionary. Its keys are party ID, and the value of each key is a list containing index of data assigned to this party.
    '''
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map


if __name__ == '__main__':

    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)

    device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))

    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(str(args))
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # --- 划分数据 ----
    logger.info("Partitioning data")
    # 给2个client划分data
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    logger.info("type of X_train: %s", str(type(X_train)))  # type of X_train: <class 'numpy.ndarray'>
    logger.info("len of X_train: %s", str(len(X_train)))    # 50000
    logger.info("len of y_train: %s", str(len(y_train)))    # 50000
    logger.info("len of X_test: %s", str(len(X_test)))      # 10000
    logger.info("len of y_test: %s", str(len(y_test)))      # 10000

    n_classes = len(np.unique(y_train))

    # 这个作为public dataset
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      args.test_batch_size)

    logger.info("len train_dl_global: %s", str(len(train_dl_global)))    # 782 = 50000/64
    logger.info("len test_dl_global: %s", str(len(test_dl_global)))    # 313 = 10000/32
    logger.info("len train_ds_global: %s", str(len(train_ds_global)))   # 50000
    logger.info("len test_ds_global: %s", str(len(test_ds_global)))     # 10000

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []

    # --- 数据加噪声 ----
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, args.test_batch_size,
                                                                                              dataidxs, noise_level,
                                                                                              party_id,
                                                                                              args.n_parties - 1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, args.test_batch_size,
                                                                                              dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=args.test_batch_size, shuffle=False)

    # --- 根据联邦学习算法分支：FedAvg/FedProx/SCAFFOLD/MOON/All_in ----
    if args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy_class(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)


    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global,
                                    device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy_class(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,
                                     test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy_class(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map,
                                                                test_dl=test_dl_global, device=device)
            total_n = sum(n_list)
            # print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n


            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                # print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    # print(updated_model[key].type())
                    # print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy_class(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            # logger.info("Global model para: %s" % (str(global_para)))
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=test_dl_global,
                                 global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy_class(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy_class(global_model, test_dl_global, get_confusion_matrix=True,
                                                     moon_model=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl=test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer,
                                      device=device)

        logger.info("All in test acc: %f" % testacc)

    elif args.alg == 'classifier_calibration':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        # logger.info("Global model para: %s" % (str(global_para)))

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            # 筛选本轮clients
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            # 将global model的参数值赋给每个client
            global_para = global_model.state_dict()
            # print(global_para)
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            nets, feature_norm_all_clients = \
                local_train_net_classifier_calibration(nets, selected, global_model,
                                                       args, net_dataidx_map, test_dl=test_dl_global, device=device)
            global_model.to('cpu')

