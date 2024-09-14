import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# import options lib
from utils.options import args_parser
# sample user
from utils.sampling import mnist_noniid, mnist_iid
# model
from models.Nets import CNNMnist
# Local
from models.Update import LocalUpdate
# other lib
import copy
import time
import random
import numpy as np
import os
from datetime import datetime
# Fed
from models.Fed import FedWeightAvg
# Test
from models.test import test_img
# Log
from utils.log import create_log_dirs

if __name__ == "__main__":
    # create args
    args = args_parser()

    # random seed
    random.seed(123)  # built-in random in python
    np.random.seed(123)  # random function in numpy
    torch.manual_seed(123)  # torch.rand() --> random in CPU
    torch.cuda.manual_seed_all(123)  # torch.rand() --> random in 1 GPU (specified)
    torch.cuda.manual_seed(123)  # torch.rand() --> random in multi GPU

    # create log files
    if args.iid:
        if args.noise:
            base_dir = f"./noise_gauss/{args.dataset}/log_main_iid"
        else:
            base_dir = f"./{args.dataset}/log_main_iid"
        log_dir, log_tensorboard, log_client = create_log_dirs(base_dir)
    else:
        if args.noise:
            base_dir = f"./noise_gauss/{args.dataset}/log_main_non_iid"
        else:
            base_dir = f"./{args.dataset}/log_main_non_iid"
        log_dir, log_tensorboard, log_client = create_log_dirs(base_dir)

    # create tensorboard
    writer = SummaryWriter(log_tensorboard)

    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    if args.dataset == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample user
        if args.iid:
            print("Using mnist iid")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("Using mnist non-iid")
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == "fashion-mnist":
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
        if args.iid:
            print("Using fashion-mnist iid")
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            print("Using fashion-mnist non-iid")
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == "cifar":
        pass
    else:
        exit('Error: unrecognized dataset')

    # Load state_dict non_iid
    # with open("global_model_fashion-mnist_epoch200_non_iid.pkl", "rb") as f:
    #     data_loaded = torch.load(f)
    # build model
    if args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
        # net_glob.load_state_dict(data_loaded)
    else:
        pass
    # print model
    print(net_glob)
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    # create list clients
    clients = [LocalUpdate(args=args, dataset=dataset_train, index=dict_users[idx])
               for idx in range(args.num_users)]
    if args.iid:
        li = [1, 10, 50, 70, 100]  # list specific epoch iid
    else:
        li = [1, 10, 50, 100, 120, 150]  # list specific epoch non-iid
    # training epochs
    for iter in range(args.epochs):
        w_locals, loss_locals, len_local_data = [], [], []
        t_start = time.time()
        for idx in range(args.num_users):
            print(f"Client {idx}")
            if args.noise:
                print("Adding noise Gaussian...")
            w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            len_local_data.append(len(dict_users[idx]))
            if (iter + 1) in li:
                epoch_path = log_client + f"/epoch_{iter + 1}"
                os.makedirs(epoch_path, exist_ok=True)
                with open(os.path.join(epoch_path, f"local_model_{idx}.pkl"), "wb") as f:
                    torch.save(w, f)
        # update global weights
        w_glob = FedWeightAvg(w_locals, len_local_data)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        with torch.no_grad():
            net_glob.eval()
            acc_t, loss_t = test_img(net_glob, dataset_test, args)
            writer.add_scalar("acc_test", acc_t, iter)
            writer.add_scalar("loss_test", loss_t, iter)
            t_end = time.time()
            print("Round {:3d}, Testing accuracy: {:.4f}, Loss: {:.4f},Time: {:.2f}s".format(iter, acc_t, loss_t,
                                                                                             (t_end - t_start)))
        # saving global_model
        id_global_model = iter + 1
        if id_global_model in li:
            with open(os.path.join(log_dir, f"global_model_{id_global_model}.pkl"), "wb") as f:
                torch.save(net_glob.state_dict(), f)
    writer.close()
















