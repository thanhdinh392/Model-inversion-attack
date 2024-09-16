import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader

from utils.options import args_parser
# model
from models.Nets import CNNMnist
# other lib
import time
from tqdm import tqdm
import os

# Test
from models.test import test_img

if __name__ == "__main__":
    args = args_parser()
    #create log dir
    log_dir = "./central_model"
    os.makedirs(log_dir, exist_ok=True)
    # device
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    # dataset options
    if args.dataset == "mnist":
        # trans_mnist = transforms.Compose([transforms.ToTensor()])
        trans_mnist = Compose(
            [
                Grayscale(num_output_channels=1),
                Resize((32, 32)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
            ]
        )
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == "fashion-mnist":
        # trans_fashion_mnist = transforms.Compose([transforms.ToTensor()])
        trans_fashion_mnist = Compose(
            [
                Grayscale(num_output_channels=1),
                Resize((32, 32)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
            ]
        )
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
    elif args.dataset == "cifar":
        pass
    else:
        exit('Error: unrecognized dataset')
    # dataloader
    train_dl = DataLoader(
        dataset=dataset_train,
        batch_size=args.bs,
        shuffle=True,
    )

    test_dl = DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        shuffle=False,
    )
    # build model
    if args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        pass
    # print model
    print(net_glob)
    net_glob.train()
    # optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    for iter in tqdm(range(args.epochs), desc="Total epoch"):
        # total_loss = 0
        start_time = time.time()  # start time local
        for batch_id, (images, labels) in enumerate(tqdm(train_dl, desc=f"epoch {iter}")):
            images = images.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            outputs = net_glob(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # total_loss += loss.item()
        # print(f"Epoch {iter + 1}/{args.epochs}, Loss: {total_loss / len(train_dl)}, Time: {during_time:.2f} seconds")
        with torch.no_grad():
            net_glob.eval()
            acc_t, loss_t = test_img(net_glob, dataset_test, args)
            end_time = time.time()
            during_time = end_time - start_time
            print("Epoch {:3d}, Testing accuracy: {:.4f}, Loss: {:.4f},Time: {:.2f}s".format(iter, acc_t, loss_t, during_time))
    with open(os.path.join(log_dir, f"central_model_{args.dataset}_epoch{args.epochs}.pkl"), "wb") as f:
        torch.save(net_glob.state_dict(), f)


