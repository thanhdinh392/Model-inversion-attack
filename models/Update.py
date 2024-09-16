import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
class DatasetSplit(Dataset):
    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = list(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        image, label = self.dataset[self.index[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, index=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, index), batch_size=self.args.local_bs, shuffle=True)
        self.lr = args.lr

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, betas=(0.5, 0.999), amsgrad=True)
        start_time = time.time()  # start time local
        total_loss = 0
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # adding noise
                if self.args.noise:
                    with torch.no_grad():  # Đảm bảo không tính gradients khi thêm nhiễu
                        for param in net.parameters():
                            noise = torch.normal(mean=0, std=self.args.sigma, size=param.size()).to(self.args.device)  # Tạo nhiễu Gaussian
                            param.add_(noise)
                total_loss += loss.item()
        end_time = time.time()
        during_time = end_time - start_time
        print(f"Epoch {iter + 1}/{self.args.local_ep}, Loss: {(total_loss / len(self.ldr_train)):.4f}, Time: {during_time:.2f} seconds")
        local_loss = total_loss / (len(self.ldr_train) * self.args.local_ep)
        return net.state_dict(), local_loss


