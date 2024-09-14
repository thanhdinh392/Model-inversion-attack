import torch
# torch.nn lib
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# torch.optim lib
import torch.optim as optim
# torch.utils lib
from torch.utils.data import DataLoader
# torchvision lib
from torchvision.datasets import *
from torchvision.transforms.transforms import *
from torchvision.transforms.functional import *
from torchvision import datasets, transforms

# torchplus lib
from torchplus.utils import save_image2
from torchplus.nn import PixelLoss

# other lib
from tqdm import tqdm
from piq import SSIMLoss
import os
import copy
import time
import warnings

# Tắt các cảnh báo FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# args
from utils.options import args_parser
# Nets
from models.Nets import Inversion, PowerAmplification, CNNMnist

if __name__ == "__main__":

    args = args_parser()
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")

    if args.dataset == "mnist":
        trans_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == "fashion-mnist":
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                             transform=trans_fashion_mnist)
    elif args.dataset == "cifar":
        pass
    else:
        exit('Error: unrecognized dataset')

    print(len(dataset_train))
    print(len(dataset_test))

    train_dl = DataLoader(
        dataset=dataset_train,
        batch_size=args.bs,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )

    test_dl = DataLoader(
        dataset=dataset_test,
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    target_classifier = CNNMnist(args).train(False).to(args.device)

    target_amplification = (
        PowerAmplification(args.num_classes, 1 / 11).train(False).to(args.device)
    )

    # client
    if args.type_fed == "client":
        print(f"Attack fed ds {args.dataset} client...")
        if args.iid:
            if args.noise:
                print("Attacking Client with noise Gauss iid")
                client_path = f"./noise_gauss/{args.dataset}/log_main_iid/log_client"
            else:
                client_path = f"./{args.dataset}/log_main_iid/log_client"
            li = [1, 10, 50, 70, 100]

        else:
            if args.noise:
                print("Attacking Client with noise Gauss non-iid")
                client_path = f"./noise_gauss/{args.dataset}/log_main_non_iid/log_client"
            else:
                client_path = f"./{args.dataset}/log_main_non_iid/log_client"
            li = [1, 10, 50, 100, 120, 150]
        # create evaluation dict for each epoch with each clients
        ssim_dict = {f"epoch_{i}": [] for i in li}
        mse_dict = {f"epoch_{i}": [] for i in li}
        pixel_dict = {f"epoch_{i}": [] for i in li}
        # create evaluation dict for each epoch with avg clients
        avg_ssim_t = {f"epoch_{i}": [] for i in li}
        avg_mse_t = {f"epoch_{i}": [] for i in li}
        avg_pixel_t = {f"epoch_{i}": [] for i in li}

        files = os.listdir(client_path)
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for epoch_file in sorted_files:
            # epoch_<file>
            print(f"Client {epoch_file}...")
            avg_ssim = 0
            avg_mse = 0
            avg_pixel = 0
            for pkl_file in sorted(os.listdir(os.path.join(client_path, epoch_file))):
                print(pkl_file)
                target_pkl = os.path.join(client_path + "/" + epoch_file, pkl_file)
                target_classifier.load_state_dict(torch.load(open(target_pkl, "rb"), map_location=args.device))
                target_classifier.requires_grad_(False)

                # model attack
                myinversion = copy.deepcopy(Inversion(args).train(True)).to(args.device)
                # optimizer
                optimizer = optim.Adam(
                    myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
                )
                # train model attack
                for epoch_id in tqdm(range(1, args.epochs + 1), desc="Total Epoch"):
                    total_loss = 0
                    t_start = time.time()
                    for i, (im, label) in enumerate(tqdm(test_dl, desc=f"epoch {epoch_id}")):
                        im = im.to(args.device)
                        label = label.to(args.device)
                        optimizer.zero_grad()
                        out = target_classifier.forward(im)
                        after_softmax = F.softmax(out, dim=-1)
                        after_softmax = target_amplification.forward(after_softmax)
                        rim = myinversion.forward(after_softmax)
                        loss = F.mse_loss(rim, im)
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                    t_end = time.time()
                    # time for each epoch
                    during_time = t_end - t_start
                    print(f"Epoch {epoch_id}/{args.epochs}, Loss: {total_loss / len(test_dl)}, Time: {during_time}")

                with torch.no_grad():
                    myinversion.eval()
                    r = 0
                    ssimloss = 0
                    mseloss = 0
                    pixelloss = 0
                    id_img = 0
                    for i, (im, label) in enumerate(tqdm(train_dl, desc=f"priv")):
                        r += 1  # calculate len(train_dl)
                        im = im.to(args.device)
                        label = label.to(args.device)
                        out = target_classifier.forward(im)
                        after_softmax = F.softmax(out, dim=-1)
                        after_softmax = target_amplification.forward(after_softmax)
                        rim = myinversion.forward(after_softmax)
                        # saving images
                        if args.iid:
                            if args.noise:
                                images_path = f"./fed_client_noise/{args.dataset}/log_main_iid/save_images/{epoch_file}/{pkl_file.split('.')[0]}"
                            else:
                                images_path = f"./fed_client/{args.dataset}/log_main_iid/save_images/{epoch_file}/{pkl_file.split('.')[0]}"
                        else:
                            if args.noise:
                                images_path = f"./fed_client_noise/{args.dataset}/log_main_non_iid/save_images/{epoch_file}/{pkl_file.split('.')[0]}"
                            else:
                                images_path = f"./fed_client/{args.dataset}/log_main_non_iid/save_images/{epoch_file}/{pkl_file.split('.')[0]}"
                        os.makedirs(images_path, exist_ok=True)
                        if (i + 1) % 25 == 0:
                            save_image2(im.detach(), f"{images_path}/input/image_{id_img}.png")
                            save_image2(rim.detach(), f"{images_path}/output/image_{id_img}.png")
                            id_img += 1
                        # evaluation
                        ssim = SSIMLoss()(rim, im)
                        mse = F.mse_loss(rim, im)
                        pixel = PixelLoss(13)(rim, im)
                        ssimloss += ssim
                        mseloss += mse
                        pixelloss += pixel

                    ssimlossavg = round((ssimloss / r).item(), 4)
                    mselossavg = round((mseloss / r).item(), 4)
                    pixellossavg = round((pixelloss / r).item(), 4)
                    # save evaluation files
                    if args.iid:  # iid
                        if args.noise:
                            eval_path = f"./fed_client_noise/{args.dataset}/log_main_iid/save_eval"
                        else:
                            eval_path = f"./fed_client/{args.dataset}/log_main_iid/save_eval"

                    else:  # non-iid
                        if args.noise:
                            eval_path = f"./fed_client_noise/{args.dataset}/log_main_non_iid/save_eval"
                        else:
                            eval_path = f"./fed_client/{args.dataset}/log_main_non_iid/save_eval"
                    os.makedirs(eval_path, exist_ok=True)

                ssim_dict[f"{epoch_file}"].append(ssimlossavg)
                mse_dict[f"{epoch_file}"].append(mselossavg)
                pixel_dict[f"{epoch_file}"].append(pixellossavg)

                print(
                    f"Epoch {epoch_file.split('_')[1]}, Client {pkl_file}, SSIM: {ssimlossavg}, MSE: {mselossavg}, Pixelloss: {pixellossavg}")
            print(
                f"Epoch {epoch_file.split('_')[1]} detail\nssim_dict {ssim_dict}\nmse_dict {mse_dict}\npixel_dict {pixel_dict}")

            avg_ssim = round(sum(ssim_dict[f"{epoch_file}"]) / len(ssim_dict[f"{epoch_file}"]), 4)
            avg_mse = round(sum(mse_dict[f"{epoch_file}"]) / len(mse_dict[f"{epoch_file}"]), 4)
            avg_pixel = round(sum(pixel_dict[f"{epoch_file}"]) / len(pixel_dict[f"{epoch_file}"]), 4)
            # ssim, mse, pixel loss avg for each epochs
            avg_ssim_t[f"{epoch_file}"].append(avg_ssim)
            avg_mse_t[f"{epoch_file}"].append(avg_mse)
            avg_pixel_t[f"{epoch_file}"].append(avg_pixel)
            print(
                f"Epoch {epoch_file.split('_')[1]} avg\navg_ssim_t {avg_ssim_t}\navg_mse_t {avg_mse_t}\navg_pixel_t{avg_pixel_t}")
            # break
        # saving evaluation files
        with open(os.path.join(eval_path, "evaluation_file.pkl"), "wb") as f:
            torch.save((ssim_dict, mse_dict, pixel_dict, avg_ssim_t, avg_mse_t, avg_pixel_t), f)

    # global
    else:
        print(f"Attack fed ds {args.dataset} global...")
        eval_dict = {"ssim": [], "mse": [], "pixel_loss": []}
        if args.iid:
            if args.noise:
                print("Attacking Global with noise Gauss iid")
                global_path = f"./noise_gauss/{args.dataset}/log_main_iid/log_global_model"
            else:
                global_path = f"./{args.dataset}/log_main_iid/log_global_model"
        else:
            if args.noise:
                print("Attacking Client with noise Gauss non-iid")
                global_path = f"./noise_gauss/{args.dataset}/log_main_non_iid/log_global_model"
            else:
                global_path = f"./{args.dataset}/log_main_non_iid/log_global_model"
        files = os.listdir(global_path)
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for pkl_file in sorted_files:
            print(pkl_file)
            target_pkl = os.path.join(global_path, pkl_file)
            target_classifier.load_state_dict(torch.load(open(target_pkl, "rb"), map_location=args.device))
            target_classifier.requires_grad_(False)

            # model attack
            myinversion = copy.deepcopy(Inversion(args).train(True)).to(args.device)
            # optimizer
            optimizer = optim.Adam(
                myinversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True
            )
            # train model attack
            for epoch_id in tqdm(range(1, args.epochs + 1), desc="Total Epoch"):
                total_loss = 0
                t_start = time.time()
                for i, (im, label) in enumerate(tqdm(test_dl, desc=f"epoch {epoch_id}")):
                    im = im.to(args.device)
                    label = label.to(args.device)
                    optimizer.zero_grad()
                    out = target_classifier.forward(im)
                    after_softmax = F.softmax(out, dim=-1)
                    after_softmax = target_amplification.forward(after_softmax)
                    rim = myinversion.forward(after_softmax)
                    loss = F.mse_loss(rim, im)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                t_end = time.time()
                # time for each epoch
                during_time = t_end - t_start
                print(f"Epoch {epoch_id}/{args.epochs}, Loss: {total_loss / len(test_dl)}, Time: {during_time}")
            with torch.no_grad():
                myinversion.eval()
                r = 0
                ssimloss = 0
                mseloss = 0
                pixelloss = 0
                id_img = 0
                for i, (im, label) in enumerate(tqdm(train_dl, desc=f"priv")):
                    r += 1  # calculate len(train_dl)
                    im = im.to(args.device)
                    label = label.to(args.device)
                    out = target_classifier.forward(im)
                    after_softmax = F.softmax(out, dim=-1)
                    after_softmax = target_amplification.forward(after_softmax)
                    rim = myinversion.forward(after_softmax)
                    # saving images
                    if args.iid:
                        if args.noise:
                            images_path = f"./fed_global_noise/{args.dataset}/log_main_iid/save_images/{pkl_file.split('.')[0]}"

                        else:
                            images_path = f"./fed_global/{args.dataset}/log_main_iid/save_images/{pkl_file.split('.')[0]}"
                    else:
                        if args.noise:
                            images_path = f"./fed_global_noise/{args.dataset}/log_main_non_iid/save_images/{pkl_file.split('.')[0]}"

                        else:
                            images_path = f"./fed_global/{args.dataset}/log_main_non_iid/save_images/{pkl_file.split('.')[0]}"
                    os.makedirs(images_path, exist_ok=True)
                    if (i + 1) % 25 == 0:
                        save_image2(im.detach(), f"{images_path}/input/image_{id_img}.png")
                        save_image2(rim.detach(), f"{images_path}/output/image_{id_img}.png")
                        id_img += 1
                    # evaluation
                    ssim = SSIMLoss()(rim, im)
                    mse = F.mse_loss(rim, im)
                    pixel = PixelLoss(13)(rim, im)
                    ssimloss += ssim
                    mseloss += mse
                    pixelloss += pixel

                ssimlossavg = round((ssimloss / r).item(), 4)
                mselossavg = round((mseloss / r).item(), 4)
                pixellossavg = round((pixelloss / r).item(), 4)
                # save evaluation files
                if args.iid:
                    if args.noise:
                        eval_path = f"./fed_global_noise/{args.dataset}/log_main_iid/save_eval"

                    else:
                        eval_path = f"./fed_global/{args.dataset}/log_main_iid/save_eval"
                else:
                    if args.noise:
                        eval_path = f"./fed_global_noise/{args.dataset}/log_main_non_iid/save_eval"

                    else:
                        eval_path = f"./fed_global/{args.dataset}/log_main_non_iid/save_eval"
                os.makedirs(eval_path, exist_ok=True)
                print(
                    f"Fed global {pkl_file.split('.')[0]}, ds {args.dataset}, SSIM: {ssimlossavg}, MSE: {mselossavg}, Pixelloss: {pixellossavg}")
                eval_dict["ssim"].append(ssimlossavg)
                eval_dict["mse"].append(mselossavg)
                eval_dict["pixel_loss"].append(pixellossavg)
        # saving evaluation files
        with open(os.path.join(eval_path, "evaluation_file.pkl"), "wb") as f:
            torch.save(eval_dict, f)





