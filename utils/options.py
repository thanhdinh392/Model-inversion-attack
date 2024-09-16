import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    """
    attack_fed.py 
    epochs: 100
    bs: 32
    """
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--bs', type=int, default=32, help="test batch size")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    # parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    # attack file
    parser.add_argument('--type_fed', type=str, default='client', help='Fed client or Fed global attack')
    parser.add_argument('--model', type=str, default='cnn', help='type model name')

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # add noise
    parser.add_argument('--noise', action='store_true', help='whether add noise or not')
    # sigma noise Gaussian
    """
    noise:
    => sigma
    min: 0.001
    max: 0.01
    """
    parser.add_argument('--sigma', type=float, help="sigma noise Gaussian")
    args = parser.parse_args()
    return args
