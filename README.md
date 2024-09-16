# SSAmplified Model Inversion Attacks on Federated Learning Frameworks
## Usage
### Hardware Requirements
Any Nvidia GPU with 8GB or larger memory is ok
### Required Runtime Libraries

* [Anaconda](https://www.anaconda.com/download/)
* [Pytorch](https://pytorch.org/) -- `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
* [zhangzp9970/torchplus](https://github.com/zhangzp9970/torchplus) -- `conda install torchplus -c zhangzp9970`

The code is compatable with the latest version of all the software.
### Datasets

* MNIST -- `torchvision.datasets.MNIST(root[, train, transform, ...])`
* Fashion-MNIST -- `torchvision.datasets.FashionMNIST(root[, train, transform, ...])`
### File Description
* main_Fl.py -- training target net (classifier) on Fed
* main_central.py -- training target net (classifier) on central
* attack_fed.py -- attacking target net(on Fed)
* attack_central.py -- attacking target central
* requirement.txt -- useful lib
* script_run_file.txt -- how to run files with many options

