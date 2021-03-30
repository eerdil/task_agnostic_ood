# Task-agnostic Out-of-Distribution Detection Using Kernel Density Estimation

PyTorch implementation of our paper [Task-agnostic Out-of-Distribution Detection Using Kernel Density Estimation](https://arxiv.org/abs/2006.10712) by Ertunc Erdil ([email](mailto:ertunc.erdil@vision.ee.ethz.ch)), Krishna Chaitanya, Neerav Karani, and Ender Konukoglu.

## Citation

If you find this code helpful in your research please cite the following paper:

```
@article{erdil2021taskagnostic,
  title={Task-agnostic out-of-distribution detection using kernel density estimation},
  author={Erdil, Ertunc and Chaitanya, Krishna and Karani, Neerav and Konukoglu, Ender},
  journal={arXiv preprint arXiv:2006.10712},
  year={2021}
}
```

## Requirements

conda create --name <env_name> # Create virtual environment - optional
conda install -c conda-forge scikit-learn
conda install numpy
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install torchsummary

## Downloading datasets and models

Download the [datasets](https://www.dropbox.com/s/xb6l2wae2lcf74t/data.tar.gz?dl=0) and copy to ./data folder

Download the [pre-trained models](https://www.dropbox.com/s/6athtgl3e9t3gku/pre_trained.tar.gz?dl=0) and copy to ./pre_trained folder

## How to run

CIFAR10 - python main_train.py config/cifar10/cifar10_kde_cfg_01.py

CIFAR100 - python main_train.py config/cifar100/cifar100_kde_cfg_01.py
