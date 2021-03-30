import torch
from torchvision import datasets

def getCIFAR10(batch_size, TF):
    data_root = 'data/cifar10'
    val_loader = None # We don't need a validation split for CIFAR10
        
    ds_train = datasets.CIFAR10(root = data_root, train = True, download = True, transform = TF)    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = True, num_workers = 1)
    
    ds_test = datasets.CIFAR10(root = data_root, train = False, download = True, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
    
    return train_loader, test_loader, val_loader

def getCIFAR100(batch_size, TF):
    data_root = 'data/cifar100'
    val_loader = None # We don't need a validation split for CIFAR100
        
    ds_train = datasets.CIFAR100(root = data_root, train = True, download = True, transform = TF)    
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = True, num_workers = 1)

    ds_test = datasets.CIFAR100(root = data_root, train = False, download = True, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
        
    return train_loader, test_loader, val_loader

def getSVHN(batch_size, TF):
    data_root = 'data/svhn'
    ds_test = datasets.SVHN(root = data_root, split = 'test', download = True, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)

    return test_loader

def getImagenet(batch_size, TF):
    data_root = 'data/Imagenet'
    ds_test = datasets.ImageFolder(data_root, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
    
    return test_loader

def getLSUN(batch_size, TF):
    data_root = 'data/LSUN'
    ds_test = datasets.ImageFolder(data_root, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
    
    return test_loader

def getiSUN(batch_size, TF):
    data_root = 'data/iSUN'
    ds_test = datasets.ImageFolder(data_root, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
    
    return test_loader

def getGaussian(batch_size, TF):
    data_root = 'data/Gaussian'
    ds_test = datasets.ImageFolder(data_root, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
    
    return test_loader
    
def getUniform(batch_size, TF):
    data_root = 'data/Uniform'
    ds_test = datasets.ImageFolder(data_root, transform = TF)
    test_loader = torch.utils.data.DataLoader(ds_test, batch_size = batch_size, shuffle = False, num_workers = 1)
    
    return test_loader