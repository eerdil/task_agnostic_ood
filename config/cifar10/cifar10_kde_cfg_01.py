import os

task_name = 'classification'
experiment_name = 'cifar10_kde_cfg_01'
method_name = 'KDE'

epsilon = 1.0 # strength of adversarial perturbation

number_of_samples_for_KDE = 5000

data_identifier_source = 'cifar10'
data_identifier_target = ['svhn', 'Imagenet', 'LSUN', 'iSUN', 'Gaussian', 'Uniform']

batch_size = 128
num_classes = 10

net_type = 'resnet'

# Can take values of 'interquartile' or 'kNN'. kNN loads pre-computed std values using the approach described in the paper. 
# Using interquatiles leads to similar results and computationally efficient 
std_type = 'kNN' 

pre_trained_net = './pre_trained/' + net_type + '_' + data_identifier_source + '.pth'

if not os.path.exists('results/' + experiment_name):
    os.mkdir('results/' + experiment_name)

if not os.path.exists('results/' + experiment_name + '/' + method_name):
    os.mkdir('results/' + experiment_name + '/' + method_name)