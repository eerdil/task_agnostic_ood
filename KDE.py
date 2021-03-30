from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

def get_features(exp_config, model, num_output, loader, perturb = False, is_train = False):
    loss_fn = nn.CrossEntropyLoss()
    epsilon = exp_config.epsilon
        
    counter = 0
    counter_batch = 0
    
    for data, target in loader:
        data = data.cuda()

        if perturb:
            for p in model.parameters():
                p.requires_grad = True
            
            data.requires_grad = True
            
            target = target.cuda()
            
            model.zero_grad()
            loss_perturbation = loss_fn(model.forward(data), target.long())
            
            loss_perturbation.backward()
            data_grad = data.grad.data
            data = fgsm_attack(data, epsilon, data_grad)
        
        _, out_features = model.feature_list(data.cuda())
        counter = 0
        for i in range(num_output):
            shapes = out_features[i].shape
            temp = out_features[i].reshape(shapes[0], shapes[1], -1)
            if counter == 0:
                temp_features = torch.mean(temp.data, axis = 2)                
            else:
                temp_features = torch.cat((temp_features, torch.mean(temp.data, axis = 2)), axis = 1)
                
            counter += 1
            
        if counter_batch == 0:
            list_features = temp_features
        else:
            list_features = torch.cat((list_features, temp_features), axis = 0)
                    
        counter_batch += data.shape[0]
        
        if is_train and (counter_batch >= exp_config.number_of_samples_for_KDE):
            break
    
    return list_features

def compute_num_layers(exp_config, model):
    model.eval()
    if exp_config.data_identifier_source == 'cifar10' or exp_config.data_identifier_source == 'cifar100':
        temp_x = torch.rand(2, 3, 32, 32).cuda()
    elif exp_config.data_identifier_source == 'abide_caltech' or exp_config.data_identifier_source == 'abide_stanford':
        temp_x = torch.rand(2, 1, 256, 256).cuda()
        
    temp_x = Variable(temp_x)
    _, temp_list = model.feature_list(temp_x)
    
    num_output = len(temp_list)
        
    return num_output

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


