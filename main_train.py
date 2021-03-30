from importlib.machinery import SourceFileLoader
from data import data_loader
from torchvision import transforms
import argparse
import torch
from torchsummary import summary
import models
import numpy as np
import KDE
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import math

def load_datasets(dataset_name, batch_size):
    train_loader, test_loader, val_loader = None, None, None
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])    
    
    if dataset_name == 'cifar10':
        train_loader, test_loader, val_loader = data_loader.getCIFAR10(batch_size = batch_size, TF = test_transforms)
        
    if dataset_name == 'cifar100':
        train_loader, test_loader, val_loader = data_loader.getCIFAR100(batch_size = batch_size, TF = test_transforms)
        
    if dataset_name == 'svhn':
        test_loader = data_loader.getSVHN(batch_size = batch_size, TF = test_transforms)
    
    if dataset_name == 'Imagenet':
        test_loader = data_loader.getImagenet(batch_size = batch_size, TF = test_transforms)
        
    if dataset_name == 'LSUN':
        test_loader = data_loader.getLSUN(batch_size = batch_size, TF = test_transforms)
        
    if dataset_name == 'iSUN':
        test_loader = data_loader.getiSUN(batch_size = batch_size, TF = test_transforms)
        
    if dataset_name == 'Gaussian':
        test_loader = data_loader.getGaussian(batch_size = batch_size, TF = test_transforms)
        
    if dataset_name == 'Uniform':
        test_loader = data_loader.getUniform(batch_size = batch_size, TF = test_transforms)        
        
    return train_loader, test_loader, val_loader

def main(exp_config):

    # =====================
    # Load network
    # =====================
    model = models.ResNet34(num_c = exp_config.num_classes)
    summary(model.cuda(), input_size=(3, 32, 32)) # display the layers of the network
        
    model.cuda() # copy the model into gpu
    
    # =========================
    # Load source dataset and pre-trained model
    # =========================
    source_train_loader, source_test_loader, _ = load_datasets(exp_config.data_identifier_source, exp_config.batch_size)
    model.load_state_dict(torch.load(exp_config.pre_trained_net))
    model.eval()
    
    # =========================
    # KDE-based OOD detection
    # =========================
    
    # Open a .txt file to save the OOD detection results
    path_to_saved_results = 'results/' + exp_config.experiment_name + '/' + exp_config.method_name + '/results_' + str(exp_config.number_of_samples_for_KDE) + '.txt'
    f = open(path_to_saved_results, "w")
    
    # Compute number of layers in the network
    num_layers = KDE.compute_num_layers(exp_config, model)
       
    # Compute features for each channel for the test set of in-distribution dataset
    # get_features function returns MxN tensor where M is the number of samples
    # and N is the number of channels
    print('Calculating features for the test set of in-distribution dataset')
    feature_in_test = KDE.get_features(exp_config, model, num_layers, source_test_loader)
    
    # Compute features for each channel for the training set of in-distribution dataset
    print('Calculating features for the training set of in-distribution dataset')
    feature_in_train = KDE.get_features(exp_config, model, num_layers, source_train_loader, is_train = True)
    
    # Compute features for each channel for the adversarially perturbed version of training set of in-distribution dataset
    print('Calculating features for the adversarial images')
    feature_in_train_perturbed = KDE.get_features(exp_config, model, num_layers, source_train_loader, perturb = True, is_train = True)
    
    # Calculate features for each OOD dataset
    print('Calculating features for each OOD dataset')
    feature_ood = {}
    for target in exp_config.data_identifier_target:
        _, ood_loader, _ = load_datasets(target, exp_config.batch_size)
        feature_ood[target] = KDE.get_features(exp_config, model, num_layers, ood_loader)
    
    if exp_config.std_type == 'kNN': # Load pre-computed sigma values for each channel using kNN as proposed in the paper - COMPUTATIONALLY INEFFICIENT
        std = torch.Tensor(np.load('results/std_%s.npy'%(exp_config.data_identifier_source))).cuda()
    elif exp_config.std_type == 'interquartile': # Compute signa values for each channel using interquartiles - COMPUTATIONALLY EFFICIENT AND LEADS TO SIMILAR RESULTS IN THE PAPER
        sorted_feature_in_train, _ = torch.sort(feature_in_train, axis = 0)
        emp_std = torch.std(feature_in_test, axis = 0)
        Q1, Q3 = torch.median(sorted_feature_in_train[0:sorted_feature_in_train.shape[0] // 2], axis = 0).values, torch.median(sorted_feature_in_train[(sorted_feature_in_train.shape[0] // 2):], axis = 0).values
        IQR = Q3 - Q1
        std = 0.9 * torch.min(torch.cat([torch.unsqueeze(emp_std, 0), torch.unsqueeze(IQR, 0)/1.34], axis = 0), axis = 0).values * (feature_in_train.shape[0]**(-1/5))

    # Calculate confidence scores using KDE for test set of the in-distribution dataset
    print('Calculating confidence scores using KDE for the test set of the in-distribution dataset')
    constant = 1 / (std * torch.sqrt(torch.Tensor([2 * math.pi]).cuda()))
    scores_in_test = 0
    for i in range(feature_in_train.shape[0]):
        zero_x = feature_in_test - feature_in_train[i]
        scores_in_test += constant * torch.exp(-0.5 * (torch.pow(torch.div(zero_x, std), 2)))
    scores_in_test /= feature_in_train.shape[0]
    scores_in_test = scores_in_test.detach().cpu().numpy()        
     
    # Calculate confidence scores using KDE for training set of the in-distribution dataset
    print('Calculating confidence scores using KDE for the training set of the in-distribution dataset')
    scores_in_train = 0
    for i in range(feature_in_train.shape[0]):
        zero_x = feature_in_train - feature_in_train[i]
        scores_in_train += constant * torch.exp(-0.5 * (torch.pow(torch.div(zero_x, std), 2)))
    scores_in_train /= feature_in_train.shape[0]
    scores_in_train = scores_in_train.detach().cpu().numpy()
    
    # Calculate confidence scores using KDE for the adversarially perturbed version of training set of the in-distribution dataset
    print('Calculating confidence scores using KDE for the adversarial images')
    scores_in_train_perturbed = 0
    for i in range(feature_in_train.shape[0]):
        zero_x = feature_in_train_perturbed - feature_in_train[i]
        scores_in_train_perturbed += constant * torch.exp(-0.5 * (torch.pow(torch.div(zero_x, std), 2)))
    scores_in_train_perturbed /= feature_in_train.shape[0]
    scores_in_train_perturbed = scores_in_train_perturbed.detach().cpu().numpy()

    # Calculate confidence scores using KDE for OOD datasets 
    print('Calculating confidence scores using KDE for OOD datasets')
    scores_ood = {}
    for target in exp_config.data_identifier_target:
        scores_ood[target] = 0
        for i in range(feature_in_train.shape[0]):
            zero_x = feature_ood[target] - feature_in_train[i]
            scores_ood[target] += constant * torch.exp(-0.5 * (torch.pow(torch.div(zero_x, std), 2)))
        scores_ood[target] /= feature_in_train.shape[0]
        scores_ood[target] = scores_ood[target].detach().cpu().numpy()
      
    # Calculate OOD detection accuracy
    print('Calculating OOD detection accuracy')
    
    # Find channels that best distinguishes scores of in-distribution test set from the adversarial images
    y_pred = np.concatenate((scores_in_test, scores_in_train_perturbed), axis = 0)
    label = np.concatenate((np.ones(scores_in_test.shape[0]), np.zeros(scores_in_train_perturbed.shape[0])), axis = 0)
    fpr_all = []
    for i in range(scores_in_test.shape[1]):
        fpr_at_95_tpr, detection_error, auroc, aupr_in = calculate_ood_detection_performance_metrics(label, y_pred[:, i], str(i), display = False)
        fpr_all.append(fpr_at_95_tpr)
    
    # Create training set to train logistic regression    
    X_train = np.concatenate((np.sort(scores_in_train[:, np.argsort(fpr_all)[:50]], axis = 1), np.sort(scores_in_train_perturbed[:, np.argsort(fpr_all)[:50]], axis = 1)), axis = 0)
    Y_train = np.concatenate((np.zeros(scores_in_train.shape[0]), np.ones(scores_in_train_perturbed.shape[0])), axis = 0)
    
    # Train logistic regression
    lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
    
    # Evaluate logistic regression on each OOD dataset and compute OOD detection accuracy
    f.write('Target \t\t FPRat95TPR \t DetErr \t AUROC \t\t AUPR_IN \n')
    print('Target \t\t TPRat95TPR \t DetErr \t AUROC \t\t AUPR_IN \n')
    for target in exp_config.data_identifier_target:
        X_test = np.concatenate((np.sort(scores_in_test[:, np.argsort(fpr_all)[:50]], axis = 1), np.sort(scores_ood[target][:, np.argsort(fpr_all)[:50]], axis = 1)), axis = 0)
        Y_test = np.concatenate((np.zeros(scores_in_test.shape[0]), np.ones(scores_ood[target].shape[0])), axis = 0)
        
        y_pred = lr.predict_proba(X_test)[:, 1]
                
        fpr_at_95_tpr, detection_error, auroc, aupr_in = calculate_ood_detection_performance_metrics(Y_test, y_pred, target, display = True)
        
        f.write(('%8s \t %.5f \t %.5f \t %.5f \t %.5f \n\n')%(target, fpr_at_95_tpr, detection_error, auroc, aupr_in))
        
    print('Results are saved to ' + path_to_saved_results)
        
    f.close()

def calculate_ood_detection_performance_metrics(labels, predictions, name, display = True):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    precision_in, recall_in, thresholds = metrics.precision_recall_curve(labels, predictions)        
    
    fpr_at_95_tpr = 0
    for j in range(tpr.size):
        if tpr[j] >= 0.95:
            fpr_at_95_tpr = fpr[j]
            break
    
    detection_error = 0.5 * (1 - 0.95) + 0.5 * fpr_at_95_tpr
    auroc = metrics.roc_auc_score(labels, predictions)
    aupr_in = metrics.auc(recall_in, precision_in)
    
    if display:
        print(('%8s \t %.5f \t %.5f \t %.5f \t %.5f \t')%(name, fpr_at_95_tpr, detection_error, auroc, aupr_in))
    
    return fpr_at_95_tpr, detection_error, auroc, aupr_in

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    config_file = args.EXP_PATH
    config_module = config_file.split('/')[-1].rstrip('.py')
        
    exp_config = SourceFileLoader(config_module, config_file).load_module() # exp_config stores configurations in the given config file under experiments folder.

    main(exp_config=exp_config)

