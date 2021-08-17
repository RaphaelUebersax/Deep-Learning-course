"""
    File to run a deep analysis over LeNet5. 
    We only make this analysis with auxiliary loss enabled, as it always gives the best results 
    by far. We try all possible combinations of batch normalization, weight sharing and dropout
    enabled/disabled. 

    As usual, we run each configuration for 10 different seeds and we take the mean.

    We log the intermediate results and we save a tensor with the final ones on 
    stored_results/lenet_deep_analysis_results.pt. 
    To understand the structure of the tensor, see readme.txt. 
"""

import torch
from torch import nn
from models import *
from train_test_functions import *

print("--------------- Loading data ---------------")

init_train_input, init_train_target, init_train_classes, test_input, test_target, \
    test_classes = load_data()


print("--------------- Data loaded ---------------")
print("")


eta = 5e-3
mini_batch_size = 100
criterion = nn.CrossEntropyLoss()
seeds = [1,2,3,4,5,6,7,8,9,10]
nb_epochs = 30

dropout_list = [False, True]
batch_norm_list = [False, True]
weight_sharing_list = [False, True]
model_s = "LeNet5"


results_seed = torch.zeros((8,len(seeds)))
results_dropout = torch.zeros((len(dropout_list),2,8))
results_batch_norm = torch.zeros((len(batch_norm_list),len(dropout_list),2,8))
results_weight_sharing = torch.zeros((len(weight_sharing_list),len(batch_norm_list),len(dropout_list),2,8))


for w, weight_sharing in enumerate(weight_sharing_list):
    for b, batch_norm in enumerate(batch_norm_list):
        for d, dropout in enumerate(dropout_list):
            for s, seed in enumerate(seeds):

                #shuffle the dataset 
                torch.manual_seed(seed)
                train_input,train_target,train_classes = shuffle_data(init_train_input, init_train_target, init_train_classes)

                model = eval(model_s + "(" + str(batch_norm) + "," + str(weight_sharing) + "," + str(dropout) + ")")

                train_auxilary_loss(model, train_input, train_target, train_classes, mini_batch_size, eta, criterion, nb_epochs)
                nb_errors_img_1_training, nb_errors_img_2_training = test_classes_fn(model, train_input, train_classes, mini_batch_size)
                nb_errors_classes_img_1, nb_errors_classes_img_2 = test_classes_fn(model, test_input, test_classes, mini_batch_size)
                nb_errors_target, nb_err_last_target = test_target_fn(model, test_input, test_target, mini_batch_size)
                nb_errors_training, nb_err_last_training = test_target_fn(model, train_input, train_target, mini_batch_size)

                # training set error rate on classes
                results_seed[0,s] = 100 * nb_errors_img_1_training / train_input.size(0)
                results_seed[1,s] = 100 * nb_errors_img_2_training / train_input.size(0)
                
                # testing set error rate on classes
                results_seed[2,s] = 100 * nb_errors_classes_img_1 / test_input.size(0)
                results_seed[3,s] = 100 * nb_errors_classes_img_2 / test_input.size(0)

                # error rate on main loss, both in train and test sets
                results_seed[4,s] = 100 * nb_errors_training / train_input.size(0)
                results_seed[5,s] = 100 * nb_errors_target / test_input.size(0)

                # error rate on the layers performing digit comparison, both in train and test sets
                results_seed[6,s] = 100 * nb_err_last_training / train_input.size(0)
                results_seed[7,s] = 100 * nb_err_last_target / test_input.size(0)

            print("")
            print("")
            print("**********************************************************")
            print("- Model: " + str(model_s) + "\t - lr: " + str(eta) + "\t - BatchNorm: " + str(batch_norm))
            print("- Weight sharing: " + str(weight_sharing) + "\t - Dropout: " + str(dropout))
            print("**********************************************************")
            print("")

            std = torch.std(results_seed,dim = 1)
            mean = torch.mean(results_seed,dim = 1)

            results_dropout[d][0] = mean
            results_dropout[d][1] = std

            print("The mean error on classes img 1 at training (%): ",round( mean[0].item(),2),"with a std of: ",round(std[0].item(),2))
            print("The mean error on classes img 2 at training (%): ", round(mean[1].item(),2),"with a std of: ", round(std[1].item(),2))
            print("The mean error on classes img 1 at testing (%): ", round(mean[2].item(),2),"with a std of: ", round(std[2].item(),2))
            print("The mean error on classes img 2 at testing (%): ", round(mean[3].item(),2),"with a std of: ", round(std[3].item(),2))
            print("The mean error on training target (%): ", round(mean[4].item(),2),"with a std of: ", round(std[4].item(),2))
            print("The mean error on testing target (%): ", round(mean[5].item(),2),"with a std of: ", round(std[5].item(),2))
            print("The last layer's mean Benchmark error at training (%): ", round(mean[6].item(),2),"with a std of: ", round(std[6].item(),2))
            print("The last layer's mean Benchmark error at testing (%): ", round(mean[7].item(),2),"with a std of: ", round(std[7].item(),2))
        
        results_batch_norm[b] = results_dropout
    
    results_weight_sharing[w] = results_batch_norm

print(results_weight_sharing)

torch.save(results_weight_sharing,'stored_results/lenet_deep_analysis_results.pt')