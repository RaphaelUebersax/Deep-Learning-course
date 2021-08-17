"""
    We run a loop performing a grid search over all possible configurations of models and
    auxiliary losses and weight sharing enabled/disabled. Batch normalization is enabled
    always.

    For each configuration we print the results specified below, and at the end we save two files
    (one for executions with auxiliary losses, other for the ones without them) on 
    stored_results/test.pt and stored_results/test_no_aux.pt.

    To understand the structure of the tensors, see readme.txt.
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


criterion = nn.CrossEntropyLoss()

models = ["Net", "LeNet5", "LeNet5_FullyConv", "ResNet"] 
weight_sharing_list = [False, True]
auxiliary_list = [False, True]
batch_norm_list = [True]

seeds_list = [1,2,3,4,5,6,7,8,9,10]
nb_epochs = 30
mini_batch_size = 100
etas = [3e-2, 5e-3, 5e-2, 5e-3]



results_seed_auxilary = torch.zeros((8,len(seeds_list)))
results_seed_no_auxilary = torch.zeros((4,len(seeds_list)))

results_weight_sharing_auxilary = torch.zeros((len(weight_sharing_list),len(batch_norm_list),len(models),2,8))
results_weight_sharing_no_auxilary = torch.zeros((len(weight_sharing_list),len(batch_norm_list),len(models),2,4))

results_batch_norm_auxilary = torch.zeros((len(batch_norm_list),len(models),2,8))
results_batch_norm_no_auxilary = torch.zeros((len(batch_norm_list),len(models),2,4))

results_model_auxilary = torch.zeros((len(models),2,8))
results_model_no_auxilary = torch.zeros((len(models),2,4))



###############################################################################

###############################################################################


for auxiliary in auxiliary_list:
    for w, weight_sharing in enumerate(weight_sharing_list):
        for b, batch_norm in enumerate(batch_norm_list):
            for m, model_s in enumerate(models):
                for i , seed in enumerate(seeds_list):

                    #shuffle the dataset 
                    torch.manual_seed(seed)
                    train_input,train_target,train_classes = shuffle_data(init_train_input, init_train_target, init_train_classes)

                    model = eval(model_s + "(" + str(batch_norm) + "," + str(weight_sharing) + ")")


                    if(auxiliary == True):
                        train_auxilary_loss(model, train_input, train_target, train_classes, mini_batch_size, etas[m], criterion, nb_epochs)
                        nb_errors_img_1_training, nb_errors_img_2_training = test_classes_fn(model, train_input, train_classes, mini_batch_size)
                        nb_errors_classes_img_1, nb_errors_classes_img_2 = test_classes_fn(model, test_input, test_classes, mini_batch_size)
                        nb_errors_target, nb_err_last_target = test_target_fn(model, test_input, test_target, mini_batch_size)
                        nb_errors_training, nb_err_last_training = test_target_fn(model, train_input, train_target, mini_batch_size)

                        # training set error rate on classes
                        results_seed_auxilary[0,i] = 100 * nb_errors_img_1_training / train_input.size(0)
                        results_seed_auxilary[1,i] = 100 * nb_errors_img_2_training / train_input.size(0)

                        # testing set error rate on classes
                        results_seed_auxilary[2,i] = 100 * nb_errors_classes_img_1 / test_input.size(0)
                        results_seed_auxilary[3,i] = 100 * nb_errors_classes_img_2 / test_input.size(0)

                        # error rate on main loss, both in train and test sets
                        results_seed_auxilary[4,i] = 100 * nb_errors_training / train_input.size(0)
                        results_seed_auxilary[5,i] = 100 * nb_errors_target / test_input.size(0)

                        # error rate on the layers performing digit comparison, both in train and test sets
                        results_seed_auxilary[6,i] = 100 * nb_err_last_training / train_input.size(0)
                        results_seed_auxilary[7,i] = 100 * nb_err_last_target / test_input.size(0)
                        


                        
                    else:
                        train(model, train_input, train_target, train_classes, mini_batch_size, etas[m], criterion, nb_epochs)
                        
                        nb_errors_target, nb_err_last_target = test_target_fn(model, test_input, test_target, mini_batch_size)
                        nb_errors_training, nb_err_last_training = test_target_fn(model, train_input, train_target, mini_batch_size)
                    

                        # error rate on main loss, both in train and test sets
                        results_seed_no_auxilary[0,i] = 100 * nb_errors_training / train_input.size(0)
                        results_seed_no_auxilary[1,i] = 100 * nb_errors_target / test_input.size(0)

                        # error rate on the layers performing digit comparison, both in train and test sets
                        results_seed_no_auxilary[2,i] = 100 * nb_err_last_training / train_input.size(0)
                        results_seed_no_auxilary[3,i] = 100 * nb_err_last_target / test_input.size(0)

                print("")
                print("")
                print("**********************************************************")
                print("- Model: " + str(model_s) + "\t - lr: " + str(etas[m]) + "\t - BatchNorm: " + str(batch_norm))
                print("- Weight sharing: " + str(weight_sharing) + "\t - Auxiliary loss: " + str(auxiliary))
                print("**********************************************************")
                print("")

                if(auxiliary == True):

                    std_auxilary = torch.std(results_seed_auxilary,dim = 1)
                    mean_auxilary = torch.mean(results_seed_auxilary,dim = 1)

                    results_model_auxilary[m][0] = mean_auxilary
                    results_model_auxilary[m][1] = std_auxilary
                    
                    print("The mean error on classes img 1 at training (%): ",round( mean_auxilary[0].item(),2),"with a std of: ",round(std_auxilary[0].item(),2))
                    print("The mean error on classes img 2 at training (%): ", round(mean_auxilary[1].item(),2),"with a std of: ", round(std_auxilary[1].item(),2))
                    print("The mean error on classes img 1 at testing (%): ", round(mean_auxilary[2].item(),2),"with a std of: ", round(std_auxilary[2].item(),2))
                    print("The mean error on classes img 2 at testing (%): ", round(mean_auxilary[3].item(),2),"with a std of: ", round(std_auxilary[3].item(),2))
                    print("The mean error on training target (%): ", round(mean_auxilary[4].item(),2),"with a std of: ", round(std_auxilary[4].item(),2))
                    print("The mean error on testing target (%): ", round(mean_auxilary[5].item(),2),"with a std of: ", round(std_auxilary[5].item(),2))
                    print("The last layer's mean Benchmark error at training (%): ", round(mean_auxilary[6].item(),2),"with a std of: ", round(std_auxilary[6].item(),2))
                    print("The last layer's mean Benchmark error at testing (%): ", round(mean_auxilary[7].item(),2),"with a std of: ", round(std_auxilary[7].item(),2))


                else:

                    std_no_auxilary = torch.std(results_seed_no_auxilary,dim = 1)
                    mean_no_auxilary = torch.mean(results_seed_no_auxilary,dim = 1)

                    results_model_no_auxilary[m][0] = mean_no_auxilary
                    results_model_no_auxilary[m][1] = std_no_auxilary

                    print("The mean error on training target (%): ", round(mean_no_auxilary[0].item(),2),"with a std of: ",round(std_no_auxilary[0].item(),2))
                    print("The mean error on testing target (%): ", round(mean_no_auxilary[1].item(),2),"with a std of: ", round(std_no_auxilary[1].item(),2))
                    print("The last layer's mean Benchmark error at training (%): ", round(mean_no_auxilary[2].item(),2),"with a std of: ", round(std_no_auxilary[2].item(),2))
                    print("The last layer's mean Benchmark error at testing (%): ", round(mean_no_auxilary[3].item(),2),"with a std of: ", round(std_no_auxilary[3].item(),2))

            if (auxiliary == True):
                results_batch_norm_auxilary[b] = results_model_auxilary
            else:
                results_batch_norm_no_auxilary[b] = results_model_no_auxilary
        
        if(auxiliary == True):
            results_weight_sharing_auxilary[w] = results_batch_norm_auxilary
        else:
            results_weight_sharing_no_auxilary[w] = results_batch_norm_no_auxilary


print(results_weight_sharing_auxilary.shape)
print(results_weight_sharing_auxilary)

print(results_weight_sharing_no_auxilary.shape)
print(results_weight_sharing_no_auxilary)

torch.save(results_weight_sharing_auxilary,'stored_results/test.pt')
torch.save(results_weight_sharing_no_auxilary,'stored_results/test_no_aux.pt')


