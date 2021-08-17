"""
This file contains the 4 models on which we are going to test techniques as
auxiliary losses, weight sharing, batch normalization or dropout. 
"""

import torch
from torch import nn
from torch.nn import functional as F


###############################################################################
# This model is based on the LeNet5 model for image classification with the 
# dimensions adapted from the MNIST dataset with 14x14 pixels. The model is a 
# siamese network that is conceived to classify both input images individually
# and then concatenate the result to compare digit on image 1 to the digit on
# image 2.

# It is composed of two convolutional layers that can be interpreted as
# "features extraction" followed by 3 linear layers that are the image 
# "classification" part of the network. The final 2 layers are then  to perform
# classification on the digit comparison task.

# The model takes 3 input parameters that decides if weight sharing is
# implemented, if batch normalization is performed after each convolutional
# layer or if dropout is used or not.
###############################################################################
class LeNet5(nn.Module):
    def __init__(self, batch_norm, weight_sharing, dropout = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(256,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

        if dropout: self.dp = nn.Dropout(0.25)

        if not weight_sharing:
            self.conv21 = nn.Conv2d(1,32, kernel_size = 3)
            self.bn21 = nn.BatchNorm2d(32)
            self.conv22 = nn.Conv2d(32,64, kernel_size = 3)
            self.bn22 = nn.BatchNorm2d(64)
            
            self.fc21 = nn.Linear(256,120)
            self.fc22 = nn.Linear(120,84)
            self.fc23 = nn.Linear(84,10)
        
        self.fc4 = nn.Linear(20,72)
        self.fc5 = nn.Linear(72,2)

        self.weight_sharing  = weight_sharing
        self.batch_norm = batch_norm
        self.dropout = dropout



    def forward(self, x):
        res = []

        if self.weight_sharing:
            for i in [0,1]:
                y = x[:,i:i+1,:,:]
                y = self.conv1(y)
                if self.batch_norm: y = self.bn1(y)
                y = F.relu(F.max_pool2d(y, kernel_size=2))
                y = self.conv2(y)
                if self.batch_norm: y = self.bn2(y)
                y = F.relu(F.max_pool2d(y, kernel_size=2))
                y = F.relu(self.fc1(y.view(-1, 256)))
                if self.dropout: y = self.dp(y)
                y = F.relu(self.fc2(y))
                if self.dropout: y = self.dp(y)
                y = self.fc3(y)
                res.append(y)

        
        else:
            y = x[:,0:1,:,:]
            y = self.conv1(y)
            if self.batch_norm: y = self.bn1(y)
            y = F.relu(F.max_pool2d(y, kernel_size=2))
            y = self.conv2(y)
            if self.batch_norm: y = self.bn2(y)
            y = F.relu(F.max_pool2d(y, kernel_size=2))
            y = F.relu(self.fc1(y.view(-1, 256)))
            if self.dropout: y = self.dp(y)
            y = F.relu(self.fc2(y))
            if self.dropout: y = self.dp(y)
            y = self.fc3(y)
            res.append(y)

            z = x[:,1:2,:,:]
            z = self.conv21(z)
            if self.batch_norm: z = self.bn21(z)
            z = F.relu(F.max_pool2d(z, kernel_size=2))
            z = self.conv22(z)
            if self.batch_norm: z = self.bn22(z)
            z = F.relu(F.max_pool2d(z, kernel_size=2))
            z = F.relu(self.fc21(z.view(-1, 256)))
            if self.dropout: z = self.dp(z)
            z = F.relu(self.fc22(z))
            if self.dropout: z = self.dp(z)
            z = self.fc23(z)
            res.append(z)
            
            
        out = torch.cat((res[0],res[1]),dim=1)
        output = F.relu(self.fc4(out))
        if self.dropout: output = self.dp(output)
        output = self.fc5(output)
        
        return res[0], res[1], output





###############################################################################
# This model is a simplified (smaller) version of the LeNet5 model described
# before for the same purpose (classification with the dimensions adapted from
# the MNIST dataset with 14x14 pixels). The goal is to have a computationally 
# less expensive model and be able to assess its influence on the perfomance.

# It is composed of two convolutional layers that can be interpreted as
# "features extraction" followed by 2 linear layers that are the image 
# "classification" part of the network. The final 2 layers are then  to perform
# classification on the digit comparison task.

# The model takes 2 input parameters that decides if weight sharing is
# implemented and if batch normalization is performed after each convolutional
# layer.
###############################################################################

class Net(nn.Module):
    def __init__(self, batch_norm, weight_sharing):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 16, kernel_size = 3)
        self.bn11 = nn.BatchNorm2d(16)
        self.conv12 = nn.Conv2d(16,32, kernel_size = 3)
        self.bn12 = nn.BatchNorm2d(32)
        
        self.fc11 = nn.Linear(128, 64)
        self.fc12 = nn.Linear(64, 10)

        if not weight_sharing:
        
            self.conv21 = nn.Conv2d(1, 16, kernel_size=3)
            self.bn21 = nn.BatchNorm2d(16)
            self.conv22 = nn.Conv2d(16,32, kernel_size = 3)
            self.bn22 = nn.BatchNorm2d(32)
            
            self.fc21 = nn.Linear(128, 64)
            self.fc22 = nn.Linear(64, 10)
        
        # self.fc3 = nn.Linear(20,72)
        # self.fc4 = nn.Linear(72,2)
        self.conv6 = nn.Conv1d(20,72, kernel_size = 1)
        self.bn6 = nn.BatchNorm1d(72)
        self.conv7 = nn.Conv1d(72,2, kernel_size = 1)
        
        self.weight_sharing = weight_sharing
        self.batch_norm = batch_norm
        
    
    
    def forward(self, x):
        res = []

        if self.weight_sharing:

            for i in [0,1]:
                y = x[:,i:i+1,:,:]
                y = self.conv11(y)
                if self.batch_norm: y = self.bn11(y)
                y = F.relu(F.max_pool2d(y, kernel_size=2))
                y = self.conv12(y)
                if self.batch_norm: y = self.bn12(y)
                y = F.relu(F.max_pool2d(y, kernel_size=2))
                y = F.relu(self.fc11(y.view(-1, 128)))
                y = self.fc12(y)
                res.append(y)

        else:
        
            y = x[:,0:1,:,:]
            z = x[:,1:2,:,:]

            y = self.conv11(y)
            if self.batch_norm: y = self.bn11(y)
            y = F.relu(F.max_pool2d(y, kernel_size=2))
            y = self.conv12(y)
            if self.batch_norm: y = self.bn12(y)
            y = F.relu(F.max_pool2d(y, kernel_size=2))
            y = F.relu(self.fc11(y.view(-1, 128)))
            y = self.fc12(y)
            res.append(y)
            
            z = self.conv21(z)
            if self.batch_norm: z = self.bn21(z)
            z = F.relu(F.max_pool2d(z, kernel_size=2))
            z = self.conv22(z)
            if self.batch_norm: z = self.bn22(z)
            z = F.relu(F.max_pool2d(z, kernel_size=2))
            z = F.relu(self.fc21(z.view(-1, 128)))
            z = self.fc22(z)
            res.append(z)
        
        # output = F.relu(self.fc3(torch.cat((res[0],res[1]),dim=1))) # adding ReLU
        # output = self.fc4(output)
        out = torch.cat((res[0],res[1]),dim=1)
        out = out[:,:,None]
        output = F.relu(self.conv6(out))
        if self.batch_norm: output = self.bn6(output)
        output = self.conv7(output).squeeze()
        
        return res[0], res[1], output
    





###############################################################################
# This model is based on the ResNet model for image classification with the 
# dimensions adapted from the MNIST dataset with 14x14 pixels. The model is a 
# siamese network that is conceived to classify both input images individually
# and then concatenate the result to compare digit on image 1 to the digit on
# image 2.

# It is composed on a ResNetBlock that is repeated several time and that keeps
# the dimension of the image. 

# The model takes 3 input parameters that decides if weight sharing is
# implemented, if batch normalization is performed after each convolutional
# layer and if we want to skip the connection on a block.
###############################################################################
class ResNetBlock(nn.Module):
    def __init__(self, batch_norm, weight_sharing, skip_connections = True):
        super().__init__()

        self.conv1 = nn.Conv2d(32, 32, kernel_size = 3,
                               padding = (3 - 1) // 2)

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3,
                               padding = (3 - 1) // 2)

        self.bn2 = nn.BatchNorm2d(32)
        self.weight_sharing = weight_sharing
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections

    def forward(self, x):
        y = self.conv1(x)
        if self.batch_norm: y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_norm: y = self.bn2(y)
        if self.skip_connections: y = y + x
        y = F.relu(y)

        return y


class ResNet(nn.Module):
    def __init__(self, batch_norm, weight_sharing, nb_residual_blocks = 3):
        super().__init__()

        self.conv = nn.Conv2d(1, 32, kernel_size = 3, padding = (3 - 1) // 2)
        self.bn = nn.BatchNorm2d(32)

        self.resnet_blocks = nn.Sequential(
            *(ResNetBlock(batch_norm, weight_sharing)
              for _ in range(nb_residual_blocks))
        )
        self.fc = nn.Linear(32, 10)

        if not weight_sharing:
            self.conv_2 = nn.Conv2d(1, 32, kernel_size = 3, padding = (3 - 1) // 2)
            self.bn_2 = nn.BatchNorm2d(32)

            self.resnet_blocks_2 = nn.Sequential(
            *(ResNetBlock(batch_norm, weight_sharing)
              for _ in range(nb_residual_blocks))
            )
            self.fc_2 = nn.Linear(32, 10)

        self.fc2 = nn.Linear(20,72)
        self.fc3 = nn.Linear(72,2)
        
        self.weight_sharing = weight_sharing
        self.batch_norm = batch_norm



    def forward(self, x):
        res = []

        if self.weight_sharing:
            for i in [0,1]:
                y = x[:,i:i+1,:,:]
                y = self.conv(y)
                if self.batch_norm: y = self.bn(y)
                y = F.relu(y)
                y = self.resnet_blocks(y)
                y = F.avg_pool2d(y, 14).view(y.size(0), -1)
                y = self.fc(y)
                res.append(y)

        else:
            y = x[:,0:1,:,:]
            y = self.conv(y)
            if self.batch_norm: y = self.bn(y)
            y = F.relu(y)
            y = self.resnet_blocks(y)
            y = F.avg_pool2d(y, 14).view(y.size(0), -1)
            y = self.fc(y)
            res.append(y)

            z = x[:,1:2,:,:]
            z = self.conv_2(z)
            if self.batch_norm: z = self.bn_2(z)
            z = F.relu(z)
            z = self.resnet_blocks_2(z)
            z = F.avg_pool2d(z, 14).view(z.size(0), -1)
            z = self.fc_2(z)
            res.append(z)
        
        out = torch.cat((res[0],res[1]),dim=1)
        output = F.relu(self.fc2(out))
        output = self.fc3(output)
       
        return res[0], res[1], output




###############################################################################
# This model is based on the LeNet5 model descibed earlier and convolutionalizes
# the linear layers to have a fully convolutional network.

# The model takes 2 input parameters that decides if weight sharing is
# implemented and if batch normalization is performed after each convolutional
# layer.
###############################################################################

class LeNet5_FullyConv(nn.Module):
    def __init__(self, batch_norm, weight_sharing):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolution of the linearize
        self.conv3 = nn.Conv2d(64, 120, kernel_size = 2)
        self.bn3 = nn.BatchNorm2d(120)
        self.conv4 = nn.Conv2d(120, 84, kernel_size = 1)
        self.bn4 = nn.BatchNorm2d(84)
        self.conv5 = nn.Conv2d(84, 10, kernel_size = 1)
        self.bn5 = nn.BatchNorm2d(10)


        if not weight_sharing:
            self.conv21 = nn.Conv2d(1,32, kernel_size = 3)
            self.bn21 = nn.BatchNorm2d(32)
            self.conv22 = nn.Conv2d(32,64, kernel_size = 3)
            self.bn22 = nn.BatchNorm2d(64)
            
            self.conv23 = nn.Conv2d(64, 120, kernel_size = 2)
            self.bn23 = nn.BatchNorm2d(120)
            self.conv24 = nn.Conv2d(120, 84, kernel_size = 1)
            self.bn24 = nn.BatchNorm2d(84)
            self.conv25 = nn.Conv2d(84, 10, kernel_size = 1)
            self.bn25 = nn.BatchNorm2d(10)

        
        self.conv6 = nn.Conv1d(20,72, kernel_size = 1)
        self.bn6 = nn.BatchNorm1d(72)
        self.conv7 = nn.Conv1d(72,2, kernel_size = 1)
        
        self.weight_sharing = weight_sharing
        self.batch_norm = batch_norm
        
        
    def forward(self, x):
        res = []

        if self.weight_sharing:
        
            for i in [0,1]:
                y = x[:,i:i+1,:,:]
                y = self.conv1(y)
                if self.batch_norm: y = self.bn1(y)
                y = F.relu(F.max_pool2d(y, kernel_size=2))
                y = self.conv2(y)
                if self.batch_norm: y = self.bn2(y)
                y = F.relu(F.max_pool2d(y, kernel_size=2))
                y = F.relu(self.conv3(y))
                if self.batch_norm: y = self.bn3(y)
                y = F.relu(self.conv4(y))
                if self.batch_norm: y = self.bn4(y)
                y = F.relu(self.conv5(y))
                if self.batch_norm: y = self.bn5(y)
                y = y.squeeze()
                res.append(y)
        
        else:

            y = x[:,0:1,:,:]
            y = self.conv1(y)
            if self.batch_norm: y = self.bn1(y)
            y = F.relu(F.max_pool2d(y, kernel_size=2))
            y = self.conv2(y)
            if self.batch_norm: y = self.bn2(y)
            y = F.relu(F.max_pool2d(y, kernel_size=2))
            y = F.relu(self.conv3(y))
            if self.batch_norm: y = self.bn3(y)
            y = F.relu(self.conv4(y))
            if self.batch_norm: y = self.bn4(y)
            y = F.relu(self.conv5(y))
            if self.batch_norm: y = self.bn5(y)
            y = y.squeeze()
            res.append(y)

            z = x[:,1:2,:,:]
            z = self.conv21(z)
            if self.batch_norm: z = self.bn21(z)
            z = F.relu(F.max_pool2d(z, kernel_size=2))
            z = self.conv22(z)
            if self.batch_norm: z = self.bn22(z)
            z = F.relu(F.max_pool2d(z, kernel_size=2))
            z = F.relu(self.conv23(z))
            if self.batch_norm: z = self.bn23(z)
            z = F.relu(self.conv24(z))
            if self.batch_norm: z = self.bn24(z)
            z = F.relu(self.conv25(z))
            if self.batch_norm: z = self.bn25(z)
            z = z.squeeze()
            res.append(z)
            
            
        out = torch.cat((res[0],res[1]),dim=1)
        out = out[:,:,None]
        output = F.relu(self.conv6(out))
        if self.batch_norm: output = self.bn6(output)
        output = self.conv7(output).squeeze()
        
        
        return res[0], res[1], output


###############################################################################
# Same as LeNet5, but adding dropout. This model will be used to compare
# techniques beyond auxiliary loss and weight sharing. Specifically we will 
# discuss how batch normalization and dropout affect our results. 
###############################################################################
class LeNet5_Dropout(nn.Module):
    def __init__(self, batch_norm, weight_sharing):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, kernel_size = 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size = 3)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(256,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

        if not weight_sharing:
            self.conv21 = nn.Conv2d(1,32, kernel_size = 3)
            self.bn21 = nn.BatchNorm2d(32)
            self.conv22 = nn.Conv2d(32,64, kernel_size = 3)
            self.bn22 = nn.BatchNorm2d(64)
            
            self.fc21 = nn.Linear(256,120)
            self.fc22 = nn.Linear(120,84)
            self.fc23 = nn.Linear(84,10)
        
        self.fc4 = nn.Linear(20,72)
        self.fc5 = nn.Linear(72,2)

        self.weight_sharing  = weight_sharing
        self.batch_norm = batch_norm