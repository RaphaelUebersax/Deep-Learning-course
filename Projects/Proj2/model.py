"""
    File containing our own deep learning framework. We employ a hierarchical structure,
    where all the classes (ReLU, Tanh, MSELoss, Linear and Sequential) derive from Module.
"""

from torch import empty as empty
from torch import set_grad_enabled
import math

set_grad_enabled(False)



################################################################################
# Parent class.
################################################################################
class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []



################################################################################
# ReLU activation function.
################################################################################
class ReLU(Module):

    def forward(self, *input):
        
        self.s = input[0].clone()

        x = input[0]
        x[x<0] = 0

        return x


    def backward(self, *gradwrtoutput):

        # dLoss / dx
        i = gradwrtoutput[0]

        s = self.s.clone()

        s[s>=0] = 1
        s[s<0] = 0
        
        # dLoss / ds
        return s * i
    


################################################################################
# Tanh activation function.
################################################################################
class Tanh(Module):

    def forward(self, *input):

        self.s = input[0].clone()

        return 2/(1+(self.s*-2).exp())-1


    def backward(self, *gradwrtoutput):

        i = gradwrtoutput[0]

        return 4 * (self.s.exp() + self.s.mul(-1).exp()).pow(-2) * i



################################################################################
# MSE loss implementation.
# Detects when we are using minibatches or single datapoints depending on 
# the dimension of the input tensor.
################################################################################
class MSELoss(Module):

    def forward(self, *input):

        e1 = input[0]
        e2 = input[1]

        if e1.dim() == 2:
            return (e1 - e2).pow(2).sum(dim=1).mean()/e1.shape[1]   # dividing by the number of samples on the minibatch
                                                                    # ensures that the loss is not too big when we have lots
                                                                    # of them.
        else:
            return (e1 - e2).pow(2).sum()/e1.shape[0]


    def backward(self, *gradwrtoutput):

        e1 = gradwrtoutput[0]
        e2 = gradwrtoutput[1]

        # dLoss / dx (see first element on the big brace slide 9 lecture 3.6)
        # e1 is the prediction, e2 is the target
        if e1.dim() == 2:
            return 2*(e1 - e2)/e1.shape[1]
        else:
            return 2*(e1 - e2)/e1.shape[0]



################################################################################
# Parameter class. 
# It is used to store the parameters of the linear layer. 
################################################################################
class Parameter():
    def __init__(self, input):
        self.p = input
        self.gradient = empty((input.shape))
        


################################################################################
# Linear layer. 
# It's optimized to work both with minibatches or single datapoints.
# The weight and bias can be initialized following Pytorch by default style 
# or with Xavier initialization (xavier = true).
################################################################################
class Linear(Module): 

    def __init__(self, input_size, output_size, xavier=False, gain=5/3):
        if not xavier:
            k = 1/input_size
        else:
            k = gain*gain*6/(input_size+output_size)
            
        self.w = Parameter(empty((output_size, input_size)).uniform_(-math.sqrt(k),math.sqrt(k)))
        self.b = Parameter(empty((output_size)).uniform_(-math.sqrt(k),math.sqrt(k)))


    def forward(self, *input):

        x = input[0]

        self.x_1 = x

        if x.dim() == 2:
            return x.mm(self.w.p.t()) + self.b.p

        else:
            return self.w.p.mv(x) + self.b.p


    def backward(self, *gradwrtoutput):
        
        # dLoss/ds
        x = gradwrtoutput[0]

        if x.dim() == 2:
            self.w.gradient = x.t().mm(self.x_1)/x.shape[0]
            self.b.gradient = x.mean(dim=0)
            
            # dLoss/dx
            return x.mm(self.w.p)

        else:
            # we use view to reshape our 1d tensors to 2d so we can multiply them.
            x_extra_dim = x.view(x.shape[0],1) # []
            self.x_1 = self.x_1.view(1,self.x_1.shape[0])
            self.w.gradient = x_extra_dim.mm(self.x_1)
            self.b.gradient = x

            # dLoss/dx
            return self.w.p.t().mv(x)
            

    def param(self): 
        
        return [self.w, self.b]



################################################################################
# Sequential.
# Defines an ordered list of linear layers and activation functions. 
################################################################################
class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers
        self.parameters = []
        
        for i in self.layers:
            self.parameters.extend(i.param())


    def forward(self, *input):

        previous_i = input[0]

        for i in self.layers:
            previous_i = i.forward(previous_i)

        return previous_i


    def backward(self, *gradwrtoutput): 

        next_i = gradwrtoutput[0]

        for i in reversed(self.layers):
            next_i = i.backward(next_i)

        return next_i

    
    def param(self): 
        
        return self.parameters
