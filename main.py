import torch
from torch import nn

"""
Notes:

nn.Module
---------

torch.nn can be used to implement a layer like a fully connected layer, a convolutional layer, a pooling layer, and
even an entire neural network by creating an instance of a torch.nn.Module object

multiple nn.Module objects can be strung together to create a larger nn.Module object

nn.Module can also be used to represent any arbitrary function `f` in PyTorch, and are looked at as being layers.

There are two functions to override in when using nn.Module. 
    1. __init__ : Here is where you will define various parameters of a layer such as filters, kernel size for a conv
                  layer, and dropout probability for the dropout layer
    2. forward : Here is where the output is defined. 
    

nn.Sequential
-------------

When initiating this class, a list of nn.Module objects in a particular sequence can be passed, which will then in turn
return a nn.Module object. When operating with the returned nn.Module object, any input passed through will be operated
on sequentially by the list of objects used to create our returned object.

Hooks
-----
Forward hooks are called on the forward() function of an Autograd.Function object, and backwards hooks
are used for the backward() function. 

- Tensor signature of backwards hook: hook(grad) -> Tensor or None
- No signature for forward hook on tensor
"""


class myNet(nn.Module):
    def __init__(self):  # Define convolutional and fully connected layer, as well as the activation and flattening func
        super().__init__()
        self.conv = nn.Conv2d(3, 10, 2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = lambda x: x.view(-1)
        self.fc1 = nn.Linear(160, 5)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x.register_hook(lambda grad: torch.clamp(grad, min=0))  # No gradient shall be backpropagated

        # print whether there is any negative grad
        x.register_hook(lambda grad: print("Gradients less than zero:", bool((grad < 0).any())))
        return self.fc1(self.flatten(x))


if __name__ == '__main__':

    net = myNet()

    for name, param in net.named_parameters():
        # if the param is from a linear and is a bias
        if "fc" in name and "bias" in name:
            param.register_hook(lambda grad: torch.zeros(grad.shape))

    out = net(torch.randn(1, 3, 8, 8))

    (1 - out).mean().backward()

    print("The biases are", net.fc1.bias.grad)  # bias grads are zero



