import torch
from torch import nn

"""
Notes:

nn.Module
---------

torch.nn can be used to implement a layer like a fully connected layer, a convolutional layer, a pooling layer, and
even an entire neural network by creating an instance of a torch.nn.Module object

multiple nn.Module objects can be strung together to create a larger nn.Module object

nn.Module can also be used to represent any arbitrary function `f` in PyTorch

There are two functions to override in when using nn.Module. 
    1. __init__ : Here is where you will define various parameters of a layer such as filters, kernel size for a conv
                  layer, and dropout probability for the dropout layer
    2. forward : Here is where the output is defined. 
    

nn.Sequential
-------------

When initiating this class, a list of nn.Module objects in a particular sequence can be passed, which will then in turn
return a nn.Module object. When operating with the returned nn.Module object, any input passed through will be operated
on sequentially by the list of objects used to create our returned object.
"""

# def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class ResidualBlock(nn.Module):
    # Define all layers and components in our model
    def __init__(self, in_channels, out_channels, stride=1):
        # super(ResidualBlock, self).__init__()
        super().__init__()  # fixed for python3 update

        # FIXME
        print(ResidualBlock.__mro__)

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    # Define forward pass behavior
    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


# Define a layer that multiplies the input by 5 on forward pass
class MyLayer(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def forward(self, x):
        return x * self.param


if __name__ == '__main__':

    myLayerObject = MyLayer(5)
    output = myLayerObject(torch.Tensor([9, 1, 5])) # calls forward inexplicitely
    # print(output)

    # Using nn.Sequential
    combinedNetwork = nn.Sequential(MyLayer(5), MyLayer(10))
    output = combinedNetwork([2, 3])
    # print(output)


