# import pytorch libraries
from torch import nn

# define encoder class
class Encoder(nn.Module):

    # define class constructor
    def __init__(self, hidden_size, bottleneck):

        # call super class constructor
        super(Encoder, self).__init__()

        # init encoder architecture
        self.linearLayers = self.init_layers(hidden_size)
        self.reluLayer = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        if bottleneck == 'linear':

            self.bottleneck = nn.Identity()

        elif bottleneck == 'lrelu':

            self.bottleneck = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        elif bottleneck == 'tanh':

            self.bottleneck = nn.Tanh()

    # init encoder layers
    def init_layers(self, layer_dimensions):

        # init layers
        layers = []

        # iterate over layer dimensions
        for i in range(0, len(layer_dimensions)-1):

            # create linear layer
            linearLayer = self.LinearLayer(layer_dimensions[i], layer_dimensions[i + 1])

            # collect linear layer
            layers.append(linearLayer)

            # register linear layer
            self.add_module('linear_' + str(i), linearLayer)

        # return layers
        return layers

    # init leaky ReLU layer
    def LinearLayer(self, input_size, hidden_size):

        # init linear layer
        linear = nn.Linear(input_size, hidden_size, bias=True)

        # init linear layer parameters
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

        # return linear layer
        return linear

    # define forward pass
    def forward(self, x):

        # iterate over distinct layers
        for i in range(0, len(self.linearLayers)):

            # case: non-bottleneck layer
            if i < len(self.linearLayers)-1:

                # run forward pass through layer
                x = self.reluLayer(self.linearLayers[i](x))

            # case: bottleneck layer
            else:

                # run forward pass through layer
                x = self.bottleneck(self.linearLayers[i](x))

        # return result
        return x
