# import pytorch libraries
import torch.nn as nn

# import project libraries
import NetworkHandler.Encoder as Encoder
import NetworkHandler.Decoder as Decoder


# define the baseline autoencoder class
class BaselineAutoencoder(nn.Module):

    # define class constructor
    def __init__(self, encoder_layers, encoder_bottleneck, decoder_layers):

        # call super class constructor
        super(BaselineAutoencoder, self).__init__()
        
        # init the encoder model
        self.encoder = Encoder.Encoder(hidden_size=encoder_layers, bottleneck=encoder_bottleneck)

        # init the decoder model
        self.decoder = Decoder.Decoder(hidden_size=decoder_layers)

    # define autoencoder model forward pass
    def forward(self, input, return_z=False):

        # run encoder forward pass
        z = self.encoder(input)

        # run decoder forward pass
        output = self.decoder(z)

        # return latent representation and output
        if return_z:
            return z, output
        else:
            return output
