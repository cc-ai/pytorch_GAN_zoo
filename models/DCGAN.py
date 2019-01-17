import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math

from .base_GAN import BaseGAN
from .utils.config import BaseConfig
from .networks.DCGAN_nets import GNet, DNet

class DCGAN(BaseGAN):
    r"""
    Implementation of DCGAN
    """
    def __init__(self,
                 latentVectorDim = 64,
                 dimG = 64,
                 dimD = 64,
                 depth = 3,
                **kwargs):
        r"""
        Args:

        Specific Arguments:
            - latentVectorDim (int): dimension of the input latent vector
            - dimG (int): reference depth of a layer in the generator
            - dimD (int): reference depth of a layer in the discriminator
            - depth (int): number of convolution layer in the model
            - **kwargs: arguments of the BaseGAN class

        """
        if not 'config' in vars(self):
            self.config = BaseConfig()

        self.config.dimG = dimG
        self.config.dimD = dimD
        self.config.depth = depth

        BaseGAN.__init__(self, latentVectorDim, **kwargs)


    def getNetG(self):

        gnet = GNet(self.config.latentVectorDim,
                    self.config.dimOutput,
                    self.config.dimG,
                    depthModel = self.config.depth,
                    generationActivation = self.lossCriterion.generationActivation)
        return gnet

    def getNetD(self):

        dnet =  DNet(self.config.dimOutput,
                     self.config.dimD,
                     self.lossCriterion.sizeDecisionLayer + self.config.categoryVectorDim,
                     depthModel = self.config.depth)
        return dnet

    def getOptimizerD(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()),
                          betas = [0.1, 0.999], lr = self.config.learningRate)

    def getOptimizerG(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                          betas = [0.1, 0.999], lr = self.config.learningRate)

    def getSize(self):
        return 2**(self.config.depth + 3)