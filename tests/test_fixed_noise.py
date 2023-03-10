import torch
import gpytorch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.likelihoods.noise_models import HomoskedasticNoise, HeteroskedasticNoise



class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



train_x = torch.randn(55, 2)
train_y = torch.randn(55, 1)
noises = torch.ones(55) * 0.01

#likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=False)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp_model = ExactGP(train_x, train_y, likelihood)

homo_noise = HeteroskedasticNoise(noise_model= gp_model, noise_indices= [1,4])

pred = likelihood(gp_model(train_x))



'''
import botorch
from botorch.models import HeteroskedasticSingleTaskGP


train_X = torch.rand(20, 2)
train_Y = torch.sin(train_X).sum(dim=1, keepdim=True)
se = torch.norm(train_X, dim=1, keepdim=True)
train_Yvar = 0.1 + se * torch.rand_like(train_Y)
model = HeteroskedasticSingleTaskGP(train_X, train_Y, train_Yvar)

aa = 1
'''