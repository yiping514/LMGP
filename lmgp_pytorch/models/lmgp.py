# Copyright © 2021 by Northwestern University.
# 
# LVGP-PyTorch is copyrighted by Northwestern University. It may be freely used 
# for educational and research purposes by  non-profit institutions and US government 
# agencies only. All other organizations may use LVGP-PyTorch for evaluation purposes 
# only, and any further uses will require prior written approval. This software may 
# not be sold or redistributed without prior written approval. Copies of the software 
# may be made by a user provided that copies are not sold or distributed, and provided 
# that copies are used under the same terms and conditions as agreed to in this 
# paragraph.
# 
# As research software, this code is provided on an "as is'' basis without warranty of 
# any kind, either expressed or implied. The downloading, or executing any part of this 
# software constitutes an implicit agreement to these terms. These terms and conditions 
# are subject to change at any time without prior notice.

from turtle import color
import torch
import math
import gpytorch
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior,GammaPrior
from gpytorch.distributions import MultivariateNormal

from lmgp_pytorch.visual.plot_latenth import plot_sep
from .gpregression import GPR
from .. import kernels
from ..priors import MollifiedUniformPrior
from typing import List

import numpy as np
from pandas import DataFrame
from category_encoders import BinaryEncoder

from lmgp_pytorch.preprocessing import setlevels
#from lmgp_pytorch.optim import fit_model_scipy, noise_tune2
from lmgp_pytorch.visual import plot_ls
import matplotlib.pyplot as plt
from math import prod

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
}


class LMGP(GPR):
    """The latent Map GP regression model (LMGP) which extends GPs to handle categorical inputs.

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no 
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as 
        integers 0,...,L-1 where L is the number of levels. For best performance, scale the 
        numerical variables to the unit hypercube.
    """
    def __init__(
        self,
        #transformation_of_A_parameters:str,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        qual_ind_lev = {},
        multiple_noise = False,
        lv_dim:int=2,
        quant_correlation_class:str='Rough_RBF',
        noise:float=1e-4,
        fix_noise:bool=False,
        lb_noise:float=1e-8,
        NN_layers:list = [],
        encoding_type = 'one-hot',
        uniform_encoding_columns = 2,
        lv_columns = [] 
    ) -> None:

        qual_index = list(qual_ind_lev.keys())
        all_index = set(range(train_x.shape[-1]))
        quant_index = list(all_index.difference(qual_index))
        num_levels_per_var = list(qual_ind_lev.values())
        #------------------- lm columns --------------------------
        lm_columns = list(set(qual_index).difference(lv_columns))
        if len(lm_columns) > 0:
            qual_kernel_columns = [*lv_columns, lm_columns]
        else:
            qual_kernel_columns = lv_columns

        #########################
        if len(qual_index) > 0:
            train_x = setlevels(train_x, qual_index=qual_index)
        #
        if multiple_noise:
            noise_indices = list(range(0,num_levels_per_var[0]))
        else:
            noise_indices = []


        if len(qual_index) == 1 and num_levels_per_var[0] < 2:
            temp = quant_index.copy()
            temp.append(qual_index[0])
            quant_index = temp.copy()
            qual_index = []
            lv_dim = 0
        elif len(qual_index) == 0:
            lv_dim = 0


        quant_correlation_class_name = quant_correlation_class

        if len(qual_index) == 0:
            lv_dim = 0


        if quant_correlation_class_name == 'Rough_RBF':
            quant_correlation_class = 'RBFKernel'
        if len(qual_index) > 0:
            ####################### Defined multiple kernels for seperate variables ###################
            qual_kernels = []
            for i in range(len(qual_kernel_columns)):
                qual_kernels.append(kernels.RBFKernel(
                    active_dims=torch.arange(lv_dim) + lv_dim * i) )
                qual_kernels[i].initialize(**{'lengthscale':1.0})
                qual_kernels[i].raw_lengthscale.requires_grad_(False)

        if len(quant_index) == 0:
            correlation_kernel = qual_kernels[0]
            for i in range(1, len(qual_kernels)):
                correlation_kernel *= qual_kernels[i]
        else:
            try:
                quant_correlation_class = getattr(kernels,quant_correlation_class)
            except:
                raise RuntimeError(
                    "%s not an allowed kernel" % quant_correlation_class
                )
            
            if quant_correlation_class_name == 'RBFKernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= torch.exp,inv_transform= torch.log)
                )
            elif quant_correlation_class_name == 'Rough_RBF':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns)*lv_dim+torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
                )
            if quant_correlation_class_name == 'RBFKernel':
                
                quant_kernel.register_prior(
                    'lengthscale_prior', MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
                
            elif quant_correlation_class_name == 'Rough_RBF':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )
            
            if len(qual_index) > 0:
                temp = qual_kernels[0]
                for i in range(1, len(qual_kernels)):
                    temp *= qual_kernels[i]
                correlation_kernel = temp*quant_kernel #+ qual_kernel + quant_kernel
            else:
                correlation_kernel = quant_kernel

        super(LMGP,self).__init__(
            train_x=train_x,train_y=train_y,noise_indices=noise_indices,
            correlation_kernel=correlation_kernel,
            noise=noise,fix_noise=fix_noise,lb_noise=lb_noise
        )

        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))
        self.register_buffer('qual_index',torch.tensor(qual_index))

        self.qual_kernel_columns = qual_kernel_columns
        # latent variable mapping
        # latent variable mapping
        self.num_levels_per_var = num_levels_per_var
        self.lv_dim = lv_dim
        self.uniform_encoding_columns = uniform_encoding_columns
        self.encoding_type = encoding_type
        self.perm =[]
        self.zeta = []
        self.perm_dict = []
        self.A_matrix = []
        if len(qual_kernel_columns) > 0:
            for i in range(len(qual_kernel_columns)):
                if type(qual_kernel_columns[i]) == int:
                    num = self.num_levels_per_var[qual_index.index(qual_kernel_columns[i])]
                    cat = [num]
                else:
                    cat = [self.num_levels_per_var[qual_index.index(k)] for k in qual_kernel_columns[i]]
                    num = sum(cat)

                zeta, perm, perm_dict = self.zeta_matrix(num_levels=cat, lv_dim = self.lv_dim)
                self.zeta.append(zeta)
                self.perm.append(perm)
                self.perm_dict.append(perm_dict)                
                model_temp = FFNN(self, input_size= num, num_classes=lv_dim, 
                    layers = NN_layers, name = str(qual_kernel_columns[i])).to(**tkwargs)
                self.A_matrix.append(model_temp.to(**tkwargs))


    def forward(self,x:torch.Tensor) -> MultivariateNormal:

        nd_flag = 0
        if x.dim() > 2:
            xsize = x.shape
            x = x.reshape(-1, x.shape[-1])
            nd_flag = 1

        if len(self.qual_kernel_columns) > 0:
            embeddings = []
            for i in range(len(self.qual_kernel_columns)):
                temp= self.transform_categorical(x=x[:,self.qual_kernel_columns[i]].clone().detach().type(torch.int64).to(tkwargs['device']), 
                    perm_dict = self.perm_dict[i], zeta = self.zeta[i])

                embeddings.append(self.A_matrix[i](temp.float().to(**tkwargs)))

            embeddings = torch.cat(embeddings, axis=-1)  
            if len(self.quant_index) > 0:
                x = torch.cat([embeddings,x[...,self.quant_index]],dim=-1)
            else:
                x = embeddings


        if nd_flag == 1:
            x = x.reshape(*xsize[:-1], -1)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x,covar_x)

    def predict(self, Xtest,return_std=True, include_noise = True):
        with torch.no_grad():
            return super().predict(Xtest, return_std = return_std, include_noise= include_noise)

    def score(self, Xtest, ytest, plot_MSE = True, title = None, seperate_levels = False):
        ypred = self.predict(Xtest, return_std=False)
        mse = ((ytest-ypred)**2).mean()
        print('################MSE######################')
        print(f'MSE = {mse:.3f}')
        print('#########################################')
        print('################Noise####################')
        noise = self.likelihood.noise_covar.noise.detach() * self.y_std**2
        print(f'The estimated noise parameter (varaince) is {noise}')
        print(f'The estimated noise std is {np.sqrt(noise)}')
        print('#########################################')

        if plot_MSE:
            plt.rcParams.update({'font.size': 19})
            _ = plt.figure(figsize=(8,6))
            _ = plt.plot(ytest.cpu().numpy(), ypred.cpu().numpy(), 'ro', label = 'Data')
            _ = plt.plot(ytest.cpu().numpy(), ytest.cpu().numpy(), 'b', label = 'MSE = ' + str(np.round(mse.detach().item(),3)))
            _ = plt.xlabel(r'Y_True')
            _ = plt.ylabel(r'Y_predict')
            _ = plt.legend()
            if title is not None:
                _ = plt.title(title)

        if seperate_levels and len(self.qual_index) > 0:
            for i in range(self.num_levels_per_var[0]):
                index = torch.where(Xtest[:,self.qual_index] == i)[0]
                _ = self.score(Xtest[index,...], ytest[index], 
                    plot_MSE=True, title = ' Only Source ' + str(i), seperate_levels=False)
        return mse, noise

    def visualize_latent(self, suptitle = None):
        if len(self.qual_kernel_columns) > 0:
            for i in range(len(self.qual_kernel_columns)):
                zeta = self.zeta[i]
                A = self.A_matrix[i]
                positions = A(zeta.float().to(**tkwargs))
                level = torch.max(self.perm[i], axis = 0)[0].tolist()
                perm = self.perm[i]
                plot_sep(positions = positions, levels = level, perm = perm, constraints_flag=True, )

        # if len(self.qual_index) > 0:
        #     plot_ls(self, constraints_flag=True, suptitle = suptitle)
        
        
    
    @classmethod
    def show(cls):
        plt.show()
        
    def get_params(self, name = None):
        params = {}
        print('###################Parameters###########################')
        for n, value in self.named_parameters():
             params[n] = value
        if name is None:
            print(params)
            return params
        else:
            if name == 'Mean':
                key = 'mean_module.constant'
            elif name == 'Sigma':
                key = 'covar_module.raw_outputscale'
            elif name == 'Noise':
                key = 'likelihood.noise_covar.raw_noise'
            elif name == 'Omega':
                for n in params.keys():
                    if 'raw_lengthscale' in n and params[n].numel() > 1:
                        key = n
            print(params[key])
            return params[key]
    

    def log_marginal_likelihood(self, X = None, y = None):
        self.eval()
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X == None:
            X = self.train_inputs[0]
            y = self.train_targets
        LL = self.likelihood(self(X)).log_prob(y)
        print('################## Log Likelihood ###########################')
        print(LL)
        print('#############################################################')
        return self.likelihood(self(X)).log_prob(y)

    def sample_y(self, size = 1, X = None, plot = False):
        if X == None:
            X = self.train_inputs[0]
        
        self.eval()
        out = self.likelihood(self(X))
        draws = out.sample(sample_shape = torch.Size([size]))
        index = np.argsort(out.loc.detach().numpy())
        if plot:
            _ = plt.figure(figsize=(12,6))
            _ = plt.scatter(list(range(len(X))), out.loc.detach().numpy()[index], color = 'red', s = 20, marker = 'o')
            _ = plt.scatter(np.repeat(np.arange(len(X)).reshape(1,-1), size, axis = 0), 
                draws.detach().numpy()[:,index], color = 'blue', s = 1, alpha = 0.5, marker = '.')
        return draws

    def get_latent_space(self):
        if len(self.qual_index) > 0:
            zeta = torch.tensor(self.zeta, dtype = torch.float64).to(**tkwargs)
            positions = self.nn_model(zeta)
            return positions.detach()
        else:
            print('No categorical Variable, No latent positions')
            return None



    def LMMAPPING(self, num_features:int, type = 'Linear',lv_dim = 2):

        if type == 'Linear':
            in_feature = num_features
            out_feature = lv_dim
            lm = torch.nn.Linear(in_feature, out_feature, bias = False)
            return lm

        else:
            raise ValueError('Only Linear type for now')    

    def zeta_matrix(self,
        num_levels:int,
        lv_dim:int,
        batch_shape=torch.Size()
    ) -> None:

        if any([i == 1 for i in num_levels]):
            raise ValueError('Categorical variable has only one level!')

        if lv_dim == 1:
            raise RuntimeWarning('1D latent variables are difficult to optimize!')
        
        for level in num_levels:
            if lv_dim > level - 0:
                lv_dim = min(lv_dim, level-1)
                raise RuntimeWarning(
                    'The LV dimension can atmost be num_levels-1. '
                    'Setting it to %s in place of %s' %(level-1,lv_dim)
                )
    
        from itertools import product
        levels = []
        for l in num_levels:
            levels.append(torch.arange(l))

        perm = list(product(*levels))
        perm = torch.tensor(perm, dtype=torch.int64)

        #-------------Mapping-------------------------
        perm_dic = {}
        for i, row in enumerate(perm):
            temp = str(row.tolist())
            if temp not in perm_dic.keys():
                perm_dic[temp] = i

        #-------------One_hot_encoding------------------
        for ii in range(perm.shape[-1]):
            if perm[...,ii].min() != 0:
                perm[...,ii] -= perm[...,ii].min()
            
        perm_one_hot = []
        for i in range(perm.size()[1]):
            perm_one_hot.append( torch.nn.functional.one_hot(perm[:,i]) )

        perm_one_hot = torch.concat(perm_one_hot, axis=1)

        return perm_one_hot, perm, perm_dic

    
    def transform_categorical(self, x:torch.Tensor,perm_dict = [], zeta = []) -> None:
        if x.dim() == 1:
            x = x.reshape(-1,1)
        # categorical should start from 0
        if self.training == False:
            x = setlevels(x)
        if self.encoding_type == 'one-hot':
            index = [perm_dict[str(row.tolist())] for row in x]

            if x.dim() == 1:
                x = x.reshape(len(x),)

            return zeta[index,:]  

        elif self.encoding_type  == 'uniform':

            temp2=np.random.uniform(0,1,(len(self.perm), self.uniform_encoding_columns))
            dict={}
            dict2={}

            for i in range(0,self.perm.shape[0]):
                dict[tuple((self.perm[i,:]).numpy())]=temp2[i,:]
            
            for i in range(0,x.shape[0]):
                dict2[i]=dict[tuple((x[i]).numpy())]
            
            x_one_hot= torch.from_numpy(np.array(list(dict2.values())))
                    
        elif self.encoding_type  == 'binary':
            dict={}
            dict2={}
            dict3={}
            dict4={}
            dict3[0]=[]
            dict2[0]=[]
            for i in range(0,self.perm.shape[0]):
                dict[tuple((self.perm[i,:]).numpy())]=str(i)
                dict3[0].append(str(i))
            
            data= DataFrame.from_dict(dict3)
            encoder= BinaryEncoder()
            data_encoded=(encoder.fit_transform(data)).to_numpy()

            for i in range(0,self.perm.shape[0]):
                dict4[str(i)]= data_encoded[i]

            for i in range(0,x.shape[0]):
                dict2[i]=dict4[dict[tuple((x[i]).numpy())]]

            x_one_hot= torch.from_numpy(np.array(list(dict2.values())))
        else:
            raise ValueError ('Invalid type')

                        
        return x_one_hot



from torch import nn
import torch.nn.functional as F 

class FFNN(nn.Module):
    def __init__(self, lmgp, input_size, num_classes, layers,name):
        super(FFNN, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = nn.Linear(input_size, layers[0]) 
            lmgp.register_parameter('fci', self.fci.weight)
            lmgp.register_prior(name = 'latent_prior_fci', prior=gpytorch.priors.NormalPrior(0.,3.), param_or_closure='fci')

            for i in range(1,self.hidden_num):
                #self.h = nn.Linear(neuran[i-1], neuran[i])
                setattr(self, 'h' + str(i), nn.Linear(layers[i-1], layers[i]))
                lmgp.register_parameter('h'+str(i), getattr(self, 'h' + str(i)).weight )
                lmgp.register_prior(name = 'latent_prior'+str(i), prior=gpytorch.priors.NormalPrior(0.,3.), param_or_closure='h'+str(i))
            
            self.fce = nn.Linear(layers[-1], num_classes)
            lmgp.register_parameter('fce', self.fce.weight)
            lmgp.register_prior(name = 'latent_prior_fce', prior=gpytorch.priors.NormalPrior(0.,3.), param_or_closure='fce')
        else:
            self.fci = Linear_MAP(input_size, num_classes, bias = False)
            lmgp.register_parameter(name, self.fci.weight)
            lmgp.register_prior(name = 'latent_prior_'+name, prior=gpytorch.priors.NormalPrior(0,1) , param_or_closure=name)
            #lmgp.sample_from_prior('latent_prior_fci')
            #lmgp.pyro_sample_from_prior()
            #NormalPrior(0,3)
            #LogNormalPrior(0,5)




    def forward(self, x, transform = lambda x: x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        if self.hidden_num > 0:
            x = torch.tanh(self.fci(x))
            for i in range(1,self.hidden_num):
                #x = F.relu(self.h(x))
                x = torch.tanh( getattr(self, 'h' + str(i))(x) )
            
            x = self.fce(x)
        else:
            #self.fci.weight.data = torch.sinh(self.fci.weight.data)
            #self.fci.weight.data = 2*(self.fci.weight.data)
            x = self.fci(x, transform)
        return x





class Linear_MAP(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        

    def forward(self, input, transform = lambda x: x):
        return F.linear(input,transform(self.weight), self.bias)

# elif transformation_of_A_parameters=='exp':
    
#     class Linear_MAP(nn.Linear):
#         def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
#             super().__init__(in_features, out_features, bias, device, dtype)
        

#         def forward(self, input):
#             return F.linear(input, torch.exp(self.weight), self.bias)

# elif transformation_of_A_parameters=='sinh':
#     class Linear_MAP(nn.Linear):
#         def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
#             super().__init__(in_features, out_features, bias, device, dtype)
        

#         def forward(self, input):
#             return F.linear(input, torch.sinh(self.weight), self.bias)