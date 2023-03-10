# LMGP-PyTorch

LMGP-Gpytorch is implementation of Latent map Gaussian process (LMGP) for modeling data with qualitative and quantititave variables. Currently the code uses the many 



## License

Please contact raminb@uci.edu for further info.


## NN-latent
- Adding NN with custom architecture to the LMGP code
- Using dimensionality reduction and clustering techniques to find latent map variables


## Note 1
There is a paramter in gpytorch kernels called active_dims which specied what dimension if the input should be used for that kernel.
This line is defined twice in lmgp.py function. The dimenions has nothing to do with the input and it depends on how we are feeding the x
in the forward method in lmgp. Cuarrently, the latent map dimensions are the first d dimeniosn and then we have other inputs. So, that's how I have adefined the active_dims.

## Note 2
Note that the paramters are coming from the priors distribution. In our matlab code, we have upper and lower bounds for the paramters and the paramters are coming from the sobol sets. I think for MLE, we should use the same approach. For Bayesian, obviously we need priors. If we add the prios in MLE -> MAP, we also get a good result.

# Note 3
Added a kernel similar to matlab called Rough_RBF. It is not clear why this works well with scipy but not with continuation. With Continuation, both RBF and rough_RBF gives me the same results.
With the new Kernel, the scipy is significantly improved. No need to do continuation.