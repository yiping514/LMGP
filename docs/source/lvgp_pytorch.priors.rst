Priors
======

Priors distributions are used for for drawing initial guesses for the hyperparameters as well as in MAP estimation.  
Most of the priors are imported from `gpytorch.priors`. Refer to the gpytorch documentation at https://docs.gpytorch.ai/en/v1.5.1/priors.html. 

Additional priors
-----------------

There are a few prior distributions that aren't implemented by `gpytorch`. 

.. autoclass:: lvgp_pytorch.priors.LogHalfHorseshoePrior
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: lvgp_pytorch.priors.MollifiedUniformPrior
   :members:
   :undoc-members:
   :show-inheritance:


Default priors for hyperparameters
----------------------------------

The default priors used for different hyperparameters are shown here. There are a few assumptions:

* Training response variable is standardized
* Numerical/Quantitative inputs are all scaled to [0,1]

.. list-table:: 
   :widths: 15, 40
   :header-rows: 1

   *  - Hyperparameter
      - Prior
   
   *  - Mean
      - `NormalPrior(loc=0,scale=1)`
   
   *  - Log variance
      - `NormalPrior(loc=0,scale=1)`
   
   *  - Log noise variane
      - `LogHalfHorseshoePrior(scale=0.01,lb=1e-6)`  

   *  - Log lengthscales
      - `MollifiedUniformPrior(a=math.log(0.01),b=math.log(10),tail_sigma=0.1)`
   
   *  - Raw latent variables (LVs)
      - `NormalPrior(loc=0,scale=1)`
   
   *  - Precision for LVs
      - `GammaPrior(concentration=2.,rate=1.)` 