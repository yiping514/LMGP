import numpy as np
import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.transforms.outcome import Standardize
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from lmgp_pytorch.models import LMGP
from lmgp_pytorch.optim import fit_model_scipy
from botorch import fit_gpytorch_model

def run_bo(model, fit_method, EI, X, y, steps, maximize_flag):

    if maximize_flag:
        best_f0 = y.max().reshape(-1,) 
    else:
        best_f0 = y.min().reshape(-1,) 

    ymin_list = []
    xmin_list = []
    cumulative_cost = []
    gain = []
    bestf = []

    for i in range(steps):
        if maximize_flag:
            best_f = y.max().reshape(-1,) 
            gain.append(best_f - best_f0)
            bestf.append(best_f)
        else:
            best_f = y.min().reshape(-1,) 
            gain.append(best_f0 - best_f)
            bestf.append(best_f)



        _ = fit_model_scipy(model, num_restarts= 24)


def run_bo_kg(model_name, train_x, train_obj, problem, 
    cost_model, N_ITER, bounds, 
    target_fidelities, 
    cost_aware_utility,
    fixed_features_list = [{2: 0.0}, {2: 1.0}, {2: 2.0}],
    qual_index = []):

    NUM_RESTARTS = 5 
    RAW_SAMPLES = 32 
    BATCH_SIZE = 2

    bounds = bounds
    target_fidelities = target_fidelities
    fixed_features_list = fixed_features_list

    def initialize_model_and_fit(model_name, train_x, train_obj):
        # define a surrogate model suited for a "training data"-like fidelity parameter
        # in dimension 6, as in [2]

        if model_name == 'botorch': #isinstance(model, SingleTaskMultiFidelityGP):
            train_obj = train_obj.reshape(-1,1)
            model = SingleTaskMultiFidelityGP(train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelity=2)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
        
        else : #isinstance(model, LMGP):
            model = LMGP(train_x, train_obj, qual_ind_lev= qual_index)
            fit_model_scipy(model, num_restarts = 12)
        return model

    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    def get_mfkg(model):
        
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=3,
            columns=[2],
            values=[0.0],
        )
        
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:,:-1],
            q=1,
            num_restarts=10 ,
            raw_samples=32 ,
            options={"batch_limit": 10, "maxiter": 200},
        )
            
        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=32 ,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=project,
        )


    def optimize_mfkg_and_get_observation(mfkg_acqf):
        """Optimizes MFKG and returns a new candidate, observation, and cost."""

        # generate new candidates
        candidates, _ = optimize_acqf_mixed(
            acq_function=mfkg_acqf,
            bounds=bounds,
            fixed_features_list= fixed_features_list,
            q=BATCH_SIZE,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            # batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
        )

        # observe new values
        cost = cost_model(candidates).sum()
        new_x = candidates.detach()
        new_obj = problem(new_x).unsqueeze(-1)
        print(f"candidates:\n{new_x}\n")
        print(f"observations:\n{new_obj}\n\n")
        return new_x, new_obj, cost


    cumulative_cost = 0.0
    cumulative_cost_hist = []
    bestf = []

    for i in range(N_ITER):
        model = initialize_model_and_fit(model_name, train_x, train_obj)
        mfkg_acqf = get_mfkg(model)
        new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
        train_x = torch.cat([train_x, new_x])
        #train_x = setlevels(train_x, [2])
        train_obj = torch.cat([train_obj, new_obj.reshape(-1,)])
        cumulative_cost += cost
        cumulative_cost_hist.append(cumulative_cost.clone().cpu().numpy())
        best_f = train_obj.max().reshape(-1,) 
        bestf.append(best_f.clone().cpu().item())

    return bestf, cumulative_cost_hist


if __name__ == '__main__':
    from lmgp_pytorch.bayesian_optimizations.cost_model import FlexibleFidelityCostModel
    from botorch.utils.sampling import draw_sobol_samples
    from lmgp_pytorch.test_functions.multi_fidelity import Augmented_branin
    from lmgp_pytorch.utils import set_seed
    from lmgp_pytorch.optim import fit_model_scipy

    tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu" if torch.cuda.is_available() else "cpu"),
        }

    set_seed(1)

    bounds = torch.tensor([[-5, 0, 0], [10, 15, 2]], **tkwargs)
    target_fidelities = {2: 0.0}
    problem = lambda x: Augmented_branin(x, negate=True, mapping = {'0.0': 1.0, '1.0': 0.75, '2.0': 0.5}).to(**tkwargs)
    fidelities = torch.tensor([0.0, 1.0, 2.0], **tkwargs)


    ################### Initialize Data ###############################
    def generate_initial_data(n=16):
    # generate training data
        train_x = draw_sobol_samples(bounds[:,:-1],n=n,q = 1, batch_shape= None).squeeze(1).to(**tkwargs)
        train_f = fidelities[torch.randint(3, (n, 1))]
        train_x_full = torch.cat((train_x, train_f), dim=1)
        train_obj = problem(train_x_full).unsqueeze(-1)  # add output dimension
        return train_x_full, train_obj

    Xtrain_x, train_obj = generate_initial_data(n=16)
    train_obj = train_obj.reshape(-1)
    qual_index = {2:3}
    #model = LMGP(Xtrain_x, train_obj, qual_ind_lev=qual_index)
    cost_model = FlexibleFidelityCostModel(values = {'0.0':1.0, '1.0': 0.75, '2.0': 0.5, '3.0': 0.25}, fixed_cost=5.0)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
    
    run_bo_kg(model_name = 'LMGP', train_x = Xtrain_x, train_obj = train_obj, problem = problem, cost_model = cost_model, 
    N_ITER = 10, bounds= bounds, target_fidelities= target_fidelities, cost_aware_utility = cost_aware_utility,qual_index = qual_index)