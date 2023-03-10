import torch
from torch import Tensor
from botorch.models.deterministic import DeterministicModel


class FlexibleFidelityCostModel(DeterministicModel):
    def __init__(
        self,
        fidelity_dims: list = [-1],
         values = {'0.0':1.0, '1.0': 0.50, '2.0': 0.25, '3.0': 0.125},
         fixed_cost: float = 0.01,
         )->None:
        r'Gets the cost according to the fidelity level'
        super().__init__()
        self.cost_values=values
        self.fixed_cost=fixed_cost
        self.fidelity_dims=fidelity_dims
        self.register_buffer("weights", torch.tensor([1.0]))
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        
        cost = list(map(lambda x: self.cost_values[str(float(x))], X[..., self.fidelity_dims].flatten()))
        cost = torch.tensor(cost).to(X)
        cost.reshape(X[..., self.fidelity_dims].shape)
        return self.fixed_cost + cost