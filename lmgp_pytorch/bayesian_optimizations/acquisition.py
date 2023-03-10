import torch

def EI_fun(best_f, mean, std, maximize = True, si = 0.01):
    
    from torch.distributions import Normal

    # deal with batch evaluation and broadcasting
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f.expand_as(mean) + si) / sigma
    
    
    if not maximize:
        u = -u
        #si = -si
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei

def EI_cost_aware(best_f, mean, std, maximize = True, si = 0.0, cost = None):
    from torch.distributions import Normal

    # deal with batch evaluation and broadcasting
    view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
    mean = mean.view(view_shape)
    sigma = std.view(view_shape)
    u = (mean - best_f.expand_as(mean) + si) / sigma
    
    if cost is None:
        cost = torch.ones(u.shape)    

    cost = cost.view(u.shape)

    if not maximize:
        u = -u
        #si = -si
    
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei/cost