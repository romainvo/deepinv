import torch
import torch.nn as nn
from deepinv.optim.optim_base import ProxOptim

class RED(ProxOptim):
    '''
    Regulatization by Denoiser (RED) algorithms for Image Restoration. Consists in replacing Id-grad_g with a denoiser.

    :param denoiser: Dennoiser model
    :param sigma_denoiser: Denoiser noise standart deviation.
    '''
    def __init__(self, denoiser, sigma_denoiser = 0.05, **kwargs):
        super().__init__(**kwargs)

        assert self.algo_name in ['GD','PGS'], 'RED only works with GD or PGD'

        self.denoiser = denoiser
        if not self.unroll : 
            if isinstance(sigma_denoiser, float):
                self.sigma_denoiser = [sigma_denoiser] * self.max_iter
            elif isinstance(sigma_denoiser, list):
                assert len(sigma_denoiser) == self.max_iter
                self.sigma_denoiser = sigma_denoiser
            else:
                raise ValueError('sigma_denoiser must be either int/float or a list of length max_iter') 
        else : 
            assert isinstance(sigma_denoiser, float) # the initial parameter is uniform across layer int in that case
            self.register_parameter(name='sigma_denoiser',
                                param=torch.nn.Parameter(torch.tensor(sigma_denoiser, device=self.device),
                                requires_grad=True))

        self.grad_g = lambda x,it : x-denoiser(x, self.sigma_denoiser[it])