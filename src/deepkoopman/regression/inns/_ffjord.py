import torch
import torch.nn as nn
from torchdiffeq import odeint
from ._base_inn import BaseINN

"""
We adapt the architecture of FFJORD (Grathwohl et al., 2019) to serve purely as a learnable invertible mapping within the Koopman learning framework.
In contrast to the original FFJORD model, we do not estimate log-densities, nor do we train using the Hutchinson trace estimator.
Instead, we leverage the continuous invertibility of the ODE-based transformation to encode and decode data between the original space and the latent Koopman-invariant space.
"""


class ODEF(nn.Module):
    def __init__(self, dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, dim)
        )

    def forward(self, t, x):
        return self.net(x)


class FFJORD(BaseINN):
    def __init__(self, input_size, hidden_size=64, solver='dopri5', atol=1e-5, rtol=1e-5, integration_time=1.0):
        super().__init__(input_size)
        self.func = ODEF(input_size, hidden_size)
        self.integration_time = torch.tensor([0.0, integration_time]).float()
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

    def forward(self, x):
        if x.dim() == 3:
            b, t, f = x.shape
            x = x.reshape(-1, f)
            z = odeint(self.func, x, self.integration_time.to(x.device), method=self.solver,
                       atol=self.atol, rtol=self.rtol)[-1]
            return z.reshape(b, t, -1)
        elif x.dim() == 2:
            z = odeint(self.func, x, self.integration_time.to(x.device), method=self.solver,
                       atol=self.atol, rtol=self.rtol)[-1]
            return z
        else:
            raise ValueError("Only 2D and 3D inputs supported")

    def inverse(self, z):
        rev_time = torch.tensor([self.integration_time[-1], self.integration_time[0]]).float().to(z.device)
        if z.dim() == 3:
            b, t, f = z.shape
            z = z.reshape(-1, f)
            x = odeint(self.func, z, rev_time, method=self.solver,
                       atol=self.atol, rtol=self.rtol)[-1]
            return x.reshape(b, t, -1)
        elif z.dim() == 2:
            x = odeint(self.func, z, rev_time, method=self.solver,
                       atol=self.atol, rtol=self.rtol)[-1]
            return x
        else:
            raise ValueError("Only 2D and 3D inputs supported")
