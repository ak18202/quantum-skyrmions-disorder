import jax.numpy as jnp
from flax import linen as nn

class RealImagFFNN(nn.Module):
    L: int
    alpha: int

    def setup(self):
        hidden_dim = self.alpha * self.L * self.L
        
        # NN for rho
        self.rho_net = nn.Sequential([
            nn.Dense(features=hidden_dim),
            nn.relu,
            nn.Dense(features=hidden_dim),
            nn.relu,
            nn.Dense(features=1)
        ])

        # NN for phi
        self.phi_net = nn.Sequential([
            nn.Dense(features=hidden_dim),
            nn.relu,
            nn.Dense(features=hidden_dim),
            nn.relu,
            nn.Dense(features=1)
        ])

    def __call__(self, s):
        x = jnp.reshape(s, (-1, self.L * self.L))
        phi = self.phi_net(x)
        phi = jnp.squeeze(phi, axis=-1)
        rho = self.rho_net(x)
        rho = jnp.squeeze(rho, axis=-1)
        return rho + 1j * phi