import torch
import torch.nn as nn


class ICODEDynamics(nn.Module):
    """
    Learnable ICODE-style dynamics:
      x_{t+1} = x_t + dt * ( f(x_t) + g(x_t) @ u_t )

    - f/g are MLPs.
    - g(x) outputs [state_dim, action_dim] gain matrix per sample.
    """

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dt: float = 0.02,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt

        self.f_net = self._build_mlp(
            in_dim=state_dim,
            out_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.g_net = self._build_mlp(
            in_dim=state_dim,
            out_dim=state_dim * action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def derivative(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return dx/dt = f(x) + g(x) @ u."""
        f_val = self.f_net(x)
        g_val = self.g_net(x).view(-1, self.state_dim, self.action_dim)
        u_vec = u.unsqueeze(-1)
        gu = torch.bmm(g_val, u_vec).squeeze(-1)
        return f_val + gu

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        dx = self.derivative(x, u)
        return x + self.dt * dx
