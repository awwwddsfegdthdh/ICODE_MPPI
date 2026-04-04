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
        context_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dt: float = 0.02,
        predict_obs_distance: bool = False,
        obs_head_hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = int(context_dim)
        self.dt = dt
        self.predict_obs_distance_enabled = bool(predict_obs_distance)
        self.obs_head_hidden_dim = int(obs_head_hidden_dim)
        self.input_norm_enabled = False
        core_in_dim = int(state_dim + self.context_dim)

        self.register_buffer("x_mean", torch.zeros((self.state_dim,), dtype=torch.float32), persistent=False)
        self.register_buffer("x_std", torch.ones((self.state_dim,), dtype=torch.float32), persistent=False)
        self.register_buffer("u_mean", torch.zeros((self.action_dim,), dtype=torch.float32), persistent=False)
        self.register_buffer("u_std", torch.ones((self.action_dim,), dtype=torch.float32), persistent=False)
        self.register_buffer("ctx_mean", torch.zeros((self.context_dim,), dtype=torch.float32), persistent=False)
        self.register_buffer("ctx_std", torch.ones((self.context_dim,), dtype=torch.float32), persistent=False)

        self.f_net = self._build_mlp(
            in_dim=core_in_dim,
            out_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.g_net = self._build_mlp(
            in_dim=core_in_dim,
            out_dim=state_dim * action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.obs_head = None
        if self.predict_obs_distance_enabled:
            # Predict next-step nearest-obstacle distance from current (x_t, u_t).
            self.obs_head = nn.Sequential(
                nn.Linear(core_in_dim + action_dim, self.obs_head_hidden_dim),
                nn.Tanh(),
                nn.Linear(self.obs_head_hidden_dim, self.obs_head_hidden_dim),
                nn.Softplus(),
                nn.Linear(self.obs_head_hidden_dim, 1),
                nn.Softplus(),
            )

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> nn.Sequential:
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _merge_state_context(self, x: torch.Tensor, ctx: torch.Tensor | None) -> torch.Tensor:
        if self.context_dim <= 0:
            return x
        if ctx is None:
            ctx = torch.zeros((x.shape[0], self.context_dim), dtype=x.dtype, device=x.device)
        return torch.cat([x, ctx], dim=-1)

    @staticmethod
    def _to_buffer_tensor(v, ref: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(v, dtype=ref.dtype, device=ref.device).reshape(-1)
        return t

    def set_input_normalization(
        self,
        x_mean,
        x_std,
        u_mean,
        u_std,
        ctx_mean=None,
        ctx_std=None,
        enabled: bool = True,
    ) -> None:
        x_mean_t = self._to_buffer_tensor(x_mean, self.x_mean)
        x_std_t = self._to_buffer_tensor(x_std, self.x_std)
        u_mean_t = self._to_buffer_tensor(u_mean, self.u_mean)
        u_std_t = self._to_buffer_tensor(u_std, self.u_std)
        if x_mean_t.numel() != self.state_dim or x_std_t.numel() != self.state_dim:
            raise ValueError(f"x normalization dim mismatch: expected {self.state_dim}")
        if u_mean_t.numel() != self.action_dim or u_std_t.numel() != self.action_dim:
            raise ValueError(f"u normalization dim mismatch: expected {self.action_dim}")
        self.x_mean.copy_(x_mean_t)
        self.x_std.copy_(torch.clamp(x_std_t, min=1e-6))
        self.u_mean.copy_(u_mean_t)
        self.u_std.copy_(torch.clamp(u_std_t, min=1e-6))

        if self.context_dim > 0:
            if ctx_mean is None:
                ctx_mean_t = torch.zeros((self.context_dim,), dtype=self.ctx_mean.dtype, device=self.ctx_mean.device)
            else:
                ctx_mean_t = self._to_buffer_tensor(ctx_mean, self.ctx_mean)
            if ctx_std is None:
                ctx_std_t = torch.ones((self.context_dim,), dtype=self.ctx_std.dtype, device=self.ctx_std.device)
            else:
                ctx_std_t = self._to_buffer_tensor(ctx_std, self.ctx_std)
            if ctx_mean_t.numel() != self.context_dim or ctx_std_t.numel() != self.context_dim:
                raise ValueError(f"ctx normalization dim mismatch: expected {self.context_dim}")
            self.ctx_mean.copy_(ctx_mean_t)
            self.ctx_std.copy_(torch.clamp(ctx_std_t, min=1e-6))

        self.input_norm_enabled = bool(enabled)

    def _normalize_inputs(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        ctx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self.input_norm_enabled:
            return x, u, ctx
        xn = (x - self.x_mean.view(1, -1).to(dtype=x.dtype, device=x.device)) / self.x_std.view(1, -1).to(dtype=x.dtype, device=x.device)
        un = (u - self.u_mean.view(1, -1).to(dtype=u.dtype, device=u.device)) / self.u_std.view(1, -1).to(dtype=u.dtype, device=u.device)
        if self.context_dim > 0:
            if ctx is None:
                ctx = torch.zeros((x.shape[0], self.context_dim), dtype=x.dtype, device=x.device)
            cn = (ctx - self.ctx_mean.view(1, -1).to(dtype=ctx.dtype, device=ctx.device)) / self.ctx_std.view(1, -1).to(dtype=ctx.dtype, device=ctx.device)
        else:
            cn = ctx
        return xn, un, cn

    def derivative(self, x: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """Return dx/dt = f(x) + g(x) @ u."""
        x_n, u_n, c_n = self._normalize_inputs(x=x, u=u, ctx=ctx)
        x_in = self._merge_state_context(x=x_n, ctx=c_n)
        f_val = self.f_net(x_in)
        g_val = self.g_net(x_in).view(-1, self.state_dim, self.action_dim)
        u_vec = u_n.unsqueeze(-1)
        gu = torch.bmm(g_val, u_vec).squeeze(-1)
        return f_val + gu

    def forward(self, x: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        dx = self.derivative(x, u, ctx=ctx)
        return x + self.dt * dx

    def predict_obstacle_distance(self, x: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Predict next-step nearest obstacle distance (scalar per sample).
        Returns shape [B].
        """
        if self.obs_head is None:
            raise RuntimeError("Obstacle-distance head is disabled for this checkpoint/model.")
        x_n, u_n, c_n = self._normalize_inputs(x=x, u=u, ctx=ctx)
        x_in = self._merge_state_context(x=x_n, ctx=c_n)
        xu = torch.cat([x_in, u_n], dim=-1)
        d = self.obs_head(xu).squeeze(-1)
        return d
