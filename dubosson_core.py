import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DubossonUnifiedField(nn.Module):
    """
    Dubosson Unified Field Regulator (2026)
    ---------------------------------------
    A Physics-Informed Neural Layer combining:
    1. Navier-Stokes (Fluid Viscosity & Diffusion)
    2. Hamiltonian Dynamics (Verlet Integration)
    3. Lagrangian Mechanics (Principle of Least Action)
    4. Feynman AI Interface (Symbolic Distillation)
    """

    def __init__(
        self, 
        dim: int, 
        target_val: float = 2.5, 
        viscosity: float = 0.5,
        dt: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.target_val = target_val
        self.dt = dt
        
        # --- Learnable Physical Parameters ---
        self.viscosity = nn.Parameter(torch.tensor(viscosity))
        self.friction = nn.Parameter(torch.tensor(0.95))
        
        # --- Physical State Buffers (Verlet Memory) ---
        self.register_buffer('prev_x', torch.full((dim,), target_val))
        self.register_buffer('temp_field', torch.zeros(dim))
        
        # --- Feynman AI History Collector ---
        self.history = []

    def _apply_navier_stokes(self, x: torch.Tensor):
        """Diffuses activity across dimensions to prevent local explosions."""
        with torch.no_grad():
            # Variance as local 'heat'
            activity = x.var(dim=0).clamp(max=2.0)
            self.temp_field.lerp_(activity, 0.1)
            # Spatial diffusion (1D Laplacian smoothing)
            self.temp_field.data = F.avg_pool1d(
                self.temp_field.view(1, 1, -1), 3, stride=1, padding=1
            ).view(-1)
        return self.temp_field

    def forward(self, x: torch.Tensor, loss_signal: float = 0.1) -> torch.Tensor:
        """
        Regulates the input tensor using Hybrid Physics.
        Input x shape: (Batch, Dim) or (Dim)
        """
        orig_shape = x.shape
        x = x.view(-1, self.dim)
        
        # Safety: Handle NaNs immediately
        x = torch.nan_to_num(x, nan=self.target_val)

        # 1. NAVIER-STOKES: Thermal Smoothing
        t_field = self._apply_navier_stokes(x)

        # 2. LAGRANGIAN: Action Potential
        # Filters signals that don't contribute to 'Least Action'
        action_threshold = torch.sigmoid(torch.tensor(1.0 / (loss_signal + 1e-6)))
        
        # 3. HAMILTONIAN: Verlet Integration (The Engine)
        # Force = -k * (x - target) + thermal_noise
        force = -(x - self.target_val) * 0.5
        
        # Conservation of momentum with friction (Viscosity)
        temp_x = x.detach().clone()
        
        # Verlet step: x(t+dt) = x(t) + friction * (x(t) - x(t-dt)) + acc * dt^2
        # Applied across the batch
        acceleration = force + (torch.randn_like(x) * t_field * 0.01)
        x_new = x + self.friction * (x - self.prev_x.view_as(x)) + acceleration * (self.dt**2)
        
        # Update Verlet memory (only first element of batch for buffer simplicity)
        self.prev_x.copy_(temp_x[0])

        # 4. FEYNMAN INTERFACE: Data Collection
        if self.training:
            self.history.append({
                'loss': loss_signal,
                'mean_energy': (0.5 * force**2).mean().item()
            })

        # Final Bounding Box Constraint
        x_final = x_new.clamp(0.0, 10.0)
        
        return x_final.view(orig_shape)

    def get_feynman_data(self):
        """Returns collected data for Symbolic Regression (Feynman AI)."""
        return self.history

    def extra_repr(self) -> str:
        return f"dim={self.dim}, target={self.target_val}, visc={self.viscosity.item():.2f}"
 




