# ==========================================
# Bloc 1 : Configuration de l'Environnement
# ==========================================
!pip install -q torch py3Dmol matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import py3Dmol
import matplotlib.pyplot as plt
from google.colab import output
output.enable_custom_widget_manager()

print("✅ Environnement prêt pour le Dubosson Unified Field.")

# ==========================================
# Bloc 2 : Le Moteur Physique 3D
# ==========================================
class Dubosson3DRegulator(nn.Module):
    def __init__(self, dim, target_radius=2.5, dt=0.1, friction=0.92):
        super().__init__()
        self.dim = dim
        self.dt = dt
        self.friction = friction
        
        # Cible symbolique (Hélice Feynman)
        t = torch.linspace(0, 10, dim)
        x_target = target_radius * torch.cos(t)
        y_target = target_radius * torch.sin(t)
        z_target = t
        self.register_buffer('target_coords', torch.stack([x_target, y_target, z_target], dim=1))
        
        # Mémoire de Verlet (Positions précédentes)
        self.register_buffer('prev_coords', self.target_coords.clone())

    def forward(self, current_coords, loss_signal=0.1):
        # 1. Force de Rappel vers la structure idéale
        force = -(current_coords - self.target_coords) * 0.5
        
        # 2. Intégration de Verlet (Stabilisation)
        temp_coords = current_coords.detach().clone()
        new_coords = current_coords + self.friction * (current_coords - self.prev_coords) + force * (self.dt**2)
        
        # Mise à jour de la mémoire
        self.prev_coords.copy_(temp_coords)
        return new_coords

def to_pdb(coords):
    """Convertit les coordonnées au format PDB."""
    pdb_lines = []
    for i, (x, y, z) in enumerate(coords):
        pdb_lines.append(f"ATOM  {i+1:>5}  CA  ALA A{i+1:>4}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C")
    return "\n".join(pdb_lines)

# ==========================================
# Bloc 3 : Simulation et Rendu 3D
# ==========================================
# 1. Paramètres
dim = 50
regulator = Dubosson3DRegulator(dim=dim)

# 2. État initial (Protéine dépliée)
current_coords = regulator.target_coords + torch.randn(dim, 3) * 3.0

# 3. Boucle de repliement
print("Folding in progress...")
for _ in range(150):
    current_coords = regulator(current_coords)

# 4. Visualisation Interactive
view = py3Dmol.view(width=800, height=500)
view.addModel(to_pdb(current_coords), 'pdb')
view.setStyle({'cartoon': {'color': 'spectrum'}, 'stick': {'radius': 0.1}})

# Liaisons Hydrogène i+4
for i in range(dim - 4):
    p1 = current_coords[i].tolist()
    p2 = current_coords[i+4].tolist()
    view.addCylinder({'start': {'x':p1[0], 'y':p1[1], 'z':p1[2]},
                      'end':   {'x':p2[0], 'y':p2[1], 'z':p2[2]},
                      'radius': 0.05, 'color': 'white', 'dashed': True})

view.zoomTo()
view.show()
print("✅ Structure 3D stabilisée.")
