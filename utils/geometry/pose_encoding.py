"""
Pose encoding utilities for geometric transformations.

Encodes rotation matrix R and translation vector T into embeddings
for use in transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D continuous representation.

    Uses the first two columns of the rotation matrix as the 6D representation.
    This is a continuous and unique representation of SO(3).

    Reference: "On the Continuity of Rotation Representations in Neural Networks"
    Zhou et al., CVPR 2019

    Args:
        R: (..., 3, 3) rotation matrix

    Returns:
        r6d: (..., 6) 6D representation
    """
    # Take first two columns
    r6d = R[..., :, :2].reshape(*R.shape[:-2], 6)
    return r6d


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).

    Args:
        R: (..., 3, 3) rotation matrix

    Returns:
        q: (..., 4) quaternion [w, x, y, z]
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    # Compute quaternion components
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # w = sqrt(1 + trace) / 2
    q[:, 0] = torch.sqrt(1.0 + trace.clamp(min=0.0)) / 2.0

    # Avoid division by zero
    w = q[:, 0].clamp(min=1e-6)

    # x, y, z components
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w)

    # Normalize
    q = F.normalize(q, p=2, dim=-1)

    return q.reshape(*batch_shape, 4)


class PoseEncoder(nn.Module):
    """
    Encode SE(3) transformation (R, T) into embeddings.

    Supports multiple encoding strategies:
    - '6d': 6D rotation representation + translation (9D total)
    - 'quaternion': Quaternion + translation (7D total)
    - 'matrix': Flattened 4x4 matrix (16D total)
    """

    def __init__(
        self,
        d_model: int = 512,
        encoding_type: str = '6d',
        use_mlp: bool = True,
        mlp_hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            d_model: Output embedding dimension
            encoding_type: Type of rotation encoding ('6d', 'quaternion', 'matrix')
            use_mlp: Whether to use MLP to project to d_model
            mlp_hidden_dim: Hidden dimension of MLP (default: 2 * d_model)
        """
        super().__init__()

        self.d_model = d_model
        self.encoding_type = encoding_type
        self.use_mlp = use_mlp

        # Determine input dimension based on encoding type
        if encoding_type == '6d':
            self.input_dim = 9  # 6D rotation + 3D translation
        elif encoding_type == 'quaternion':
            self.input_dim = 7  # 4D quaternion + 3D translation
        elif encoding_type == 'matrix':
            self.input_dim = 16  # 4x4 matrix flattened (only use 12 unique values)
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")

        # MLP projection
        if use_mlp:
            hidden_dim = mlp_hidden_dim or (2 * d_model)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            # Simple linear projection
            self.mlp = nn.Linear(self.input_dim, d_model)

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """
        Encode SE(3) transformation matrix.

        Args:
            T: (..., 4, 4) transformation matrix

        Returns:
            pose_emb: (..., d_model) pose embedding
        """
        batch_shape = T.shape[:-2]

        # Extract rotation and translation
        R = T[..., :3, :3]  # (..., 3, 3)
        t = T[..., :3, 3]   # (..., 3)

        # Encode rotation based on type
        if self.encoding_type == '6d':
            r_encoded = rotation_matrix_to_6d(R)  # (..., 6)
        elif self.encoding_type == 'quaternion':
            r_encoded = rotation_matrix_to_quaternion(R)  # (..., 4)
        elif self.encoding_type == 'matrix':
            # Flatten entire 4x4 matrix (16 values)
            pose_vec = T.reshape(*batch_shape, 16)  # (..., 16)
            pose_emb = self.mlp(pose_vec)
            return pose_emb
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

        # Concatenate rotation and translation
        pose_vec = torch.cat([r_encoded, t], dim=-1)  # (..., 6+3) or (..., 4+3)

        # Project to d_model
        pose_emb = self.mlp(pose_vec)  # (..., d_model)

        return pose_emb
