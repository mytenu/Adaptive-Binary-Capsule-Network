"""
Adaptive Binary Capsule Network (ABCN)
Paper: "Adaptive Binary Capsule Network for Complex Image Recognition"
Authors: Mavis Serwaa Yeboah, Patrick Kwabena Mensah, Adebayo Felix Adekoya, Mighty Abra Ayidzoe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ─────────────────────────────────────────────────────────────
# 1.  Gaussian Kernel Generation  (Algorithm 1)
# ─────────────────────────────────────────────────────────────
def gaussian_kernel(size: int, std: float) -> torch.Tensor:
    """
    Generate a 2-D Gaussian kernel.

    Args:
        size (int): Kernel size (must be odd).
        std  (float): Standard deviation of the Gaussian.

    Returns:
        Tensor of shape (size, size), normalised to sum = 1.
    """
    d = float(size)
    half = math.ceil(d / 2)
    coords = torch.arange(-half + 1, half + 1, dtype=torch.float32)  # range(-d/2+1, d/2+1)
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    g = torch.exp(-(x ** 2 + y ** 2) / (2.0 * std ** 2))
    g = g / g.sum()                                                   # normalise
    return g


# ─────────────────────────────────────────────────────────────
# 2.  Adaptive Contrast Spatial Filtering (ACSF)  (Algorithm 2)
# ─────────────────────────────────────────────────────────────
class ACSFLayer(nn.Module):
    """
    Adaptive Contrast Spatial Filtering layer.

    Steps per channel k:
        1. Adaptive thresholding  T(x,y) = (I - mu) / (sigma * s)
        2. Gaussian kernel generation
        3. Spatial filtering  P = T * G  (depthwise convolution)

    Args:
        in_channels  (int):   Number of input feature-map channels.
        kernel_size  (int):   Size of the Gaussian kernel (default 3).
        std          (float): Std-dev for Gaussian kernel (default 1.0).
        scale        (float): Scaling factor s in adaptive thresholding (default 1.0).
    """

    def __init__(self, in_channels: int, kernel_size: int = 3,
                 std: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.std = std
        self.scale = scale
        self.padding = kernel_size // 2

        # Build a fixed (non-learnable) depthwise Gaussian kernel
        kernel = gaussian_kernel(kernel_size, std)          # (k, k)
        # Expand to (in_channels, 1, k, k) for depthwise conv
        kernel = kernel.unsqueeze(0).unsqueeze(0)           # (1,1,k,k)
        kernel = kernel.repeat(in_channels, 1, 1, 1)        # (C,1,k,k)
        self.register_buffer("gaussian", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            Enhanced feature maps of the same shape.
        """
        # ── Adaptive thresholding ──────────────────────────────────
        mu  = x.mean(dim=[2, 3], keepdim=True)              # per-channel mean
        sig = x.std (dim=[2, 3], keepdim=True).clamp(min=1e-6)
        t   = (x - mu) / (sig * self.scale)                 # zero mean, unit var

        # ── Spatial filtering (depthwise conv with Gaussian) ───────
        out = F.conv2d(t, self.gaussian,
                       padding=self.padding,
                       groups=self.in_channels)
        return out


# ─────────────────────────────────────────────────────────────
# 3.  Local Binary Pattern (LBP) layer
# ─────────────────────────────────────────────────────────────
class LBPLayer(nn.Module):
    """
    Differentiable approximation of the Local Binary Pattern (LBP) descriptor.

    For each pixel (x, y) with 8 neighbours in a 3×3 patch, each neighbour
    is compared to the centre pixel using a soft threshold (sigmoid).  The
    8 resulting binary-like values are summed with positional weights 2^i to
    produce one LBP value per pixel, exactly mirroring the paper's decimal
    encoding step.

    Args:
        in_channels (int): Number of input channels.
        radius      (int): Neighbourhood radius (default 1 → 3×3 kernel).
    """

    def __init__(self, in_channels: int, radius: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.radius = radius
        self.temperature = 1.0   # controls sharpness of soft threshold

        # Fixed 8-neighbour offsets for a 3×3 patch (excluding centre)
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]
        # Binary weights  2^0, 2^1, …, 2^7
        weights = [2 ** i for i in range(8)]

        self.register_buffer(
            "offsets",
            torch.tensor(offsets, dtype=torch.long)
        )
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32).view(1, 8, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            LBP feature map (B, C, H, W)
        """
        B, C, H, W = x.shape
        pad = self.radius
        x_pad = F.pad(x, [pad] * 4, mode="reflect")

        neighbours = []
        for dy, dx in self.offsets.tolist():
            # Shift to extract neighbours
            ny = dy + pad
            nx = dx + pad
            neighbour = x_pad[:, :, ny:ny + H, nx:nx + W]  # (B,C,H,W)
            neighbours.append(neighbour)

        # Stack: (B, C, 8, H, W)
        neighbours = torch.stack(neighbours, dim=2)
        centre = x.unsqueeze(2)                             # (B, C, 1, H, W)

        # Soft threshold: sigmoid((neighbour - centre) / T)
        diff  = (neighbours - centre) / self.temperature
        bits  = torch.sigmoid(diff)                         # ≈ 1 if >= centre

        # Weighted sum → decimal LBP value  (B, C, H, W)
        # weights shape: (1, 8, 1, 1) → broadcast over (B, C, 8, H, W) mean reduction
        # We need  sum over the 8 neighbours
        weights = self.weights.unsqueeze(0).unsqueeze(0)    # (1,1,8,1,1)
        lbp = (bits * weights).sum(dim=2)                   # (B, C, H, W)
        return lbp


# ─────────────────────────────────────────────────────────────
# 4.  Squash activation  (Eq. 20)
# ─────────────────────────────────────────────────────────────
def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    mj = (||rj||^2 / (1 + ||rj||^2)) * (rj / ||rj||)
    """
    norm_sq = (x ** 2).sum(dim=dim, keepdim=True)
    norm    = norm_sq.sqrt().clamp(min=1e-8)
    scale   = norm_sq / (1.0 + norm_sq)
    return scale * x / norm


# ─────────────────────────────────────────────────────────────
# 5.  Primary Capsule layer
# ─────────────────────────────────────────────────────────────
class PrimaryCapsules(nn.Module):
    """
    Converts the final conv feature map into a set of primary capsules.

    Args:
        in_channels    (int): Channels from preceding conv layer.
        num_capsules   (int): Number of capsule types (default 32).
        capsule_dim    (int): Dimension of each capsule vector (default 8).
        kernel_size    (int): Conv kernel size (default 9).
        stride         (int): Stride (default 2).
    """

    def __init__(self, in_channels: int, num_capsules: int = 32,
                 capsule_dim: int = 8, kernel_size: int = 9, stride: int = 2):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim  = capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            (B, num_primary_caps, capsule_dim)
        """
        out = self.conv(x)                                  # (B, N*D, H', W')
        B = out.size(0)
        out = out.view(B, self.num_capsules, self.capsule_dim, -1)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(B, -1, self.capsule_dim)             # (B, num_caps, D)
        return squash(out)


# ─────────────────────────────────────────────────────────────
# 6.  Class Capsule layer with Dynamic Routing  (Eqs. 17-20)
# ─────────────────────────────────────────────────────────────
class ClassCapsules(nn.Module):
    """
    Class capsule layer using dynamic routing by agreement.

    Args:
        num_in_capsules  (int): Number of primary capsules.
        in_capsule_dim   (int): Dim of primary capsule vectors (default 8).
        num_classes      (int): Number of output class capsules.
        out_capsule_dim  (int): Dim of class capsule vectors (default 16).
        num_routing      (int): Number of routing iterations (default 3).
    """

    def __init__(self, num_in_capsules: int, in_capsule_dim: int = 8,
                 num_classes: int = 4, out_capsule_dim: int = 16,
                 num_routing: int = 3):
        super().__init__()
        self.num_routing     = num_routing
        self.num_classes     = num_classes
        self.out_capsule_dim = out_capsule_dim

        # Transformation matrices  W_ij  (Eq. 17)
        self.W = nn.Parameter(
            torch.randn(1, num_in_capsules, num_classes,
                        out_capsule_dim, in_capsule_dim) * 0.01
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, num_in_caps, in_capsule_dim)
        Returns:
            (B, num_classes, out_capsule_dim)
        """
        B, N, D = u.shape
        K = self.num_classes

        # Prediction vectors  u_hat_{j|i} = W_ij @ u_i  (Eq. 17)
        u_hat = torch.matmul(
            self.W,                                          # (1, N, K, D_out, D_in)
            u.unsqueeze(2).unsqueeze(-1).expand(B, N, K, D, 1)
            # → (B, N, K, D_in, 1)
        ).squeeze(-1)                                        # (B, N, K, D_out)

        # Detach u_hat for routing log updates (no gradient through b)
        u_hat_detached = u_hat.detach()

        # Log coupling coefficients  b_{ij} = 0 initially
        b = torch.zeros(B, N, K, device=u.device)

        for i in range(self.num_routing):
            # Coupling coefficients via softmax  (Eq. 18)
            c = F.softmax(b, dim=2)                         # (B, N, K)

            if i == self.num_routing - 1:
                # Last iteration: use real u_hat (so gradients flow)
                s = (c.unsqueeze(-1) * u_hat).sum(dim=1)    # (B, K, D_out)
                v = squash(s, dim=-1)
            else:
                s = (c.unsqueeze(-1) * u_hat_detached).sum(dim=1)
                v = squash(s, dim=-1)
                # Agreement: b += u_hat · v  (dot product)
                agreement = (u_hat_detached * v.unsqueeze(1)).sum(dim=-1)
                b = b + agreement

        return v                                             # (B, K, D_out)


# ─────────────────────────────────────────────────────────────
# 7.  Decoder network
# ─────────────────────────────────────────────────────────────
class CapsuleDecoder(nn.Module):
    """
    Three fully-connected layers: 512 → 1024 → 2352.
    Reconstructs the input image from the masked class capsule vector.

    Args:
        in_dim       (int): Capsule vector dim × num_classes.
        image_pixels (int): Total pixels in output (H×W×C).  Default 28*28*3=2352.
    """

    def __init__(self, in_dim: int, image_pixels: int = 28 * 28 * 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, image_pixels),
            nn.Sigmoid(),
        )
        self.image_pixels = image_pixels

    def forward(self, v: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            v      : (B, num_classes, capsule_dim)
            labels : (B,) ground-truth class indices for masking.
                     If None, uses the most-active capsule (inference).
        Returns:
            Reconstructed image flattened to (B, image_pixels).
        """
        B, K, D = v.shape
        lengths  = v.norm(dim=-1)                           # (B, K)

        if labels is not None:
            mask = F.one_hot(labels, K).float()             # (B, K)
        else:
            mask = F.one_hot(lengths.argmax(dim=-1), K).float()

        masked = (v * mask.unsqueeze(-1)).view(B, K * D)    # (B, K*D)
        return self.fc(masked)


# ─────────────────────────────────────────────────────────────
# 8.  Full Adaptive Binary Capsule Network
# ─────────────────────────────────────────────────────────────
class AdaptiveBinaryCapsNet(nn.Module):
    """
    Full ABCN architecture (Figure 2 in the paper):

        Input (B, 3, 28, 28)
          ↓  ACSF Layer 1      (3 channels)
          ↓  LBP Layer         (3 channels)
          ↓  ACSF Layer 2      (3 channels)
          ↓  Conv 1  64 × 3×3  → ReLU
          ↓  Conv 2  256 × 3×3 → ReLU
          ↓  Conv 3  256 × 9×9 → ReLU
          ↓  Primary Capsules  (32 caps × 8-D)
          ↓  Class Capsules    (num_classes caps × 16-D)  dynamic routing
          ↓  Decoder           512 → 1024 → 2352

    Args:
        num_classes  (int): Number of output classes (default 4).
        image_size   (int): Spatial size of input images (default 28).
        in_channels  (int): Input image channels (default 3).
    """

    def __init__(self, num_classes: int = 4,
                 image_size: int = 28, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.image_size  = image_size
        self.in_channels = in_channels

        # ── Encoder ────────────────────────────────────────────────
        self.acsf1 = ACSFLayer(in_channels,  kernel_size=3, std=1.0)
        self.lbp   = LBPLayer (in_channels,  radius=1)
        self.acsf2 = ACSFLayer(in_channels,  kernel_size=3, std=1.0)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,  256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ── Capsule layers ──────────────────────────────────────────
        self.primary_caps = PrimaryCapsules(
            in_channels=256,
            num_capsules=32,
            capsule_dim=8,
            kernel_size=3,
            stride=2,
        )

        # Compute number of primary capsules dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            dummy = self._forward_encoder(dummy)
            num_primary = self.primary_caps(dummy).shape[1]

        self.class_caps = ClassCapsules(
            num_in_capsules=num_primary,
            in_capsule_dim=8,
            num_classes=num_classes,
            out_capsule_dim=16,
            num_routing=3,
        )

        # ── Decoder ─────────────────────────────────────────────────
        self.decoder = CapsuleDecoder(
            in_dim=num_classes * 16,
            image_pixels=image_size * image_size * in_channels,
        )

    # ── Helper ──────────────────────────────────────────────────────
    def _forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.acsf1(x)
        x = self.lbp(x)
        x = self.acsf2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    # ── Forward ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            x      : (B, C, H, W)
            labels : (B,) integer class indices, used for masked decoder.
                     Pass None during inference.
        Returns:
            class_probs  : (B, num_classes)  — capsule vector lengths (class scores).
            reconstructed: (B, C*H*W)        — reconstructed image.
            v            : (B, num_classes, 16) — raw class capsule vectors.
        """
        enc = self._forward_encoder(x)
        u   = self.primary_caps(enc)                        # (B, N_prim, 8)
        v   = self.class_caps(u)                            # (B, K, 16)

        class_probs = v.norm(dim=-1)                        # (B, K)

        reconstructed = self.decoder(v, labels)             # (B, C*H*W)

        return class_probs, reconstructed, v
