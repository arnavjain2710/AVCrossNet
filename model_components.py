"""
Model components for AVCrossNet architecture.
Contains vision encoders, SSM layers, and fusion modules.
"""
import math
from pathlib import Path
import numpy as np
import torch
try:
    import torchaudio
except ImportError:
    torchaudio = None  # Optional dependency, only used in Denoiser.denoise
from einops.layers.torch import EinMix
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from huggingface_hub import hf_hub_download
from torch.nn import MultiheadAttention
from timm.models.layers import trunc_normal_, DropPath
import librosa


# ============================================================================
# Vision Components
# ============================================================================

class LayerNorm(nn.Module):
    """Layer normalization supporting both channels_first and channels_last formats."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """Global Response Normalization layer."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, H, W = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y


class Block(nn.Module):
    """ConvNeXt Block with SE integration."""
    def __init__(self, dim, drop_path=0., reduction=16):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se_block = SEBlock(dim, reduction)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.se_block(x)
        x = input + self.drop_path(x)
        return x


class CustomConvNeXtV2(nn.Module):
    """Custom ConvNeXtV2 image encoder."""
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[48, 96, 192, 384],
                 drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class VideoFrameEncoder(nn.Module):
    """Video frame encoder that processes video frames using an image encoder."""
    def __init__(self, image_encoder, num_classes):
        super(VideoFrameEncoder, self).__init__()
        self.image_encoder = image_encoder(
            in_chans=1, num_classes=num_classes, 
            depths=[3, 3, 6, 3], dims=[48, 96, 192, 384], 
            drop_path_rate=0.
        )

    def forward(self, x):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        encoded_frames = self.image_encoder(x)
        encoded_frames = encoded_frames.view(B, N, -1)
        return encoded_frames


# ============================================================================
# SSM (State Space Model) Components
# ============================================================================

@torch.compiler.disable
def fft_conv(equation, input, kernel, *args):
    """FFT-based convolution operation."""
    input, kernel = input.float(), kernel.float()
    args = tuple(arg.cfloat() for arg in args)
    n = input.shape[-1]

    kernel_f = torch.fft.rfft(kernel, 2 * n)
    input_f = torch.fft.rfft(input, 2 * n)
    output_f = torch.einsum(equation, input_f, kernel_f, *args)
    output = torch.fft.irfft(output_f, 2 * n)

    return output[..., :n]


def ssm_basis_kernels(A, B, log_dt, length):
    """Compute SSM basis kernels."""
    log_A_real, A_imag = A.T
    lrange = torch.arange(length, device=A.device)
    dt = log_dt.exp()

    dtA_real, dtA_imag = -dt * F.softplus(log_A_real), dt * A_imag
    return (dtA_real[:, None] * lrange).exp() * torch.cos(dtA_imag[:, None] * lrange), B * dt[:, None]


def opt_ssm_forward(input, K, B_hat, C):
    """Optimized SSM forward pass with einsum contractions."""
    batch, c_in, _ = input.shape
    c_out, coeffs = C.shape

    if (1 / c_in + 1 / c_out) > (1 / batch + 1 / coeffs):
        if c_in * c_out <= coeffs:
            kernel = torch.einsum('dn,nc,nl->dcl', C, B_hat, K)
            return fft_conv('bcl,dcl->bdl', input, kernel)
    else:
        if coeffs <= c_in:
            x = torch.einsum('bcl,nc->bnl', input, B_hat)
            x = fft_conv('bnl,nl->bnl', x, K)
            return torch.einsum('bnl,dn->bdl', x, C)

    return fft_conv('bcl,nl,nc,dn->bdl', input, K, B_hat, C)


class SSMLayer(nn.Module):
    """State Space Model layer for audio processing."""
    def __init__(self, num_coeffs: int, in_channels: int, out_channels: int, repeat: int):
        from torch.backends import opt_einsum
        assert opt_einsum.is_available()
        opt_einsum.strategy = 'optimal'

        super().__init__()

        init_parameter = lambda mat: Parameter(torch.tensor(mat, dtype=torch.float))
        normal_parameter = lambda fan_in, shape: Parameter(torch.randn(*shape) * math.sqrt(2 / fan_in))

        A_real, A_imag = 0.5 * np.ones(num_coeffs), math.pi * np.arange(num_coeffs)
        log_A_real = np.log(np.exp(A_real) - 1)
        B = np.ones(num_coeffs)
        A = np.stack([log_A_real, A_imag], -1)
        log_dt = np.linspace(np.log(0.001), np.log(0.1), repeat)

        A = np.tile(A, (repeat, 1))
        B = np.tile(B[:, None], (repeat, in_channels)) / math.sqrt(in_channels)
        log_dt = np.repeat(log_dt, num_coeffs)

        self.log_dt, self.A, self.B = init_parameter(log_dt), init_parameter(A), init_parameter(B)
        self.C = normal_parameter(num_coeffs * repeat, (out_channels, num_coeffs * repeat))

    def forward(self, input):
        K, B_hat = ssm_basis_kernels(self.A, self.B, self.log_dt, input.shape[-1])
        return opt_ssm_forward(input, K, B_hat, self.C)


class LayerNormFeature(nn.Module):
    """Layer normalization for feature tensors."""
    def __init__(self, features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(features)

    def forward(self, input):
        return self.layer_norm(input.moveaxis(-1, -2)).moveaxis(-1, -2)


class Denoiser(nn.Module):
    """Audio denoiser with SSM-based encoder-decoder architecture."""
    def __init__(self, CCA, UDA, in_channels=1, channels=[16, 32, 64, 96, 128, 256],
                 num_coeffs=16, repeat=16, resample_factors=[4, 4, 2, 2, 2, 2], pre_conv=True):
        super().__init__()
        self.CCA = CCA
        self.UDA = UDA(CCA)
        depth = len(channels)
        self.depth = depth
        self.channels = [in_channels] + channels
        self.num_coeffs = num_coeffs
        self.repeat = repeat
        self.pre_conv = pre_conv
        self.mha = MultiheadAttention(embed_dim=100, num_heads=4, batch_first=True)
        self.down_linear = nn.Linear(100, 1)

        self.down_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=True) 
            for (c_in, c_out, r) in zip(self.channels[:-1], self.channels[1:], resample_factors)
        ])
        self.up_ssms = nn.ModuleList([
            self.ssm_pool(c_in, c_out, r, downsample=False) 
            for (c_in, c_out, r) in zip(self.channels[1:], self.channels[:-1], resample_factors)
        ])
        self.hid_ssms = nn.Sequential(
            self.ssm_block(self.channels[-1], True), 
            self.ssm_block(self.channels[-1], True),
        )
        self.last_ssms = nn.Sequential(
            self.ssm_block(self.channels[0], True), 
            self.ssm_block(self.channels[0], False),
        )

    def ssm_pool(self, in_channels, out_channels, resample_factor, downsample=True):
        if downsample:
            return nn.Sequential(
                self.ssm_block(in_channels, use_activation=True),
                EinMix('b c (t r) -> b d t', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_factor),
            )
        else:
            return nn.Sequential(
                EinMix('b c t -> b d (t r)', weight_shape='c d r', c=in_channels, d=out_channels, r=resample_factor),
                self.ssm_block(out_channels, use_activation=True),
            )

    def ssm_block(self, channels, use_activation=False):
        block = nn.Sequential()
        if channels > 1 and self.pre_conv:
            block.append(nn.Conv1d(channels, channels, 3, 1, 1, groups=channels))
        block.append(SSMLayer(self.num_coeffs, channels, channels, self.repeat))
        if use_activation:
            if channels > 1:
                block.append(LayerNormFeature(channels))
            block.append(nn.SiLU())
        return block

    def upscale_tensor(self, input_tensor):
        """Upscale tensor from (k, batch, 256) to (batch, 256*k, 1)."""
        batch = input_tensor.shape[1]
        reshaped_tensor = input_tensor.reshape(batch, 256 * input_tensor.shape[0], 1)
        return reshaped_tensor

    def downscale_tensor(self, x, k, batch):
        """Downscale tensor from (batch, len, 1) to (k, batch, 256)."""
        x = x.view(k, batch, 256)
        return x

    def forward(self, input, vf):
        x, skips = input, []

        # Encoder
        for ssm in self.down_ssms:
            skips.append(x)
            x = ssm(x)

        # Neck with UDA fusion
        x = x.permute(2, 0, 1)
        k, batch, _ = x.shape
        x = self.upscale_tensor(x)
        x = self.UDA(x, vf)
        x = self.downscale_tensor(x, k, batch)
        x = x.permute(1, 2, 0)
        x = self.hid_ssms(x)

        # Decoder
        for (ssm, skip) in zip(self.up_ssms[::-1], skips[::-1]):
            x = ssm[0](x)
            x = x + skip
            x = ssm[1](x)

        return self.last_ssms(x)

    def denoise_single(self, noisy):
        assert noisy.ndim == 2, f"noisy input should be shaped (samples, length)"
        noisy = noisy[:, None, :]
        padding = 256 - noisy.shape[-1] % 256
        noisy = F.pad(noisy, (0, padding))
        denoised = self.forward(noisy)
        return denoised.squeeze(1)[..., :-padding]

    def denoise_multiple(self, noisy_samples):
        audio_lens = [noisy.shape[-1] for noisy in noisy_samples]
        max_len = max(audio_lens)
        noisy_samples = torch.stack([F.pad(noisy, (0, max_len - noisy.shape[-1])) for noisy in noisy_samples])
        denoised_samples = self.denoise_single(noisy_samples)
        return [denoised[..., :audio_len] for (denoised, audio_len) in zip(denoised_samples, audio_lens)]

    def denoise(self, noisy_dir, denoised_dir=None):
        noisy_dir = Path(noisy_dir)
        denoised_dir = None if denoised_dir is None else Path(denoised_dir)

        noisy_files = [fn for fn in noisy_dir.glob('*.wav')]
        noisy_samples = [torch.tensor(librosa.load(wav_file, sr=16000)[0]) for wav_file in noisy_files]
        print("denoising...")
        denoised_samples = self.denoise_multiple(noisy_samples)

        if denoised_dir is not None:
            if torchaudio is None:
                raise ImportError("torchaudio is required for saving audio files. Install it with: pip install torchaudio")
            print("saving audio files...")
            for (denoised, noisy_fn) in zip(denoised_samples, noisy_files):
                torchaudio.save(denoised_dir / f"{noisy_fn.stem}.wav", denoised[None, :], 16000)

        return denoised_samples

    def from_pretrained(self, repo_id):
        print(f"loading weights from {repo_id}...")
        model_weights_path = hf_hub_download(repo_id=repo_id, filename="weights.pt")
        self.load_state_dict(torch.load(model_weights_path), strict=False)
        print("Denoiser loaded from pretrained!")


# ============================================================================
# Fusion Components
# ============================================================================

class CustomCrossAttention(nn.Module):
    """
    Cross-Modal Gated Attention with Probabilistic Fusion (CMGAPF).
    
    Implements probabilistic cross-attention mechanism for audio-visual fusion
    as specified in the CCA manual. Uses Gaussian distributions to model feature
    uncertainty and gated fusion for adaptive feature combination.
    
    The attention mechanism uses learnable linear projections for Query (Q) and Key (K):
    - Q_a, K_v for audio-to-visual attention
    - Q_v, K_a for visual-to-audio attention
    
    The attention scores are weighted by Gaussian PDF: N(feature; μ, Σ) to incorporate
    probabilistic uncertainty in the cross-modal fusion.
    
    Args:
        da: Dimensionality of audio features
        dv: Dimensionality of visual features
        d_model: Common dimension for Q/K projections (default: max(da, dv))
    """
    def __init__(self, da, dv, d_model=None, use_diagonal_cov=False, return_mean=False):
        """
        Initialize CMGAPF module.
        
        Args:
            da: Dimensionality of audio features
            dv: Dimensionality of visual features
            d_model: Common dimension for Q/K projections (default: max(da, dv))
            use_diagonal_cov: If True, use diagonal covariance approximation for efficiency.
                            If False, use full covariance (more accurate but computationally expensive).
            return_mean: If True, return mean μ_f instead of sampling from N(μ_f, Σ_f).
                       Useful for deterministic behavior in eval mode.
        """
        super(CustomCrossAttention, self).__init__()
        self.da = da
        self.dv = dv
        self.d_model = d_model if d_model is not None else max(da, dv)
        self.use_diagonal_cov = use_diagonal_cov
        self.return_mean = return_mean
        
        # Learnable Query and Key projections for attention
        # Audio-to-Visual attention: Q from audio, K from visual
        self.Q_a = nn.Linear(da, self.d_model)  # Query projection from audio
        self.K_v = nn.Linear(dv, self.d_model)  # Key projection from visual
        self.V_v = nn.Linear(dv, dv)  # Value projection from visual (optional, can use V directly)
        
        # Visual-to-Audio attention: Q from visual, K from audio
        self.Q_v = nn.Linear(dv, self.d_model)  # Query projection from visual
        self.K_a = nn.Linear(da, self.d_model)  # Key projection from audio
        self.V_a = nn.Linear(da, da)  # Value projection from audio (optional, can use A directly)
        
        # Gate parameters for audio and visual features
        self.W_A_g = nn.Parameter(torch.randn(1, da + dv))
        self.W_V_g = nn.Parameter(torch.randn(1, dv + da))
        self.b_A_g = nn.Parameter(torch.randn(1))
        self.b_V_g = nn.Parameter(torch.randn(1))
        
        # Projection layers for dimension mismatch in fusion
        if da != dv:
            self.v_to_a_proj = nn.Linear(dv, da)  # Projects visual to audio dimension
            self.a_to_v_proj = nn.Linear(da, dv)  # Projects audio to visual dimension
        else:
            self.v_to_a_proj = nn.Identity()
            self.a_to_v_proj = nn.Identity()

    def showgrads(self):
        """Debug utility to print gradients."""
        print(self.W_A_g.grad)
        print(self.W_V_g.grad)
        print(self.b_A_g.grad)
        print(self.b_V_g.grad)

    def _sample_from_gaussian(self, mu, Sigma, device):
        """
        Sample from multivariate Gaussian N(μ, Σ) using reparameterization trick.
        
        Uses reparameterization: x = μ + L @ ε, where L is Cholesky factor of Σ
        and ε ~ N(0, I). This makes the sampling differentiable.
        
        Args:
            mu: Mean tensor of shape (batch, T, d) or (batch, d)
            Sigma: Covariance tensor of shape (batch, T, d, d), (batch, d, d), or (d, d)
            device: Device to perform computation on
            
        Returns:
            samples: Sampled features of shape (batch, T, d)
        """
        if mu.dim() == 2:
            # (batch, d) -> expand to (batch, 1, d) for consistency
            mu = mu.unsqueeze(1)
            expand_time = True
        else:
            expand_time = False
        
        batch_size, T, d = mu.shape
        
        # Handle different Sigma shapes
        if Sigma.dim() == 2:
            # Single covariance matrix (d, d) -> expand to (batch, T, d, d)
            Sigma = Sigma.unsqueeze(0).unsqueeze(0).expand(batch_size, T, -1, -1)
        elif Sigma.dim() == 3:
            # (batch, d, d) -> expand to (batch, T, d, d)
            Sigma = Sigma.unsqueeze(1).expand(-1, T, -1, -1)
        # If Sigma.dim() == 4, it's already (batch, T, d, d) - no expansion needed
        
        eps_reg = 1e-6
        eye = torch.eye(d, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, T, -1, -1)
        Sigma_reg = Sigma + eye * eps_reg
        
        if self.use_diagonal_cov:
            # Diagonal approximation: much faster for large dimensions
            # Extract diagonal: (batch, T, d)
            diag_Sigma = torch.diagonal(Sigma_reg, dim1=-2, dim2=-1)  # (batch, T, d)
            std = torch.sqrt(torch.clamp(diag_Sigma, min=eps_reg))  # (batch, T, d)
            
            # Sample: x = μ + std * ε, where ε ~ N(0, I)
            eps = torch.randn_like(mu)  # (batch, T, d)
            samples = mu + std * eps  # (batch, T, d)
        else:
            # Full covariance: use Cholesky decomposition
            # Reshape for batch Cholesky: (batch*T, d, d)
            batch_T = batch_size * T
            Sigma_flat = Sigma_reg.view(batch_T, d, d)  # (batch*T, d, d)
            
            try:
                # Cholesky: L such that L @ L^T = Σ
                L_flat = torch.linalg.cholesky(Sigma_flat)  # (batch*T, d, d)
                L = L_flat.view(batch_size, T, d, d)  # (batch, T, d, d)
                
                # Sample: x = μ + L @ ε, where ε ~ N(0, I)
                eps = torch.randn(batch_size, T, d, 1, device=device)  # (batch, T, d, 1)
                L_eps = torch.matmul(L, eps).squeeze(-1)  # (batch, T, d)
                samples = mu + L_eps  # (batch, T, d)
            except RuntimeError:
                # Fallback to diagonal if Cholesky fails (e.g., non-positive definite)
                diag_Sigma = torch.diagonal(Sigma_reg, dim1=-2, dim2=-1)  # (batch, T, d)
                std = torch.sqrt(torch.clamp(diag_Sigma, min=eps_reg))  # (batch, T, d)
                eps = torch.randn_like(mu)  # (batch, T, d)
                samples = mu + std * eps  # (batch, T, d)
        
        if expand_time:
            samples = samples.squeeze(1)  # (batch, d)
        
        return samples

    def _compute_gaussian_pdf_weights(self, features, mu, Sigma, device):
        """
        Compute Gaussian probability density function weights for features.
        
        Vectorized implementation for GPU efficiency. Computes Gaussian PDF
        for all batch elements simultaneously.
        
        Args:
            features: Tensor of shape (batch, T, d) - features to weight
            mu: Tensor of shape (batch, d) - mean of Gaussian
            Sigma: Tensor of shape (batch, d, d) - covariance of Gaussian
            device: Device to perform computation on
            
        Returns:
            weights: Tensor of shape (batch, T) - Gaussian PDF weights
        """
        batch_size, T, d = features.shape
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        eye = torch.eye(d, device=device).unsqueeze(0).expand(batch_size, -1, -1)  # (batch, d, d)
        Sigma_reg = Sigma + eye * eps
        
        # Compute difference: (batch, T, d) - (batch, 1, d) -> (batch, T, d)
        diff = features - mu.unsqueeze(1)  # (batch, T, d)
        
        # Vectorized Cholesky decomposition for all batches
        try:
            # Cholesky decomposition: (batch, d, d)
            L = torch.linalg.cholesky(Sigma_reg)  # (batch, d, d)
            
            # Log determinant: log|Σ| = 2 * sum(log(diag(L)))
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # (batch,)
            
            # Solve: (x - μ)^T Σ^(-1) (x - μ) for all batches and time steps
            # L @ y = diff^T, where diff^T is (batch, d, T)
            diff_T = diff.transpose(1, 2)  # (batch, d, T)
            y = torch.linalg.solve_triangular(L, diff_T, upper=False)  # (batch, d, T)
            
            # Mahalanobis distance: sum of squares along d dimension
            mahalanobis = torch.sum(y ** 2, dim=1)  # (batch, T)
            
            # Log PDF: -0.5 * (d*log(2π) + log|Σ| + Mahalanobis^2)
            log_det_expanded = log_det.unsqueeze(1)  # (batch, 1)
            # Clamp mahalanobis to prevent extreme values
            mahalanobis = torch.clamp(mahalanobis, max=100.0)
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det_expanded + mahalanobis)  # (batch, T)
            # Clamp log_prob to prevent underflow
            log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
            weights = torch.exp(log_prob)  # (batch, T)
            
        except RuntimeError:
            # Fallback: use diagonal approximation if Cholesky fails
            # This is still vectorized across batches
            diag_Sigma = torch.diagonal(Sigma_reg, dim1=-2, dim2=-1)  # (batch, d)
            diag_Sigma = torch.clamp(diag_Sigma, min=eps)  # Ensure positive
            diag_Sigma_expanded = diag_Sigma.unsqueeze(1)  # (batch, 1, d)
            
            # Mahalanobis with diagonal approximation
            mahalanobis = torch.sum((diff ** 2) / diag_Sigma_expanded, dim=2)  # (batch, T)
            mahalanobis = torch.clamp(mahalanobis, max=100.0)
            
            # Log determinant: sum of log of diagonal elements
            log_det = torch.sum(torch.log(diag_Sigma + eps), dim=-1)  # (batch,)
            log_det_expanded = log_det.unsqueeze(1)  # (batch, 1)
            
            # Log PDF
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det_expanded + mahalanobis)  # (batch, T)
            log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
            weights = torch.exp(log_prob)  # (batch, T)
        
        # Replace any NaN or Inf with small positive value
        weights = torch.where(torch.isnan(weights) | torch.isinf(weights), 
                              torch.ones_like(weights) * 1e-8, weights)
        # Normalize to prevent extreme values
        weights = torch.clamp(weights, min=1e-10, max=1e10)
        
        return weights

    def forward(self, A, V):
        """
        Forward pass implementing CMGAPF.
        
        Args:
            A: Audio features of shape (batch, T, da)
            V: Visual features of shape (batch, T, dv)
            
        Returns:
            combined_features: Combined features of shape (batch, T, d_f)
                where d_f is the output dimension (currently returns mean only)
        """
        device = A.device
        batch_size, T, da = A.shape
        _, _, dv = V.shape

        # ========================================================================
        # Step 1: Compute Gaussian statistics for visual features
        # ========================================================================
        # Mean: μ_v ∈ R^(batch, dv)
        mu_v = torch.mean(V, dim=1)  # (batch, dv)
        
        # Covariance: Σ_v ∈ R^(batch, dv, dv)
        # Compute covariance for each batch element with better numerical stability
        eps_reg = 1e-6
        Sigma_v_list = []
        for b in range(batch_size):
            V_b = V[b]  # (T, dv)
            V_b_centered = V_b - V_b.mean(dim=0, keepdim=True)
            if T > 1:
                Sigma_v_b = (V_b_centered.T @ V_b_centered) / (T - 1)  # (dv, dv)
            else:
                Sigma_v_b = torch.eye(dv, device=device) * eps_reg
            Sigma_v_b = Sigma_v_b + torch.eye(dv, device=device) * eps_reg
            # Replace NaN/Inf
            nan_mask = torch.isnan(Sigma_v_b) | torch.isinf(Sigma_v_b)
            if nan_mask.any():
                Sigma_v_b = torch.where(nan_mask, torch.eye(dv, device=device) * eps_reg, Sigma_v_b)
            Sigma_v_list.append(Sigma_v_b)
        Sigma_v = torch.stack(Sigma_v_list)  # (batch, dv, dv)

        # ========================================================================
        # Step 2: Audio-to-Visual Probabilistic Cross-Attention
        # Attention_{A→V}(a_t) = (1/Z_a) * Σ_j N(v_j; μ_v, Σ_v) * Softmax(Q_a K_v^T / √d)
        # ========================================================================
        # Learnable Query projection from audio features
        Q_a = self.Q_a(A)  # (batch, T, d_model) - learned projection from audio
        
        # Learnable Key projection from visual features
        K_v = self.K_v(V)  # (batch, T, d_model) - learned projection from visual
        
        # Compute scaled dot-product attention scores
        # Q_a @ K_v^T: (batch, T, d_model) @ (batch, d_model, T) -> (batch, T, T)
        scaled_dot_product_av = torch.matmul(Q_a, K_v.transpose(1, 2)) / np.sqrt(self.d_model)
        
        # Compute Gaussian PDF weights for visual features
        # These weights incorporate the probabilistic nature: N(v_j; μ_v, Σ_v)
        gaussian_weights_v = self._compute_gaussian_pdf_weights(V, mu_v, Sigma_v, device)  # (batch, T)
        
        # Apply Gaussian weighting to attention scores
        # Formula: Attention = (1/Z) * Σ N(v_j; μ_v, Σ_v) * Softmax(QK^T / √d)
        # We multiply Gaussian PDF weights with attention scores before softmax
        # Broadcast: (batch, T, T) * (batch, 1, T) -> (batch, T, T)
        attention_scores_av = scaled_dot_product_av * gaussian_weights_v.unsqueeze(1)
        attention_weights_av = F.softmax(attention_scores_av, dim=-1)  # (batch, T, T)
        
        # Weighted visual values (can use V directly or projected V)
        V_proj = self.V_v(V)  # (batch, T, dv) - optional value projection
        weighted_visual_values_av = torch.matmul(attention_weights_av, V_proj)  # (batch, T, dv)
        
        # Normalization factor Z_a (sum of attention weights)
        Z_a = torch.sum(attention_weights_av, dim=-1, keepdim=True) + 1e-8  # (batch, T, 1)
        attention_result_av = weighted_visual_values_av / Z_a  # (batch, T, dv)

        # ========================================================================
        # Step 3: Compute Gaussian statistics for audio features
        # ========================================================================
        # Mean: μ_a ∈ R^(batch, da)
        mu_a = torch.mean(A, dim=1)  # (batch, da)
        
        # Covariance: Σ_a ∈ R^(batch, da, da)
        Sigma_a_list = []
        for b in range(batch_size):
            A_b = A[b]  # (T, da)
            A_b_centered = A_b - A_b.mean(dim=0, keepdim=True)
            if T > 1:
                Sigma_a_b = (A_b_centered.T @ A_b_centered) / (T - 1)  # (da, da)
            else:
                Sigma_a_b = torch.eye(da, device=device) * eps_reg
            Sigma_a_b = Sigma_a_b + torch.eye(da, device=device) * eps_reg
            # Replace NaN/Inf
            nan_mask = torch.isnan(Sigma_a_b) | torch.isinf(Sigma_a_b)
            if nan_mask.any():
                Sigma_a_b = torch.where(nan_mask, torch.eye(da, device=device) * eps_reg, Sigma_a_b)
            Sigma_a_list.append(Sigma_a_b)
        Sigma_a = torch.stack(Sigma_a_list)  # (batch, da, da)

        # ========================================================================
        # Step 4: Visual-to-Audio Probabilistic Cross-Attention
        # Attention_{V→A}(v_t) = (1/Z_v) * Σ_j N(a_j; μ_a, Σ_a) * Softmax(Q_v K_a^T / √d)
        # ========================================================================
        # Learnable Query projection from visual features
        Q_v = self.Q_v(V)  # (batch, T, d_model) - learned projection from visual
        
        # Learnable Key projection from audio features
        K_a = self.K_a(A)  # (batch, T, d_model) - learned projection from audio
        
        # Compute scaled dot-product attention scores
        # Q_v @ K_a^T: (batch, T, d_model) @ (batch, d_model, T) -> (batch, T, T)
        scaled_dot_product_va = torch.matmul(Q_v, K_a.transpose(1, 2)) / np.sqrt(self.d_model)
        
        # Compute Gaussian PDF weights for audio features
        # These weights incorporate the probabilistic nature: N(a_j; μ_a, Σ_a)
        gaussian_weights_a = self._compute_gaussian_pdf_weights(A, mu_a, Sigma_a, device)  # (batch, T)
        
        # Apply Gaussian weighting to attention scores
        # Formula: Attention = (1/Z) * Σ N(a_j; μ_a, Σ_a) * Softmax(QK^T / √d)
        attention_scores_va = scaled_dot_product_va * gaussian_weights_a.unsqueeze(1)
        attention_weights_va = F.softmax(attention_scores_va, dim=-1)  # (batch, T, T)
        
        # Weighted audio values (can use A directly or projected A)
        A_proj = self.V_a(A)  # (batch, T, da) - optional value projection
        weighted_audio_values_va = torch.matmul(attention_weights_va, A_proj)  # (batch, T, da)
        
        # Normalization factor Z_v
        Z_v = torch.sum(attention_weights_va, dim=-1, keepdim=True) + 1e-8  # (batch, T, 1)
        attention_result_va = weighted_audio_values_va / Z_v  # (batch, T, da)

        # ========================================================================
        # Step 5: Gated Cross-Attention Fusion (Vectorized)
        # g^A_t = σ(W^A_g · [a_t; v^att_t] + b^A_g)
        # ã_t = g^A_t · a_t + (1 - g^A_t) · v^att_t
        # ========================================================================
        # Vectorized gated fusion across all batches and time steps
        
        # Compute gates: g^A_t and g^V_t for all (batch, T)
        # Concatenate original features with attention results (before projection)
        # [a_t; v^att_t] -> (batch, T, da + dv)
        concat_a = torch.cat((A, attention_result_av), dim=2)  # (batch, T, da + dv)
        # Apply gate: W_A_g @ concat_a + b_A_g
        # W_A_g: (1, da + dv), concat_a: (batch, T, da + dv)
        g_A = torch.sigmoid(
            torch.einsum('ij,btj->bti', self.W_A_g, concat_a) + self.b_A_g
        )  # (batch, T, 1)
        
        # Concatenate: [v_t; a^att_t] -> (batch, T, dv + da)
        concat_v = torch.cat((V, attention_result_va), dim=2)  # (batch, T, dv + da)
        # Apply gate: W_V_g @ concat_v + b_V_g
        g_V = torch.sigmoid(
            torch.einsum('ij,btj->bti', self.W_V_g, concat_v) + self.b_V_g
        )  # (batch, T, 1)
        
        # Project attention results for dimension compatibility in fusion
        v_t_att_proj = self.v_to_a_proj(attention_result_av)  # (batch, T, da)
        a_t_att_proj = self.a_to_v_proj(attention_result_va)  # (batch, T, dv)
        
        # Fused features: ã_t and ṽ_t (vectorized)
        a_fused = g_A * A + (1 - g_A) * v_t_att_proj  # (batch, T, da)
        v_fused = g_V * V + (1 - g_V) * a_t_att_proj  # (batch, T, dv)
        
        # ========================================================================
        # Step 6: Compute Covariance for Fused Features
        # Σ_ã_t and Σ_ṽ_t (per batch)
        # ========================================================================
        Sigma_a_fused_list = []
        Sigma_v_fused_list = []
        
        for b in range(batch_size):
            # Compute covariance for fused audio features
            a_fused_b = a_fused[b]  # (T, da)
            # Add small noise to prevent singular covariance
            a_fused_b = a_fused_b + torch.randn_like(a_fused_b) * 1e-5
            
            # Compute covariance with better numerical stability
            # Center the data
            a_fused_b_centered = a_fused_b - a_fused_b.mean(dim=0, keepdim=True)
            # Compute covariance manually for better control
            if T > 1:
                Sigma_a_fused = (a_fused_b_centered.T @ a_fused_b_centered) / (T - 1)  # (da, da)
            else:
                # If T == 1, use identity
                Sigma_a_fused = torch.eye(da, device=device) * eps_reg
            
            # Ensure positive definite by adding regularization
            Sigma_a_fused = Sigma_a_fused + torch.eye(da, device=device) * eps_reg
            # Replace any NaN or Inf with identity
            nan_mask = torch.isnan(Sigma_a_fused) | torch.isinf(Sigma_a_fused)
            if nan_mask.any():
                Sigma_a_fused = torch.where(nan_mask, torch.eye(da, device=device) * eps_reg, Sigma_a_fused)
            # Clamp diagonal to prevent negative values
            diag = torch.diagonal(Sigma_a_fused)
            diag = torch.clamp(diag, min=eps_reg)
            Sigma_a_fused = Sigma_a_fused - torch.diag(torch.diagonal(Sigma_a_fused)) + torch.diag(diag)
            Sigma_a_fused_list.append(Sigma_a_fused)
            
            # Compute covariance for fused visual features
            v_fused_b = v_fused[b]  # (T, dv)
            # Add small noise to prevent singular covariance
            v_fused_b = v_fused_b + torch.randn_like(v_fused_b) * 1e-5
            
            # Center the data
            v_fused_b_centered = v_fused_b - v_fused_b.mean(dim=0, keepdim=True)
            # Compute covariance
            if T > 1:
                Sigma_v_fused = (v_fused_b_centered.T @ v_fused_b_centered) / (T - 1)  # (dv, dv)
            else:
                Sigma_v_fused = torch.eye(dv, device=device) * eps_reg
            
            # Ensure positive definite
            Sigma_v_fused = Sigma_v_fused + torch.eye(dv, device=device) * eps_reg
            # Replace any NaN or Inf
            nan_mask = torch.isnan(Sigma_v_fused) | torch.isinf(Sigma_v_fused)
            if nan_mask.any():
                Sigma_v_fused = torch.where(nan_mask, torch.eye(dv, device=device) * eps_reg, Sigma_v_fused)
            # Clamp diagonal
            diag = torch.diagonal(Sigma_v_fused)
            diag = torch.clamp(diag, min=eps_reg)
            Sigma_v_fused = Sigma_v_fused - torch.diag(torch.diagonal(Sigma_v_fused)) + torch.diag(diag)
            Sigma_v_fused_list.append(Sigma_v_fused)
        
        Sigma_a_fused = torch.stack(Sigma_a_fused_list)  # (batch, da, da)
        Sigma_v_fused = torch.stack(Sigma_v_fused_list)  # (batch, dv, dv)
        
        # ========================================================================
        # Step 7: Statistical Fusion with Gaussian Sampling
        # μ_f = α_t · ã_t + β_t · ṽ_t
        # Σ_f = α_t² · Σ_ã_t + β_t² · Σ_ṽ_t
        # Sample: f_t ~ N(μ_f, Σ_f)
        # ========================================================================
        # Compute cosine similarity for all (batch, T)
        # Normalize A and V for cosine similarity (add eps for numerical stability)
        eps_norm = 1e-8
        A_norm = F.normalize(A, p=2, dim=2)  # (batch, T, da)
        V_norm = F.normalize(V, p=2, dim=2)  # (batch, T, dv)
        
        if da == dv:
            # Direct cosine similarity
            S_at_vt = F.cosine_similarity(A_norm, V_norm, dim=2)  # (batch, T)
        else:
            # For different dimensions, project to common dimension
            min_dim = min(da, dv)
            S_at_vt = torch.sum(A_norm[:, :, :min_dim] * V_norm[:, :, :min_dim], dim=2)  # (batch, T)
        
        # Clamp to prevent extreme values
        S_at_vt = torch.clamp(S_at_vt, min=-1.0, max=1.0)
        
        # Influence coefficients: α_t and β_t (vectorized)
        exp_S = torch.exp(S_at_vt)  # (batch, T)
        alpha = exp_S / (exp_S + 1)  # (batch, T)
        beta = 1 / (exp_S + 1)  # (batch, T)
        
        # Project v_fused to da dimension for combination
        v_fused_proj = self.v_to_a_proj(v_fused)  # (batch, T, da)
        
        # Combined feature mean: μ_f = α_t · ã_t + β_t · ṽ_t
        alpha_expanded = alpha.unsqueeze(2)  # (batch, T, 1)
        beta_expanded = beta.unsqueeze(2)  # (batch, T, 1)
        mu_f = alpha_expanded * a_fused + beta_expanded * v_fused_proj  # (batch, T, da)
        
        # Combined feature covariance: Σ_f = α_t² · Σ_ã_t + β_t² · Σ_ṽ_t
        # Project Sigma_v_fused to da dimension if needed
        if da != dv:
            # For covariance projection, we use a simplified approach
            # Project each row/column of covariance matrix
            # This is an approximation - full projection would be more complex
            Sigma_v_fused_proj = torch.zeros(batch_size, da, da, device=device)
            for b in range(batch_size):
                # Use the projection layer's weight to transform covariance
                # Approximation: project both dimensions
                proj_weight = self.v_to_a_proj.weight  # (da, dv)
                Sigma_v_fused_proj[b] = proj_weight @ Sigma_v_fused[b] @ proj_weight.T
            Sigma_v_fused_proj = Sigma_v_fused_proj + torch.eye(da, device=device).unsqueeze(0) * eps_reg
        else:
            Sigma_v_fused_proj = Sigma_v_fused  # (batch, da, da)
        
        # Compute Σ_f for each (batch, t)
        # α_t² · Σ_ã_t + β_t² · Σ_ṽ_t
        alpha_sq = alpha ** 2  # (batch, T)
        beta_sq = beta ** 2  # (batch, T)
        
        # Expand for broadcasting: (batch, T, 1, 1) for covariance matrices
        alpha_sq_expanded = alpha_sq.unsqueeze(2).unsqueeze(3)  # (batch, T, 1, 1)
        beta_sq_expanded = beta_sq.unsqueeze(2).unsqueeze(3)  # (batch, T, 1, 1)
        
        # Expand covariance matrices: (batch, 1, da, da)
        Sigma_a_expanded = Sigma_a_fused.unsqueeze(1)  # (batch, 1, da, da)
        Sigma_v_expanded = Sigma_v_fused_proj.unsqueeze(1)  # (batch, 1, da, da)
        
        # Compute Σ_f: (batch, T, da, da)
        Sigma_f = alpha_sq_expanded * Sigma_a_expanded + beta_sq_expanded * Sigma_v_expanded
        
        # ========================================================================
        # Step 8: Sample from Gaussian N(μ_f, Σ_f) or return mean
        # ========================================================================
        if self.return_mean or not self.training:
            # Return mean for deterministic behavior (useful in eval mode)
            fused_features = mu_f  # (batch, T, da)
        else:
            # Sample fused features from the Gaussian distribution
            # This uses the reparameterization trick for differentiability
            fused_features = self._sample_from_gaussian(mu_f, Sigma_f, device)  # (batch, T, da)
        
        # Check for NaN/Inf and replace with mean if needed
        nan_mask = torch.isnan(fused_features) | torch.isinf(fused_features)
        if nan_mask.any():
            fused_features = torch.where(nan_mask, mu_f, fused_features)
        
        return fused_features


class UDA(nn.Module):
    """Unified Domain Adaptation module for audio-visual alignment."""
    def __init__(self, blackbox):
        super(UDA, self).__init__()
        self.blackbox = blackbox(100, 100)

    def upsample(self, input_tensor, len_audio, len_video):
        """
        Upsamples the input tensor from shape (batch, len_video, 100) to (batch, len_audio, 1).
        """
        reshaped_tensor = input_tensor.view(-1, 100 * len_video)
        upsampled_tensor = F.interpolate(reshaped_tensor.unsqueeze(1), size=(len_audio,), mode='linear', align_corners=False)
        final_output = upsampled_tensor.unsqueeze(-1)
        return final_output[:, 0, ...]

    def downsample(self, input_tensor, len_audio, len_video):
        """
        Downsamples the input tensor from shape (batch, len_audio, 1) to (batch, len_video, 100).
        """
        input_tensor = input_tensor.squeeze(-1)
        resized_tensor = F.interpolate(input_tensor.unsqueeze(1), size=(100 * len_video,), mode='linear', align_corners=False)
        reshaped_tensor = resized_tensor.view(-1, len_video, 100)
        return reshaped_tensor

    def forward(self, input_vec, input_vec_small):
        B, L, _ = input_vec.shape
        B, vf, _ = input_vec_small.shape
        downscaled_input = self.downsample(input_vec, L, vf)
        blackbox_output = self.blackbox(downscaled_input, input_vec_small)
        final_output = input_vec + self.upsample(blackbox_output, L, vf)
        return final_output

