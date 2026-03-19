from collections import OrderedDict
from pathlib import Path
from warnings import warn
import torch
import torch.nn as nn

import deepinv as dinv
from deepinv.physics import LinearPhysicsMultiScaler, PhysicsCropper
from deepinv.utils.tensorlist import TensorList
from deepinv.models.base import Reconstructor, Denoiser
from deepinv.models.utils import LayerNorm_AF2d

import timm
import math

def modulate(x, scale=None, shift=None, gate=None):
    if scale is not None:
        x = x * scale
    if shift is not None:
        x = x + shift
    if gate is not None:
        x = x * gate
    return x

class CSEmbedding(nn.Module):
    # old kernel size was 7 (not parametrizable for the moment, k=3 breaks dependency with first checkpoints)
    # embed dim was not set as kwarg, default to 64 for low complexity
    def __init__(self, z_dim: int, embed_dim: int=64, kernel_size=3, activation: str = "SiLU", deep_embedding: bool=False):
        super(CSEmbedding, self).__init__()
        
        self.deep_embedding = deep_embedding
        
        D = z_dim * 128**2 # FIXME
        
        if not self.deep_embedding:
            # b = 2*torch.pi * torch.rand((embed_dim,), generator=torch.Generator().manual_seed(1000))
            # self.register_buffer('b', b, persistent=False)
            W = torch.randn(embed_dim, D, generator=torch.Generator().manual_seed(1000)) # * math.sqrt(2 * gamma)
            W = W / math.sqrt(embed_dim) # normalize by output dimension for stability
            self.register_buffer('W', W, persistent=False)
            
            self.linear1 = torch.nn.Linear(embed_dim, embed_dim)
            self.act = torch.nn.SiLU()
            self.linear2 = torch.nn.Linear(embed_dim, embed_dim)
            
            # self.linear1 = nn.Conv2d(z_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
            # self.act = getattr(nn, activation)()
            # self.linear2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size//2)
            
        else:
            self.net = timm.create_model('resnet10t', in_chans=z_dim, pretrained=False)
            self.linear = nn.Conv2d(512, embed_dim, kernel_size=1, bias=False)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.deep_embedding:
            z = self.net.forward_features(z) # returns [B,512,4,4] for 128x128 inputs
            z = torch.mean(z, dim=[2,3], keepdim=True) # global average pooling to get [B,512,1,1]
            z = self.linear(z) # project to embed_dim, get [B,embed_dim,1,1]
        else:
            z = z.flatten(start_dim=1) #@ self.W
            z = torch.einsum('BD,dD->Bd', z, self.W) # [B, d]
            # z = z + self.b
            z = self.act(z)
            
            z = self.linear1(z)
            z = self.act(z)
            z = self.linear2(z)
        return z

class RMSNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(RMSNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
    
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=(-3,-2,-1), keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return (self.weight.view(1, -1, 1, 1) * x).to(input_dtype)

class LayerEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = False, use_scale_shift_norm: bool=False):
        super(LayerEmbedding, self).__init__()
        
        self.use_scale_shift_norm = use_scale_shift_norm
        out_channels = 2 * out_channels if use_scale_shift_norm else out_channels
        
        self.linear_in = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.linear1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.act1 = nn.ReLU(inplace=False)

        self.linear2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias)
        self.act2 = nn.ReLU(inplace=False)
     
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_in = self.linear_in(z)
        skip = self.skip(x_in)
        x = self.linear1(x_in)
        x = self.act1(x)
        x = self.linear2(x) + skip
        x = self.act2(x)
        return x

class RAMv2(Reconstructor, Denoiser):
    r"""
    Reconstruct Anything Model (RAM) foundation model.

    RAM :footcite:t:`terris2025reconstruct` is a convolutional neural network model that has been trained to work on a large variety
    of linear image reconstruction tasks and datasets (deblurring, inpainting, denoising, tomography, MRI, etc.).

    See :ref:`sphx_glr_auto_examples_models_demo_foundation_model.py` for examples on the performance of RAM and how to fine-tune the
    foundation model on a specific problem and dataset.

    The model works both as a reconstructor or denoiser:

    * Reconstructor: RAM takes a :ref:`physics operator <physics>` `model(y, physics)` with an optional noise model defined in the physics
    * Denoiser: RAM takes optional Gaussian and/or Poisson noise levels (optionally set to 0) `model(y, sigma=sigma, gamma=gamma)`

    .. note::

        The physics operator should be normalized (i.e. have unit norm) for best results.
        Use :func:`physics.compute_norm() <deepinv.physics.LinearPhysics.compute_norm>` to check this.

    :param Sequence in_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param str device: Device to which the model should be moved. If None, the model will be created on the default device.
    :param bool, str pretrained: If `True`, the model will be initialized with pretrained weights. If `str`, load from file.
    :param float sigma_threshold: Threshold (minimum value) for the noise level. Default is 1e-3.

    |sep|

      >>> import deepinv as dinv
      >>> x = dinv.utils.load_example("butterfly.png")
      >>> physics = dinv.physics.Downsampling(filter="bicubic")
      >>> y = physics(x)
      >>> model = dinv.models.RAM() # doctest:+ELLIPSIS
      ...
      >>> x_hat = model(y, physics) # run model
      >>> dinv.metric.PSNR()(x_hat, x) > 31.98
      tensor([True])

    """

    def __init__(
        self,
        in_channels=(1, 2, 3),
        device=None,
        pretrained=True,
        normalization=False,
        encoder_conditioning=True, # if False measurement conditioning is removed from the encoder parts
        decoder_conditioning=True,
        first_only_conditioning=False,
        first_only_conditioning_list=[],
        last_only_conditioning=False, # if True, only the last decoder block has measurement conditioning,
        last_only_conditioning_list=[],
        conditioning_dim=0,
        z_embedding_dim=0,
        use_scale_shift_gate_norm=False,
        use_scale_shift_norm=False,
        use_scale_norm=False,
        channel_only_conditioning=False, # if True, only use channel-wise conditioning (no spatial conditioning), only for the measurement conditioning part
        cross_attention_conditioning=False, # if True, use cross attention for the conditioning instead of concatenation (only for the measurement conditioning part)
        deep_embedding=False, # if True, use a deeper embedding network for the conditioning (only for the measurement conditioning part)
        krylov_degree=2,
        forward_iterates=False,
        nc: tuple[int,int,int,int] = (64, 128, 256, 512),
        nb: int = 4
    ):
        super(RAMv2, self).__init__()

        # custom additions
        self.is_normalization = normalization
        self.encoder_conditioning = encoder_conditioning
        self.decoder_conditioning = decoder_conditioning
        
        self.first_only_conditioning = first_only_conditioning
        self.first_only_conditioning_list = first_only_conditioning_list if len(first_only_conditioning_list) > 0 \
            else ([first_only_conditioning] * 4 if first_only_conditioning else None)
        
        self.last_only_conditioning = last_only_conditioning
        self.last_only_conditioning_list = last_only_conditioning_list if len(last_only_conditioning_list) > 0 \
            else ([last_only_conditioning] * 4 if last_only_conditioning else None)
            
        self.conditioning_dim = conditioning_dim if z_embedding_dim == 0 else 0
        self.z_embedding_dim = z_embedding_dim
        assert not use_scale_shift_norm or not use_scale_norm, "use_scale_shift_norm and use_scale_norm cannot be both True."
        self.use_scale_shift_gate_norm = use_scale_shift_gate_norm
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_scale_norm = use_scale_norm
        self.channel_only_conditioning = channel_only_conditioning
        self.cross_attention_conditioning = cross_attention_conditioning
        self.deep_embedding = deep_embedding
        self.krylov_degree = krylov_degree
        self.forward_iterates = forward_iterates
        
        assert not (self.first_only_conditioning and self.last_only_conditioning), "first_only_conditioning and last_only_conditioning cannot be both True."
        
        # nc = [64, 128, 256, 512]  # number of channels in the network
        self.nc = nc
        self.nb = nb
        self.in_channels = in_channels
        self.fact_realign = torch.nn.Parameter(torch.tensor([1.0], device=device))

        self.separate_head = isinstance(in_channels, list)

        if isinstance(in_channels, tuple):
            in_channels = list(in_channels)

        if isinstance(in_channels, list):
            in_channels_first = []
            for i in range(len(in_channels)):
                if self.z_embedding_dim == 0:
                    in_channels_first.append(in_channels[i] + 2 + conditioning_dim)
                else:
                    in_channels_first.append(in_channels[i] + 2)

        self.cs_embedding = CSEmbedding(
            z_dim=conditioning_dim,
            embed_dim=self.z_embedding_dim,
            activation="ReLU",
            deep_embedding=self.deep_embedding
        ) if self.z_embedding_dim > 0 else None

        # check if in_channels is a list
        self.m_head = InHead(in_channels_first, nc[0])

        self.m_down1 = BaseEncBlock(
            nc[0], nc[0], nb=self.nb, img_channels=in_channels, decode_upscale=1, 
            normalization=self.is_normalization,
            conditioning=False if self.last_only_conditioning else (self.first_only_conditioning or self.encoder_conditioning),
            conditioning_list=self.first_only_conditioning_list,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )
        self.m_down2 = BaseEncBlock(
            nc[1], nc[1], nb=self.nb, img_channels=in_channels, decode_upscale=2, 
            normalization=self.is_normalization,
            conditioning=False if (self.last_only_conditioning or self.first_only_conditioning) else self.encoder_conditioning,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )
        self.m_down3 = BaseEncBlock(
            nc[2], nc[2], nb=self.nb, img_channels=in_channels, decode_upscale=4, 
            normalization=self.is_normalization,
            conditioning=False if (self.last_only_conditioning or self.first_only_conditioning) else self.encoder_conditioning,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )
        self.m_body = BaseEncBlock(
            nc[3], nc[3], nb=self.nb, img_channels=in_channels, decode_upscale=8, 
            normalization=self.is_normalization,
            conditioning=False if (self.last_only_conditioning or self.first_only_conditioning) else self.decoder_conditioning,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )
        self.m_up3 = BaseEncBlock(
            nc[2], nc[2], nb=self.nb, img_channels=in_channels, decode_upscale=4, 
            normalization=self.is_normalization,
            conditioning=False if (self.last_only_conditioning or self.first_only_conditioning) else self.decoder_conditioning,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )
        self.m_up2 = BaseEncBlock(
            nc[1], nc[1], nb=self.nb, img_channels=in_channels, decode_upscale=2, 
            normalization=self.is_normalization,
            conditioning=False if (self.last_only_conditioning or self.first_only_conditioning) else self.decoder_conditioning,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )
        self.m_up1 = BaseEncBlock(
            nc[0], nc[0], nb=self.nb, img_channels=in_channels, decode_upscale=1, 
            normalization=self.is_normalization,
            conditioning=False if self.first_only_conditioning else (self.decoder_conditioning or self.last_only_conditioning),
            conditioning_list=self.last_only_conditioning_list,
            krylov_degree=self.krylov_degree,
            z_embedding_dim=self.z_embedding_dim,
            use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
            use_scale_shift_norm=self.use_scale_shift_norm,
            use_scale_norm=self.use_scale_norm,
            channel_only_conditioning=self.channel_only_conditioning,
            cross_attention_conditioning=self.cross_attention_conditioning,
            deep_embedding=self.deep_embedding
        )

        self.pool1 = downsample_strideconv(nc[0], nc[1], bias=False, mode="2")
        self.pool2 = downsample_strideconv(nc[1], nc[2], bias=False, mode="2")
        self.pool3 = downsample_strideconv(nc[2], nc[3], bias=False, mode="2")
        self.up3 = upsample_convtranspose(nc[3], nc[2], bias=False, mode="2")
        self.up2 = upsample_convtranspose(nc[2], nc[1], bias=False, mode="2")
        self.up1 = upsample_convtranspose(nc[1], nc[0], bias=False, mode="2")

        self.m_tail = OutTail(nc[0], in_channels)

        self.sigma_threshold = 5e-3
        self.gain_threshold = 1e-4

        # load pretrained weights from hugging face
        if pretrained:
            if isinstance(pretrained, (str, Path)):
                self.load_state_dict(
                    torch.load(pretrained, map_location=device, weights_only=True)
                )
            else:
                self.load_state_dict(
                    torch.hub.load_state_dict_from_url(
                        "https://huggingface.co/mterris/ram/resolve/main/ram.pth.tar"
                    ),
                    strict=False,
                )

        if device is not None:
            self.to(device)

    def constant2map(self, value, x):
        r"""
        Converts a constant value to a map of the same size as the input tensor x.

        :param float value: constant value
        :param torch.Tensor x: input tensor
        :return torch.Tensor: a tensor of size (B, 1, W, H) containing constant maps of shapes (W, H) for each value in the batch.
        """
        import einops

        if isinstance(value, torch.Tensor):
            if value.ndim > 0:
                value_map = value.view(x.size(0), 1, 1, 1)
                value_map = value_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                value_map = einops.repeat(
                    value, "-> b 1 h w", b=x.size(0), h=x.size(2), w=x.size(3)
                )
        else:
            value_map = (
                torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                * value
            )
        return value_map

    def base_conditioning(self, x, sigma, gain):
        r"""
        Stacks the sigma and gain value as additional channel dimensions to the input tensor.

        :param torch.Tensor x: Input tensor
        :param float sigma: Gaussian noise level
        :param float gain: Poisson noise gain
        :return torch.Tensor: Input tensor with additional channels for sigma and gain
        """
        noise_level_map = self.constant2map(sigma, x)
        gain_map = self.constant2map(gain, x)
        return torch.cat((x, noise_level_map, gain_map), 1)

    def realign_input(self, x, physics, y, sigma):
        r"""
        Realign the input x based on the measurements y and the physics model.
        Applies the proximity operator of the L2 norm with respect to the physics model.

        :param torch.Tensor x: Input tensor
        :param deepinv.physics.Physics physics: Physics model
        :param torch.Tensor y: Measurements
        :return torch.Tensor: Realigned input tensor
        """
        if hasattr(physics, "factor"):
            f = physics.factor
        else:
            f = 1.0

        if isinstance(y, TensorList):
            num = y[0].reshape(y[0].shape[0], -1).abs().mean(1)
        else:
            num = y.reshape(y.shape[0], -1).abs().mean(1)

        snr = num / (sigma + 1e-4)  # SNR equivariant
        gamma = 1 / (1e-4 + 1 / (snr * f**2))
        gamma = gamma[(...,) + (None,) * (x.dim() - 1)]
        gamma = gamma * self.fact_realign
        gamma = gamma.clamp(min=1e-8)  # clamp to avoid negative or zero gamma
        model_input = physics.prox_l2(x, y, gamma=gamma)

        return model_input

    def forward_unet(self, x0, sigma=None, gain=None, physics=None, y=None, cond=None):
        r"""
        Forward pass of the UNet model.

        :param torch.Tensor x0: init image
        :param float sigma: Gaussian noise level
        :param float gamma: Poisson noise gain
        :param deepinv.physics.Physics physics: physics measurement operator
        :param torch.Tensor y: measurements
        """
        img_channels = x0.shape[1]
        physics = LinearPhysicsMultiScaler(physics, x0.shape[-3:], device=x0.device)

        y_list = []
        for scale in [0, 1, 2, 3]:
            factor = 2**scale
            physics.set_scale(scale)
            y_list.append(
                krylov_embeddings(
                    physics.downsample(x0, scale=scale), physics, factor, N=self.krylov_degree
                )
            )

        physics.set_scale(0)

        if self.separate_head and img_channels not in self.in_channels:
            raise ValueError(
                f"Input image has {img_channels} channels, but the network only has heads for {self.in_channels} channels."
            )

        if y is not None:
            x0 = self.realign_input(x0, physics, y, sigma)

        x0 = self.base_conditioning(x0, sigma, gain)

        if cond is not None and self.conditioning_dim > 0:
            x0 = torch.cat((x0, cond), dim=1)
            
        if cond is not None and self.cs_embedding is not None:
            cond = self.cs_embedding(cond)

        x1 = self.m_head(x0)

        x1_ = self.m_down1(
            x1, physics=physics, y=y_list[0], img_channels=img_channels, scale=0, z=cond
        )
        x2 = self.pool1(x1_)

        x3_ = self.m_down2(
            x2, physics=physics, y=y_list[1], img_channels=img_channels, scale=1, z=cond
        )
        x3 = self.pool2(x3_)

        x4_ = self.m_down3(
            x3, physics=physics, y=y_list[2], img_channels=img_channels, scale=2, z=cond
        )
        x4 = self.pool3(x4_)

        x = self.m_body(
            x4, physics=physics, y=y_list[3], img_channels=img_channels, scale=3, z=cond
        )

        x = self.up3(x + x4)
        x = self.m_up3(
            x, physics=physics, y=y_list[2], img_channels=img_channels, scale=2, z=cond
        )

        x = self.up2(x + x3)
        x = self.m_up2(
            x, physics=physics, y=y_list[1], img_channels=img_channels, scale=1, z=cond
        )

        x = self.up1(x + x2)
        x = self.m_up1(
            x, physics=physics, y=y_list[0], img_channels=img_channels, scale=0, z=cond
        )

        x = self.m_tail(x + x1, img_channels)

        return x
    
    def pre_forward(self, y, physics=None, sigma=None, gain=None, img_size=None):
        
        if isinstance(y, TensorList):
            max_val = y[0].abs().reshape(y[0].size(0), -1).amax(dim=1, keepdim=False)
        else:
            max_val = y.abs().reshape(y.size(0), -1).amax(dim=1, keepdim=False)

        # rescale elements in the batch where max_val > 5 * sigma_threshold
        rescale_val = torch.where(
            max_val > 5 * self.sigma_threshold,
            torch.tensor(1.0, device=max_val.device, dtype=max_val.dtype),
            max_val,
        )

        if isinstance(y, TensorList):
            for yi in y:
                yi /= rescale_val.view([yi.shape[0]] + [1] * (yi.ndim - 1))
        else:
            y = y / rescale_val.view([y.shape[0]] + [1] * (y.ndim - 1))

        if physics is None:
            physics = dinv.physics.Denoising(noise_model=dinv.physics.ZeroNoise())

        if img_size is None:
            if hasattr(physics, "img_shape") and physics.img_shape is not None:
                img_size = physics.img_shape
            elif hasattr(physics, "img_size") and physics.img_size is not None:
                img_size = physics.img_size
            else:
                img_size = physics.A_adjoint(y).shape[1:]

        sigma, gain = self.obtain_sigma_gain(
            physics=physics,
            sigma=sigma,
            gain=gain,
            rescale_val=rescale_val,
            device=y.device,
        )

        pad = (-img_size[-2] % 8, -img_size[-1] % 8)

        use_pad = False
        if pad[0] != 0 or pad[1] != 0:
            physics = PhysicsCropper(physics, pad)
            use_pad = True

        x_in = physics.A_adjoint(y)

        sigma = torch.maximum(
            sigma, torch.tensor(self.sigma_threshold, device=x_in.device)
        )
        sigma = self._handle_sigma(sigma)

        gain = torch.maximum(
            gain, torch.tensor(self.gain_threshold, device=x_in.device)
        )
        gain = self._handle_sigma(gain) 
        
        out_dict = {
            "x_in": x_in,
            "y": y,
            "sigma": sigma,
            "gain": gain,
            "physics": physics,
            "use_pad": use_pad,
            "rescale_val": rescale_val
        } 
        
        return out_dict
    
    def post_forward(self, out, physics=None, use_pad=None, rescale_val=None):
        if use_pad:
            out = physics.remove_pad(out)

        out = out * rescale_val.view([out.shape[0]] + [1] * (out.ndim - 1))
        
        return out

    def forward(self, y, physics=None, sigma=None, gain=None, img_size=None, cond=None, use_pad=None, rescale_val=None, pre_forward=True, x_in=None):
        r"""
        Reconstructs a signal estimate from measurements y

        .. note::

            The noise levels `sigma` and `gain` can be kept as `None` if a noise model with :class:`GaussianNoise <deepinv.physics.GaussianNoise>`,
            :class:`PoissonNoise <deepinv.physics.PoissonNoise>` or :class:`PoissonGaussianNoise <deepinv.physics.PoissonGaussianNoise>`
            is specified in the physics. If both are provided, the `sigma` and `gain` values provided to the model will be used.

        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float, torch.Tensor sigma: Gaussian noise level.
        :param float, torch.Tensor gain: Poisson noise level.
        :param tuple img_size: (optional) size of the image to reconstruct. If None, will be inferred automatically from the physics.
        :return: torch.Tensor: reconstructed signal estimate
        """

        if pre_forward:
            pre_dict = self.pre_forward(
                y, physics=physics, sigma=sigma, gain=gain, img_size=img_size
            )
            x_in = pre_dict["x_in"]
            y = pre_dict["y"]
            sigma = pre_dict["sigma"]
            gain = pre_dict["gain"]
            physics = pre_dict["physics"]
            use_pad = pre_dict["use_pad"]
            rescale_val = pre_dict["rescale_val"]
        else:
            assert x_in is not None, "x_in should be provided if pre_forward is False."

        out = self.forward_unet(x_in, sigma=sigma, gain=gain, physics=physics, y=y, cond=cond)

        if use_pad:
            out = physics.remove_pad(out)

        out = out * rescale_val.view([out.shape[0]] + [1] * (out.ndim - 1))

        return out
    
    def forward_encoder(self, y, physics=None, sigma=None, gain=None, img_size=None, cond=None, use_pad=None, rescale_val=None, pre_forward=True, x_in=None):
        
        if pre_forward:
            pre_dict = self.pre_forward(
                y, physics=physics, sigma=sigma, gain=gain, img_size=img_size
            )
            x_in = pre_dict["x_in"]
            y = pre_dict["y"]
            sigma = pre_dict["sigma"]
            gain = pre_dict["gain"]
            physics = pre_dict["physics"]
            use_pad = pre_dict["use_pad"]
            rescale_val = pre_dict["rescale_val"]
        else:
            assert x_in is not None, "x_in should be provided if pre_forward is False."


        x0 = x_in

        img_channels = x0.shape[1]
        physics = LinearPhysicsMultiScaler(physics, x0.shape[-3:], device=x0.device)

        y_list = []
        for scale in [0, 1, 2, 3]:
            factor = 2**scale
            physics.set_scale(scale)
            y_list.append(
                krylov_embeddings(
                    physics.downsample(x0, scale=scale), physics, factor, N=self.krylov_degree
                )
            )

        physics.set_scale(0)

        if self.separate_head and img_channels not in self.in_channels:
            raise ValueError(
                f"Input image has {img_channels} channels, but the network only has heads for {self.in_channels} channels."
            )

        if y is not None:
            x0 = self.realign_input(x0, physics, y, sigma)

        x0 = self.base_conditioning(x0, sigma, gain)

        if cond is not None and self.conditioning_dim > 0:
            x0 = torch.cat((x0, cond), dim=1)
            
        if cond is not None and self.cs_embedding is not None:
            cond = self.cs_embedding(cond)

        x1 = self.m_head(x0)

        x1_ = self.m_down1(
            x1, physics=physics, y=y_list[0], img_channels=img_channels, scale=0, z=cond
        )
        x2 = self.pool1(x1_)

        x3_ = self.m_down2(
            x2, physics=physics, y=y_list[1], img_channels=img_channels, scale=1, z=cond
        )
        x3 = self.pool2(x3_)

        x4_ = self.m_down3(
            x3, physics=physics, y=y_list[2], img_channels=img_channels, scale=2, z=cond
        )
        x4 = self.pool3(x4_)

        x = self.m_body(
            x4, physics=physics, y=y_list[3], img_channels=img_channels, scale=3, z=cond
        )
        
        return x

    def obtain_sigma_gain(self, physics, sigma, gain, rescale_val, device="cpu"):
        r"""
        Defines the sigma and gain values to be used in the model.

        If a noise model is specified in the physics, the sigma and gain values will be taken from the noise model.
        Else, the sigma and gain values will be set to the thresholds defined in the model (if not provided).

        :param deepinv.physics.LinearPhysics physics: Physics operator
        :param float, torch.Tensor sigma: Gaussian noise level. If None, will be set to the threshold.
        :param float, torch.Tensor gain: Poisson noise level. If None, will be set to the threshold.
        :param torch.Tensor rescale_val: Rescale value to apply to the sigma and gain values.
        :param str, torch.device device: Device to which the sigma and gain values should be moved.
        """
        if sigma is None:
            sigma = (
                physics.noise_model.sigma / rescale_val
                if hasattr(physics.noise_model, "sigma")
                else self.sigma_threshold
                * torch.ones(rescale_val.size(0), device=device)
            )
            if isinstance(sigma, TensorList):
                sigma = sigma.abs().max()
        else:
            sigma = sigma / rescale_val
            if hasattr(physics.noise_model, "sigma"):
                warn(
                    "Both sigma provided to the model and a noise model in the physics. The sigma provided to the model will be used."
                )

        if gain is None:
            gain = (
                physics.noise_model.gain / rescale_val
                if hasattr(physics.noise_model, "gain")
                else self.gain_threshold
                * torch.ones(rescale_val.size(0), device=device)
            )
            if isinstance(gain, TensorList):
                gain = gain.abs().max()
        else:
            gain = gain / rescale_val
            if hasattr(physics.noise_model, "gain"):
                warn(
                    "Both gain provided to the model and a noise model in the physics. The gain provided to the model will be used."
                )

        if not isinstance(sigma, torch.Tensor) and not isinstance(sigma, TensorList):
            sigma = torch.tensor(sigma, device=device)
        if not isinstance(gain, torch.Tensor) and not isinstance(gain, TensorList):
            gain = torch.tensor(gain, device=device)

        return sigma, gain


class BaseEncBlock(nn.Module):
    r"""
    Base encoding block for the RAM model.

    This block consists of multiple convolutional residual blocks.

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param bool bias: Whether to use bias in the convolution.
    :param int nb: Number of residual blocks in the encoding block.
    :param int, list[int] img_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param int decode_upscale: Upscaling factor for the decoding convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        nb=4,
        img_channels=None,
        decode_upscale=None,
        normalization=False,
        conditioning=True,
        conditioning_list=None,
        krylov_degree=2, # number of krylov iterations, if 1 
        z_embedding_dim=0,
        use_scale_shift_gate_norm=False,
        use_scale_shift_norm=False,
        use_scale_norm=False,
        channel_only_conditioning=False,
        cross_attention_conditioning=False,
        deep_embedding=False
        
    ):
        super(BaseEncBlock, self).__init__()
        
        # custom variables
        self.is_normalization = normalization
        self.conditioning = conditioning_list if conditioning_list is not None else [conditioning] * nb
        self.krylov_degree = krylov_degree
        self.z_embedding_dim = z_embedding_dim
        assert not use_scale_shift_norm or not use_scale_norm, "use_scale_shift_norm and use_scale_norm cannot be both True."
        self.use_scale_shift_gate_norm = use_scale_shift_gate_norm
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_scale_norm = use_scale_norm
        self.channel_only_conditioning = channel_only_conditioning
        self.cross_attention_conditioning = cross_attention_conditioning
        self.deep_embedding = deep_embedding
        
        self.enc = nn.ModuleList(
            [
                ResBlock(
                    in_channels,
                    out_channels,
                    bias=bias,
                    img_channels=img_channels,
                    decode_upscale=decode_upscale,
                    normalization=self.is_normalization,
                    conditioning=self.conditioning[k],
                    N=self.krylov_degree,
                    z_embedding_dim=z_embedding_dim,
                    use_scale_shift_gate_norm=self.use_scale_shift_gate_norm,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    use_scale_norm=self.use_scale_norm,
                    channel_only_conditioning=self.channel_only_conditioning,
                    cross_attention_conditioning=self.cross_attention_conditioning,
                    deep_embedding=self.deep_embedding
                )
                for k in range(nb)
            ]
        )

    def forward(self, x, physics=None, y=None, img_channels=None, scale=0, z=None):
        r"""
        Forward pass of the encoding block.

        :param torch.Tensor x: Input tensor
        :param deepinv.physics.Physics physics: Physics
        :param torch.Tensor y: Measurements
        :param int img_channels: Number of input channels.
        :param int scale: Scale factor for the encoding block.
        """
        for i in range(len(self.enc)):
            x = self.enc[i](
                x, physics=physics, y=y, img_channels=img_channels, scale=scale, z=z
            )
        return x


def krylov_embeddings(y, p, factor, v=None, N=4, x_init=None):
    r"""
    Efficient Krylov subspace embedding computation with parallel processing.

    :param torch.Tensor y: Input tensor.
    :param deepinv.physics.Physics p: A deepinv physics.
    :param float factor: Scaling factor.
    :param torch.Tensor v: Precomputed values to subtract from Krylov sequence. Defaults to None.
    :param int N: Number of Krylov iterations. Defaults to 4.
    :param torch.Tensor x_init: Initial guess. Defaults to None.
    :return: torch.Tensor: a stacked tensor over the channel dimension containing Krylov embeddings.
    """

    if x_init is None:
        x = y
    else:
        x = x_init

    norm = factor**2  # Precompute normalization factor
    AtA = lambda u: p.A_adjoint_A(u) * norm  # Define the linear operator

    v = v if v is not None else torch.zeros_like(x)

    out = x
    # Compute Krylov basis
    x_k = x.clone()
    for i in range(N - 1):
        x_k = AtA(x_k) - v
        out = torch.cat([out, x_k], dim=1)

    return out


class MeasCondBlock(nn.Module):
    r"""
    Measurement conditioning block for the RAM model.

    :param int out_channels: Number of output channels.
    :param int img_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param int decode_upscale: Upscaling factor for the decoding convolution.
    :param int N: Number of Krylov iterations.
    :param int depth_encoding: Depth of the encoding convolution.
    :param int c_mult: Multiplier for the number of channels.
    """

    def __init__(
        self,
        out_channels=64,
        img_channels=None,
        decode_upscale=None,
        N=4,
        depth_encoding=1,
        c_mult=1,
    ):
        super(MeasCondBlock, self).__init__()

        self.separate_head = isinstance(img_channels, list)

        assert img_channels is not None, "decode_dimensions should be provided"
        assert decode_upscale is not None, "decode_upscale should be provided"

        self.N = N
        self.c_mult = c_mult
        self.relu_encoding = nn.ReLU(inplace=False)
        self.decoding_conv = Tails(
            out_channels, img_channels, depth=1, scale=1, bias=False, c_mult=self.c_mult
        )
        self.encoding_conv = Heads(
            img_channels,
            out_channels,
            depth=depth_encoding,
            scale=1,
            bias=False,
            c_mult=self.c_mult * N,
            c_add=N,
            relu_in=False,
            skip_in=True,
        )

    def forward(self, x, y, physics, img_channels=None, scale=1):
        physics.set_scale(scale)
        dec = self.decoding_conv(x, img_channels)
        factor = 2 ** (scale)
        meas_y = y
        meas_dec = krylov_embeddings(
            None, physics, factor, N=self.N, x_init=dec[:, :img_channels, ...]
        )
        for c in range(1, self.c_mult):
            meas_cur = krylov_embeddings(
                None,
                physics,
                factor,
                N=self.N,
                x_init=dec[:, img_channels * c : img_channels * (c + 1)],
            )
            meas_dec = torch.cat([meas_dec, meas_cur], dim=1)
        meas = torch.cat([meas_y, meas_dec], dim=1)
        cond = self.encoding_conv(meas)
        emb = self.relu_encoding(cond)
        return emb

from deepinv.models.jit import RMSNorm, SwiGLUFFN


def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()

    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class CrossAttentionBlock(nn.Module):
    def __init__(self, q_channels, kv_channels, out_channels=None, qk_norm=True, qkv_bias=True, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        self.q_channels = q_channels
        
        self.head_dim = out_channels // num_heads
        self.num_heads = num_heads
        assert out_channels % num_heads == 0, "q_channels must be divisible by num_heads"
        
        self.kv_channels = kv_channels
        self.out_channels = out_channels if out_channels is not None else q_channels
        
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        
        self.q_proj = nn.Linear(self.q_channels, self.out_channels, bias=qkv_bias)
        self.kv_proj = nn.Linear(self.kv_channels, 2 * self.out_channels, bias=qkv_bias)  # Project K and V together for efficiency
        # self.k_proj = nn.Linear(self.kv_channels, self.out_channels, bias=qkv_bias)
        # self.v_proj = nn.Linear(self.kv_channels, self.out_channels, bias=qkv_bias)
        
        self.proj = nn.Linear(self.out_channels, self.out_channels, bias=qkv_bias)
        
        # self.mlp_norm = RMSNorm(self.out_channels)
        # self.mlp = SwiGLUFFN(self.out_channels, self.out_channels, bias=True)
        
    def forward(self, x, context):
        # Reshape context to (B, C)
        context = context.mean(dim=[2, 3])[:, None]  # Reduce spatial dimensions to (B, 1, C)
        
        B,C,H,W = x.shape
        
        # Project Q, K, V
        query = self.q_proj(x.flatten(2).permute(0, 2, 1))  # Shape: (B, H*W, C)
        # key = self.k_proj(context)  # Shape: (B, C)
        # value = self.v_proj(context)  # Shape: (B, C)
        key_value = self.kv_proj(context)  # Shape: (B, 1, 2*C)
        
        # head_dim = self.out_channels // self.num_heads
        query = query.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # Shape: (B, num_heads, H*W, head_dim)
        key_value = key_value.reshape(B, 1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # Shape: (2, B, num_heads, 1, head_dim)
        key, value = key_value[0], key_value[1] # Shapes: (B, num_heads, 1, head_dim)

        # Compute attention weights
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )  # Shape: (B, num_heads, H*W, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, H*W, self.out_channels)  # Shape: (B, H*W, C)
        
        attn_output = self.proj(attn_output)  # Shape: (B, H*W, C)
        attn_output = torch.nn.functional.silu(attn_output)  # Non-linearity
        attn_output = attn_output.permute(0, 2, 1).view_as(x)  # Reshape back to (B, C, H, W)

        return attn_output

class ResBlock(nn.Module):
    r"""
    Convolutional residual block.

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param int kernel_size: Size of the convolution kernel.
    :param int stride: Stride of the convolution.
    :param int padding: Padding for the convolution.
    :param bool bias: Whether to use bias in the convolution.
    :param int, list[int] img_channels: Number of input channels. If a list is provided, the model will have separate heads for each channel.
    :param int decode_upscale: Upscaling factor for the decoding convolution.
    :param bool head: Whether this is a head block.
    :param bool tail: Whether this is a tail block.
    :param int N: Number of Krylov iterations.
    :param int c_mult: Multiplier for the number of channels.
    :param int depth_encoding: Depth of the encoding convolution.
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        img_channels=None,
        decode_upscale=None,
        head=False,
        tail=False,
        N=2,
        c_mult=2,
        depth_encoding=2,
        normalization=False,
        conditioning=True,
        z_embedding_dim=0,
        use_scale_shift_gate_norm=False,
        use_scale_shift_norm=False,
        use_scale_norm=False,
        channel_only_conditioning=False,
        cross_attention_conditioning=False,
        deep_embedding=False
    ):
        super(ResBlock, self).__init__()

        # custom variables
        self.conditioning = conditioning
        self.is_normalization = normalization
        self.z_embedding_dim = z_embedding_dim

        if not head and not tail:
            assert (
                in_channels == out_channels
            ), "Only support in_channels==out_channels."
        self.separate_head = isinstance(img_channels, list)
        self.is_head = head
        self.is_tail = tail
        
        if self.is_head:
            self.head = InHead(img_channels, out_channels, input_layer=True)

        if not self.is_head and not self.is_tail:
            self.conv1 = conv(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias,
                "C",
            )
            self.nl = nn.ReLU(inplace=True)
            self.conv2 = conv(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias,
                "C",
            )

            if self.is_normalization:
                # self.normalization = LayerNorm_AF2d(out_channels)
                self.norm = RMSNorm2d(out_channels)
                # self.norm1 = RMSNorm2d(out_channels)
                # self.norm2 = RMSNorm2d(out_channels)
        
        self.gain = None
        self.PhysicsBlock = None
        if self.conditioning:
            self.gain = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            self.PhysicsBlock = MeasCondBlock(
                out_channels=out_channels,
                c_mult=c_mult,
                img_channels=img_channels,
                decode_upscale=decode_upscale,
                N=N,
                depth_encoding=depth_encoding,
            )
    
        assert not (use_scale_shift_norm and use_scale_norm), "use_scale_shift_norm and use_scale_norm cannot be both True."
        self.use_scale_shift_gate_norm = use_scale_shift_gate_norm
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_scale_norm = use_scale_norm

        self.deep_embedding = deep_embedding
        self.cross_attention_conditioning = cross_attention_conditioning
        if self.z_embedding_dim > 0:
            # NOTE In diffusion for timestep embedding, layer is usually smaller
            # though it's only channel-wise addition, so conditioning signal is weaker
            # self.emb_layers = nn.Sequential(
            #     nn.SiLU(),
            #     nn.Linear(
            #         emb_channels,
            #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            #     ),
            # )
            
            # FIXME, legacy for first experiments, with CSEmbedStem k=7
            # self.cs_embedding = CSEmbedding(
            #     z_dim=self.z_embedding_dim,
            #     embed_dim=out_channels,
            #     activation="ReLU",
            #     kernel_size=3
            # )
            self.channel_only_conditioning = channel_only_conditioning
            
            if self.cross_attention_conditioning:
                self.cross_attention = CrossAttentionBlock(
                    q_channels=out_channels,
                    kv_channels=z_embedding_dim,
                    out_channels=out_channels
                )
                
                # self.mlp_norm = RMSNorm(out_channels)
                # self.mlp = SwiGLUFFN(out_channels, out_channels * 4, bias=True)
            else:
                # self.cs_norm = RMSNorm2d(self.z_embedding_dim)
                if self.deep_embedding:
                    # self.cs_norm = RMSNorm2d(self.z_embedding_dim)
                    self.cs_embedding = nn.Sequential(
                        # nn.Conv2d(
                        #     in_channels=self.z_embedding_dim,
                        #     out_channels=self.z_embedding_dim,
                        #     kernel_size=1,
                        #     bias=True # only True for rozet
                        # ),
                        nn.SiLU(),
                        nn.Conv2d(
                        in_channels=self.z_embedding_dim,
                        out_channels=out_channels * 3 if use_scale_shift_gate_norm else (out_channels * 2 if use_scale_shift_norm else out_channels),
                        kernel_size=1,
                        bias=True # only True for rozet
                        )
                    )
                    # nn.init.constant_(self.cs_embedding[-1].weight, 0)
                    # nn.init.constant_(self.cs_embedding[-1].bias, 0)
                    self.cs_embedding[-1].weight.data.mul_(1e-2)
                    self.cs_embedding[-1].bias.data.mul_(1e-2)

                    # for logging purposes
                    self.cs_embedding.use_scale_shift_gate_norm = use_scale_shift_gate_norm
                    self.cs_embedding.use_scale_shift_norm = use_scale_shift_norm
                    self.cs_embedding.use_scale_norm = use_scale_norm
                else: 
                    # self.cs_embedding = LayerEmbedding(
                    #     in_channels=self.z_embedding_dim,
                    #     out_channels=out_channels,
                    #     kernel_size=3 if not self.deep_embedding else 1,
                    #     bias=False,
                    #     use_scale_shift_norm=use_scale_shift_norm
                    # )
                    self.cs_embedding = nn.Sequential(
                        nn.Linear(self.z_embedding_dim, self.z_embedding_dim),
                        nn.SiLU(),
                        nn.Linear(
                            self.z_embedding_dim, 
                            out_channels * 3 if use_scale_shift_gate_norm 
                            else (out_channels * 2 if use_scale_shift_norm 
                            else out_channels)
                        ),
                    )             

                    self.cs_embedding[-1].weight.data.mul_(1e-2)
                    self.cs_embedding[-1].bias.data.mul_(1e-2)

                    # for logging purposes
                    self.cs_embedding.use_scale_shift_gate_norm = use_scale_shift_gate_norm
                    self.cs_embedding.use_scale_shift_norm = use_scale_shift_norm
                    self.cs_embedding.use_scale_norm = use_scale_norm

        else:
            if self.use_scale_shift_gate_norm:
                self.adaLN_zero = nn.Parameter(torch.randn(3, out_channels, 1, 1))
                self.adaLN_zero.data.mul_(1e-2)

    def rozet_forward(self, x, physics=None, y=None, img_channels=None, scale=0, z=None):
        scaling, shift, gate = None, None, None
        if self.z_embedding_dim > 0 and z is not None:            
            if self.cross_attention_conditioning:
                pass
            else:
                if self.deep_embedding:
                    z_emb = self.cs_embedding(z)
                else:
                    # z_emb.shape = (B, C) for linear embedding, need to unsqueeze to (B, C, 1, 1) for channel-wise addition
                    z_emb = self.cs_embedding(z)[...,None,None]
                    # raise NotImplementedError("Only deep_embedding is implemented for now.")
                
                if self.use_scale_shift_gate_norm:
                    scaling, shift, gate = z_emb.chunk(3, dim=1)
                elif self.use_scale_shift_norm:
                    scaling, shift = z_emb.chunk(2, dim=1)
                elif self.use_scale_norm:
                    scaling = z_emb
                else:
                    shift = z_emb
        else:
            if self.use_scale_shift_gate_norm:
                scaling, shift, gate = self.adaLN_zero.chunk(3, dim=0)

        if self.is_normalization:
            u = modulate(self.norm(x), scale=scaling, shift=shift, gate=None)
        else:
            u = modulate(x, scale=scaling, shift=shift, gate=None)

        if self.cross_attention_conditioning:
            u = u + self.cross_attention(u, z)

        # identical to self.ffn in rozet's unet
        u = self.conv1(u)
        u = self.nl(u)
        u_2 = self.conv2(u)
        
        if self.conditioning:
            emb_grad = self.PhysicsBlock(
                u, y, physics, img_channels=img_channels, scale=scale
            )
            u_1 = self.gain * emb_grad
            return x + u_2 + u_1
        
        if gate is not None:
            u = (x + gate * u_2) * torch.rsqrt(1 + gate * gate)
        else:
            u = x + u_2

        return u

    def vanilla_forward(self, x, physics=None, y=None, img_channels=None, scale=0, z=None):
        r"""
        Forward pass of the residual block.

        :param torch.Tensor x: Input tensor
        :param deepinv.physics.Physics physics: Physics
        :param torch.Tensor y: Measurements
        :param int img_channels: Number of input channels.
        :param int scale: Scale factor for the encoding block.
        """
        u = self.conv1(x)
        # Conv -> batch -> ReLU: https://github.com/huggingface/pytorch-image-models/blob/5853f99e4dfbeb9cd83b86f10d8c352227b6a08f/timm/models/resnet.py#L132
        if self.is_normalization:
            u = self.normalization(u)
        u = self.nl(u)
        u_2 = self.conv2(u)
        if self.conditioning:
            emb_grad = self.PhysicsBlock(
                u, y, physics, img_channels=img_channels, scale=scale
            )
            u_1 = self.gain * emb_grad
            return x + u_2 + u_1
        else:
            return x + u_2
        
    def custom_forward(self, x, physics=None, y=None, img_channels=None, scale=0, z=None):
        scale1, shift1, scale2, shift2 = None, None, None, None
        if self.z_embedding_dim > 0 and z is not None:
            if self.deep_embedding:
                z_emb = self.cs_embedding(z)
        
            if self.use_scale_shift_norm:
                scale1, shift1, scale2, shift2 = z_emb.chunk(4, dim=1)
            elif self.use_scale_norm:
                scale1, scale2 = z_emb.chunk(2, dim=1)
            else:
                shift1, shift2 = z_emb.chunk(2, dim=1)

            if self.is_normalization:
                u = modulate(self.norm1(x), scale=scale1, shift=shift1)
            else:
                u = modulate(x, scale=scale1, shift=shift1) 
            u = self.nl(u)
            x = x + self.conv1(u)

            # Conv -> batch -> ReLU: https://github.com/huggingface/pytorch-image-models/blob/5853f99e4dfbeb9cd83b86f10d8c352227b6a08f/timm/models/resnet.py#L132
            if self.is_normalization:
                u = modulate(self.norm2(x), scale=scale2, shift=shift2)
            else:
                u = modulate(x, scale=scale2, shift=shift2)
                
            u = self.nl(u)
            u_2 = x + self.conv2(u)
            
            return u_2

    def forward(self, x, physics=None, y=None, img_channels=None, scale=0, z=None):
        r"""
        Forward pass of the residual block.

        :param torch.Tensor x: Input tensor
        :param deepinv.physics.Physics physics: Physics
        :param torch.Tensor y: Measurements
        :param int img_channels: Number of input channels.
        :param int scale: Scale factor for the encoding block.
        """
        
        if self.z_embedding_dim > 0 and z is not None:
            return self.rozet_forward(x, physics=physics, y=y, img_channels=img_channels, scale=scale, z=z)
            # return self.custom_forward(x, physics=physics, y=y, img_channels=img_channels, scale=scale, z=z)
        else:
            # return self.vanilla_forward(x, physics=physics, y=y, img_channels=img_channels, scale=scale, z=z)
            return self.rozet_forward(x, physics=physics, y=y, img_channels=img_channels, scale=scale, z=z)
        
        # if self.z_embedding_dim > 0 and z is not None:            
        #     if self.cross_attention_conditioning:
        #         # u_2 = u_2 + self.cross_attention(u_2, z)
        #         u_2 = u_2 + self.cross_attention(u_2, z)
        #         # u_2 = u_2 \
        #         #     + self.mlp(
        #         #         self.mlp_norm(
        #         #             u_2.flatten(2).permute(0, 2, 1)
        #         #         )
        #         #     ).permute(0, 2, 1).view_as(u_2)
                
        #     else:
        #         if self.deep_embedding:
        #             z_emb = self.cs_embedding(self.cs_norm(z))
        #         else:
        #             if z.shape[2] == 1 and z.shape[3] == 1:
        #                 z_emb = self.cs_embedding(self.cs_norm(z))
        #             else:
        #                 z_resized = nn.functional.interpolate(
        #                     z, size=u_2.shape[2:], mode="bilinear", align_corners=False
        #                 )
        #                 z_emb = self.cs_embedding(z_resized)
                        
        #         if self.channel_only_conditioning:
        #             z_emb = z_emb.mean(dim=[2, 3], keepdim=True)
                
        #         if self.use_scale_shift_norm:
        #             scale, shift = z_emb.chunk(2, dim=1)
        #             u_2 = u_2 * (1 + scale) + shift
        #         elif self.use_scale_norm:
        #             scale = z_emb
        #             u_2 = u_2 * (1 + scale)
        #         else:
        #             u_2 = u_2 + z_emb
        
        if self.conditioning:
            emb_grad = self.PhysicsBlock(
                u, y, physics, img_channels=img_channels, scale=scale
            )
            u_1 = self.gain * emb_grad
            return x + u_2 + u_1
        else:
            return x + u_2

class InHead(torch.nn.Module):
    r"""
    Input head for the RAM model.

    This module applies a convolution to the input tensor based on the number of input channels.

    :param list[int] in_channels_list: List of input channels for each head.
    :param int out_channels: Number of output channels for the convolution.
    :param bool bias: Whether to use bias in the convolution.
    :param bool input_layer: If True, this will be considered as an input layer (necessitating a channel number adjustment), otherwise it will not.
    """

    def __init__(self, in_channels_list, out_channels, bias=False, input_layer=False):
        super(InHead, self).__init__()
        self.in_channels_list = in_channels_list
        self.input_layer = input_layer
        for i, in_channels in enumerate(in_channels_list):
            conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
            )
            setattr(self, f"conv{i}", conv)

    def forward(self, x):
        in_channels = x.size(1) - 1 if self.input_layer else x.size(1)

        # find index
        i = self.in_channels_list.index(in_channels)
        x = getattr(self, f"conv{i}")(x)

        return x


class OutTail(torch.nn.Module):
    r"""
    Output tail for the RAM model.

    This module applies a convolution to the input tensor based on the number of output channels.

    :param int in_channels: Number of input channels.
    :param list[int] out_channels_list: List of output channels for each tail.
    :param bool bias: Whether to use bias in the convolution.
    """

    def __init__(self, in_channels, out_channels_list, bias=False):
        super(OutTail, self).__init__()
        self.in_channels = in_channels
        self.out_channels_list = out_channels_list
        for i, out_channels in enumerate(out_channels_list):
            conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=bias,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="zeros",
            )
            setattr(self, f"conv{i}", conv)

    def forward(self, x, out_channels):
        i = self.out_channels_list.index(out_channels)
        x = getattr(self, f"conv{i}")(x)

        return x


class Heads(torch.nn.Module):
    r"""
    General heads module for the RAM model.

    :param list[int] in_channels_list: List of input channels for each head.
    :param int out_channels: Number of output channels for the convolution.
    :param int depth: Depth of the head block.
    :param int scale: Scale factor for the downsampling or upsampling.
    :param bool bias: Whether to use bias in the convolution.
    :param str mode: Mode for the upsampling, e.g., "bilinear".
    :param int c_mult: Multiplier for the number of channels.
    :param int c_add: Additional channels to add to the input.
    :param bool relu_in: If True, applies ReLU activation after the input convolution.
    :param bool skip_in: If True, applies a skip connection from the input to the output.
    """

    def __init__(
        self,
        in_channels_list,
        out_channels,
        depth=2,
        scale=1,
        bias=True,
        mode="bilinear",
        c_mult=1,
        c_add=0,
        relu_in=False,
        skip_in=False,
    ):
        super(Heads, self).__init__()
        self.in_channels_list = [c * (c_mult + c_add) for c in in_channels_list]
        self.scale = scale
        self.mode = mode
        for i, in_channels in enumerate(self.in_channels_list):
            setattr(
                self,
                f"head{i}",
                HeadBlock(
                    in_channels,
                    out_channels,
                    depth=depth,
                    bias=bias,
                    relu_in=relu_in,
                    skip_in=skip_in,
                ),
            )

        if self.mode == "":
            self.nl = torch.nn.ReLU(inplace=False)
            if self.scale != 1:
                for i, in_channels in enumerate(in_channels_list):
                    setattr(
                        self,
                        f"down{i}",
                        downsample_strideconv(
                            in_channels, in_channels, bias=False, mode=str(self.scale)
                        ),
                    )

    def forward(self, x):
        in_channels = x.size(1)
        i = self.in_channels_list.index(in_channels)

        if self.scale != 1:
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(
                    x, scale_factor=1 / self.scale, mode="bilinear", align_corners=False
                )
            else:
                x = getattr(self, f"down{i}")(x)
                x = self.nl(x)

        # find index
        x = getattr(self, f"head{i}")(x)

        return x


class Tails(torch.nn.Module):
    r"""
    General tails module for the RAM model.

    :param int in_channels: Number of input channels.
    :param list[int] out_channels_list: List of output channels for each tail.
    :param int depth: Depth of the tail block.
    :param int scale: Scale factor for the upsampling.
    :param bool bias: Whether to use bias in the convolution.
    :param str mode: Mode for the upsampling, e.g., "bilinear".
    :param int c_mult: Multiplier for the number of channels.
    :param bool relu_in: If True, applies ReLU activation after the input convolution.
    :param bool skip_in: If True, applies a skip connection from the input to the output.
    """

    def __init__(
        self,
        in_channels,
        out_channels_list,
        depth=2,
        scale=1,
        bias=True,
        mode="bilinear",
        c_mult=1,
        relu_in=False,
        skip_in=False,
    ):
        super(Tails, self).__init__()
        self.out_channels_list = out_channels_list
        self.scale = scale
        for i, out_channels in enumerate(out_channels_list):
            setattr(
                self,
                f"tail{i}",
                HeadBlock(
                    in_channels,
                    out_channels * c_mult,
                    depth=depth,
                    bias=bias,
                    relu_in=relu_in,
                    skip_in=skip_in,
                ),
            )

        self.mode = mode
        if self.mode == "":
            self.nl = torch.nn.ReLU(inplace=False)
            if self.scale != 1:
                for i, out_channels in enumerate(out_channels_list):
                    setattr(
                        self,
                        f"up{i}",
                        upsample_convtranspose(
                            out_channels * c_mult,
                            out_channels * c_mult,
                            bias=bias,
                            mode=str(self.scale),
                        ),
                    )

    def forward(self, x, out_channels):
        i = self.out_channels_list.index(out_channels)
        x = getattr(self, f"tail{i}")(x)
        # find index
        if self.scale != 1:
            if self.mode == "bilinear":
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.scale, mode="bilinear", align_corners=False
                )
            else:
                x = getattr(self, f"up{i}")(x)

        return x


class HeadBlock(torch.nn.Module):
    r"""
    Head block for the RAM model.

    This module applies a series of convolutions to the input tensor, with optional skip connections and ReLU activations.

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param int kernel_size: Size of the convolution kernel.
    :param bool bias: Whether to use bias in the convolution.
    :param int depth: Depth of the head block.
    :param bool relu_in: If True, applies ReLU activation after the input convolution.
    :param bool skip_in: If True, applies a skip connection from the input to the output.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bias=True,
        depth=2,
        relu_in=False,
        skip_in=False,
    ):
        super(HeadBlock, self).__init__()

        padding = kernel_size // 2

        c = out_channels if depth < 2 else in_channels

        # NOTE: c=in_channels, because default is depth=2
        self.convin = torch.nn.Conv2d(
            in_channels, c, kernel_size, padding=padding, bias=bias
        )
        self.zero_conv_skip = torch.nn.Conv2d(in_channels, c, 1, bias=False)
        self.depth = depth
        self.nl_1 = torch.nn.ReLU(inplace=False)
        self.nl_2 = torch.nn.ReLU(inplace=False)
        self.relu_in = relu_in # default is False
        self.skip_in = skip_in # default is True

        for i in range(depth - 1):
            if i < depth - 2:
                c_in, c = in_channels, in_channels
            else:
                c_in, c = in_channels, out_channels

            setattr(
                self,
                f"conv1{i}",
                torch.nn.Conv2d(c_in, c_in, kernel_size, padding=padding, bias=bias),
            )
            setattr(
                self,
                f"conv2{i}",
                torch.nn.Conv2d(c_in, c, kernel_size, padding=padding, bias=bias),
            )
            setattr(self, f"skipconv{i}", torch.nn.Conv2d(c_in, c, 1, bias=False))

    def forward(self, x):

        if self.skip_in and self.relu_in: # default is False
            x = self.nl_1(self.convin(x)) + self.zero_conv_skip(x)
        elif self.skip_in and not self.relu_in: # default is True
            x = self.convin(x) + self.zero_conv_skip(x)
        else:
            x = self.convin(x)

        for i in range(self.depth - 1):
            aux = getattr(self, f"conv1{i}")(x)
            aux = self.nl_2(aux)
            aux_0 = getattr(self, f"conv2{i}")(aux)
            aux_1 = getattr(self, f"skipconv{i}")(x)
            x = aux_0 + aux_1

        return x


def sequential(*args):
    r"""
    Creates a sequential container from the provided arguments.

    This function takes as input a list of modules or Sequential containers and returns a single nn.Sequential container.

    Function borrowed from https://github.com/xinntao/BasicSR.

    :param args: Modules or Sequential containers to be combined.
    :return: nn.Sequential container containing all the modules.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CR",
):
    r"""
    Conv + ReLU layer with optional transposed convolution.

    Takes as input a string `mode` that defines the sequence of operations.

    Code borrowed from https://github.com/cszn/DPIR/tree/master/models

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param int kernel_size: Size of the convolution kernel.
    :param int stride: Stride of the convolution.
    :param int padding: Padding for the convolution.
    :param bool bias: Whether to use bias in the convolution.
    :param str mode: Sequence of operations, e.g., "CR", "CT", "C", "T", etc.
    """
    L = []
    for t in mode:
        if t == "C":
            L.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "T":
            L.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            )
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError("Undefined type: ".format(t))
    return sequential(*L)


def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    padding=0,
    bias=True,
    mode="2R",
):
    r"""
    Upsample using ConvTranspose2d + ReLU layer.

    Takes as input a string `mode` that defines the sequence of operations.

    Code borrowed from https://github.com/cszn/DPIR/tree/master/models

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param int padding: Padding for the convolution.
    :param bool bias: Whether to use bias in the convolution.
    :param str mode: Sequence of operations, e.g., "2R", "3", "4R", etc.
    """
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
        "8",
    ], "mode examples: 2, 2R, 2R, 3, ..., 4R."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
    )
    return up1


def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    padding=0,
    bias=True,
    mode="2R",
):
    r"""
    Downsample using Conv2d with stride + ReLU layer.

    Takes as input a string `mode` that defines the sequence of operations.

    :param int in_channels: Number of input channels.
    :param int out_channels: Number of output channels.
    :param int padding: Padding for the convolution.
    :param bool bias: Whether to use bias in the convolution.
    :param str mode: Sequence of operations, e.g., "2R", "3", "4R", etc.
    """
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
        "8",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
    )
    return down1
