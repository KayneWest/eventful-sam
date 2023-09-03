import itertools
from math import sqrt, prod
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as func
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F

import torch
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


LN_EPS = 1e-6

def expand_col_index(index, target_shape):
    old_shape = index.shape
    new_dims = len(target_shape) - index.ndim
    index = index.view(old_shape[:-1] + (1,) * new_dims + old_shape[-1:])
    index = index.expand(target_shape[:-1] + (-1,))
    return index

def expand_row_index(index, target_shape):
    old_shape = index.shape
    new_dims = len(target_shape) - index.ndim
    index = index.view(old_shape[:-1] + (1,) * (new_dims - 1) + (old_shape[-1], 1))
    index = index.expand(target_shape[:-2] + (-1, target_shape[-1]))
    return index

def numeric_tuple(x, length):
    """
    Expands a single numeric value (int, float, complex, or bool) into a
    tuple of a specified length. If the value is not of the specified
    types, does nothing.

    :param x: The input value
    :param length: The length of tuple to return if x is of the
    specified types
    """
    return (x,) * length if isinstance(x, (int, float, complex, bool)) else tuple(x)

def pad_to_size(x, size, pad_tensor=None):
    # padding = [0, size[1] - x.shape[-1], 0, size[0] - x.shape[-2]]
    # x = func.pad(x, padding, fill=0, padding_mode="constant")
    # The two lines above are not working as expected - maybe there's a
    # bug in func.pad? In the meantime we'll use the concat-based
    # padding code below.
    if pad_tensor is None:
        pad_tensor = torch.zeros((1,) * x.ndim, dtype=x.dtype, device=x.device)
    for dim in list(range(-1, -len(size) - 1, -1)):
        expand_shape = list(x.shape)
        expand_shape[dim] = size[dim] - x.shape[dim]
        if expand_shape[dim] == 0:
            continue

        # torch.concat allocates a new tensor. So, we're safe to use
        # torch.expand here (instead of torch.repeat) without worrying
        # about different elements of x referencing the same data.
        x = torch.concat([x, pad_tensor.expand(expand_shape)], dim)
    return x

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c=2
        if(out_dim==320 or out_dim==448 or out_dim==576):
            stride_c=1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimpleSTGTGate(nn.Module):
    """
    This class implements a simple version of the gating logic described
    in "Spatio-Temporal Gated Transformers for Efficient Video
    Processing". This is intended to be used as an experimental
    baseline.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Options other than structure="row" have not
        yet been implemented
        """
        super().__init__()

        # Currently,
        assert structure == "row"

        self.first = True
        self.policy = None
        self.p = None

    def forward(self, c):
        if self.first:
            return self.forward_first(c)
        else:
            return self.forward_incremental(c)

    def forward_first(self, c):
        self.first = False
        self.p = c
        return c, None

    def forward_incremental(self, c):
        if self.count_mode:
            self.counts["gate_flops"] += c.numel()
        index = self.policy(c - self.p, dim=-1)
        c_tilde = c.gather(dim=-2, index=expand_row_index(index, c.shape))
        self.p = c
        return c_tilde, index

    def reset_self(self):
        self.first = True
        self.p = None


class TokenBuffer(nn.Module):
    """
    Defines a token buffer.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Whether tokens should be indexed along the
        last ("col") or second-to-last ("row") dimension
        """
        super().__init__()
        assert structure in ["row", "col"]
        self.structure = structure
        self.first = True
        self.b = None

    def forward(self, x, index):
        """
        Warning - the output is a direct reference to self.b (a state
        tensor).
        """
        if self.first:
            return self.forward_first(x)
        else:
            return self.forward_incremental(x, index)

    def forward_first(self, x):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.b = x.clone()
        return self.b

    def forward_incremental(self, x, index):
        """
        Forward pass after the first time step (incremental update).
        """
        if self.structure == "row":
            index = expand_row_index(index, self.b.shape)
            dim = -2
        else:
            index = expand_col_index(index, self.b.shape)
            dim = -1
        self.b.scatter_(dim=dim, index=index, src=x)
        return self.b

    def reset_self(self):
        self.first = True
        self.b = None


class TokenGate(nn.Module):
    """
    Defines a token gate.

    TokenGate.policy defines the token selection policy.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Whether tokens should be indexed along the
        last ("col") or second-to-last ("row") dimension
        """
        super().__init__()
        assert structure in ["row", "col"]
        self.structure = structure
        self.first = True
        self.policy = None
        self.p = None

    def forward(self, c, forced_index=None):
        """
        :param c: Warning - self.p (a state tensor) retains a direct
        reference to the last value of this input
        :param forced_index: A set of indices to force-update (instead
        of letting the policy decide)
        """
        if self.first:
            return self.forward_first(c)
        else:
            return self.forward_incremental(c, forced_index=forced_index)

    def forward_first(self, c):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.p = c
        return c, None

    def forward_incremental(self, c, forced_index=None):
        """
        Forward pass after the first time step (incremental update).
        """
        if self.count_mode:
            self.counts["gate_flops"] += self.p.numel()
        dim, expanded, index = self._apply_policy(c - self.p, forced_index)
        c_tilde = c.gather(dim=dim, index=expanded)
        self.p.scatter_(dim=dim, index=expanded, src=c_tilde)
        return c_tilde, index

    def _apply_policy(self, x, forced_index):
        dim = -2 if (self.structure == "row") else -1
        if forced_index is None:
            index = self.policy(x, dim=(-1 if (self.structure == "row") else -2))
        else:
            index = forced_index
        if self.structure == "row":
            expanded = expand_row_index(index, x.shape)
        else:
            expanded = expand_col_index(index, x.shape)
        return dim, expanded, index

    def reset_self(self):
        self.first = True
        self.p = None


class TokenDeltaGate(TokenGate):
    """
    Defines a token delta gate.
    """

    def __init__(self, structure="row"):
        """
        :param structure: Whether tokens should be indexed along the
        last ("col") or second-to-last ("row") dimension
        """
        super().__init__(structure=structure)

    def forward_first(self, c):
        c = super().forward_first(c)[0]
        return c, None, None

    def forward_incremental(self, c, forced_index=None):
        """
        :param c: Warning - self.p (a state tensor) retains a direct
        reference to the last value of this input
        :param forced_index: A set of indices to force-update (instead
        of letting the policy decide)
        """
        if self.count_mode:
            self.counts["gate_flops"] += self.p.numel()
        e = c - self.p
        dim, expanded, index = self._apply_policy(e, forced_index)
        c_tilde = c.gather(dim=dim, index=expanded)
        e_tilde = e.gather(dim=dim, index=expanded)
        self.p.scatter_(dim=dim, index=expanded, src=c_tilde)
        return c_tilde, e_tilde, index


class MatmulBuffer(nn.Module):
    """
    Defines a buffer for updating the query-key product.
    """
    def __init__(self):
        super().__init__()
        self.first = True
        self.product = None
        #self.matmul = CountedMatmul()

    def forward(self, q, k, index_q, index_k):
        """
        Warning - the output is a direct reference to self.product (a
        state tensor).
        """
        if self.first:
            return self.forward_first(q, k)
        else:
            return self.forward_incremental(q, k, index_q, index_k)

    def forward_first(self, q, k):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.product = q @ k
        return self.product

    def forward_incremental(self, q, k, index_q, index_k):
        """
        Forward pass after the first time step (incremental update).
        """
        q_tilde = q.gather(dim=-2, index=expand_row_index(index_q, q.shape))
        k_tilde = k.gather(dim=-1, index=expand_col_index(index_k, k.shape))
        self.product.scatter_(
            dim=-2,
            index=expand_row_index(index_q, self.product.shape),
            src=q_tilde @ k
        )
        self.product.scatter_(
            dim=-1,
            index=expand_col_index(index_k, self.product.shape),
            src=q @ k_tilde
        )
        return self.product

    def reset_self(self):
        self.first = True
        self.product = None


class MatmulDeltaAccumulator(nn.Module):
    """
    Defines a buffer for updating the +-value product.
    """
    def __init__(self):
        super().__init__()
        self.first = True
        self.product = None
        #self.matmul = CountedMatmul()

    def forward(self, a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde):
        """
        Warning - the output is a direct reference to self.product (a
        state tensor).
        """
        if self.first:
            return self.forward_first(a_n_tilde, v_n_tilde)
        else:
            return self.forward_incremental(
                a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde
            )

    def forward_first(self, a, v):
        """
        Forward pass on the first time step (flush).
        """
        self.first = False
        self.product = a @ v
        return self.product

    def forward_incremental(self, a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde):
        """
        Forward pass after the first time step (incremental update).
        """
        if self.count_mode:
            self.counts["accumulator_flops"] += (
                v_n_tilde.numel() + 2 * self.product.numel()
            )
        self.product += a_n_tilde @ v_delta_tilde
        self.product += a_delta_tilde  v_n_tilde - v_delta_tilde
        return self.product

    def reset_self(self):
        self.first = True
        self.product = None


class CountedAdd(nn.Module):
    """
    An addition operator that counts flops.
    """

    def forward(self, a, b, inplace=False):
        if inplace:
            a += b
            result = a
        else:
            result = a + b
        if self.count_mode:
            self.counts["add_flops"] += result.numel()
        return result

class CountedEinsum(nn.Module):
    """
    Einsum (Einstein summation) operation that counts flops.
    """

    def forward(self, equation, *operands):
        if self.count_mode:
            # There might be some cases here I haven't considered. But
            # this works fine for inner products.
            op_map = torch.einsum(equation, *[torch.ones_like(x) for x in operands])
            self.counts["einsum_flops"] += int(op_map.sum())
        return torch.einsum(equation, *operands)

class RelativePositionEmbedding(nn.Module):
    """
    Defines relative position embeddings.
    """
    def __init__(self, attention_size, embedding_size, head_dim, pool_size=None):
        """
        :param attention_size: The expected size of the attention window
        :param embedding_size: The size (in tokens) assumed for position
        embeddings
        :param head_dim: The dimensionality of each attention head
        :param pool_size: The pooling size (if self-attention pooling is
        being used - see the pool_size parameter to Block.
        """
        super().__init__()
        self.attention_size = attention_size
        self.embedding_size = embedding_size
        self.pool_size = pool_size
        self.y_embedding = nn.Parameter(
            torch.zeros(2 * embedding_size[0] - 1, head_dim)
        )
        self.x_embedding = nn.Parameter(
            torch.zeros(2 * embedding_size[1] - 1, head_dim)
        )
        self.add = CountedAdd()
        self.einsum = CountedEinsum()
        self.y_relative = None
        self.x_relative = None

    # This is based on the add_decomposed_rel_pos function here:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
    # noinspection PyTypeChecker
    def forward(self, x, q, inplace=True):
        a = self.attention_size

        # Unflatten the spatial dimensions.
        if self.pool_size is None:
            p = a
        else:
            p = (a[0] // self.pool_size[0], a[1] // self.pool_size[1])
        x = x.view(x.shape[:2] + a + p)
        q = q.view(q.shape[:2] + a + q.shape[-1:])

        # Apply the relative position embedding.
        if self.y_relative is None:
            # Cache y_relative and x_relative (assuming the weights
            # don't change, their values don't change between model
            # invocations).
            self.y_relative = self._get_relative(self.y_embedding, dim=0)
            self.x_relative = self._get_relative(self.x_embedding, dim=1)
        
        x += torch.einsum("abhwc,hkc->abhwk", q, self.y_relative).unsqueeze(dim=-1)
        x += torch  .einsum("abhwc,wkc->abhwk", q, self.x_relative).unsqueeze(dim=-2)

        # Re-flatten the spatial dimensions.
        x = x.view(x.shape[:2] + (prod(a), prod(p)))

        return x

    # This is a simplification of the get_rel_pos function here:
    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/utils.py
    def _get_relative(self, embedding, dim):
        range_0 = torch.arange(self.embedding_size[dim]).unsqueeze(dim=1)
        range_1 = torch.arange(self.embedding_size[dim]).unsqueeze(dim=0)
        relative = embedding[range_0 - range_1 + self.embedding_size[dim] - 1]
        if self.embedding_size != self.attention_size:
            relative = relative.transpose(0, 2).unsqueeze(dim=0)
            relative = func.interpolate(
                relative, self.attention_size, mode="bicubic", align_corners=False
            )
            relative = relative.squeeze(dim=0).transpose(0, 2)
        if self.pool_size is not None:
            relative = relative.transpose(1, 2)
            relative = func.avg_pool1d(relative, self.pool_size[dim])
            relative = relative.transpose(1, 2)
        return relative

    def reset_self(self):
        # Clear the cached values of x_relative and y_relative whenever
        # the model is reset (just in case new weights get loaded).
        self.y_relative = None
        self.x_relative = None

class Linear(nn.Module):
    """
    Linear transform operation that counts flops.
    """

    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        :param in_features: Dimensionality of input vectors
        :param out_features: Dimensionality of output vectors
        :param device: Transform matrix device
        :param dtype: Transform matrix data type
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        shape = (out_features, in_features)
        self.weight = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))

    def forward_bias(self, x):
        result = x + self.bias
        return result

    def forward_linear(self, x):
        return func.linear(x, self.weight)

    def forward(self, x):
        result = func.linear(x, self.weight, self.bias)
        return result


#### eventful
## dim,
## heads,
## mlp_ratio,
## drop_path_rate=0.0,
class EventfulAttention(torch.nn.Module):
    def __init__(self, 
                dim, 
                # key_dim, 
                input_size,
                num_heads=8,
                ats_fraction=None,
                relative_embedding_size=None,
                matmul_2_cast=None,
                pool_size=None,
                window_size=None,
                # EventfulTokenwiseBlock
                gate_before_ln=False,
                is_tokenwise=False,
            ):
        super().__init__()
        """
        :param dim: The number of dimensions in a token
        :param heads: The number of attention heads (None for no
        multi-headed attention)
        :param input_size: The expected size of the inputs in tokens
        :param mlp_ratio: The ratio of the MLP dimensionality to the
        token dimensionality
        :param ats_fraction: The fraction of tokens to retain if
        using Adaptive Token Sampling (ATS)
        :param drop_path_rate: Drop path ratio (for use when training)
        :param relative_embedding_size: The size (in tokens) assumed for
        relative position embeddings
        :param matmul_2_cast: Typecast for the attention-value product
        (None, "float16", or "bfloat16"). Helps save some memory when
        using an A-gate, without a noticeable impact on accuracy.
        :param pool_size: Pooling ratio to use with self-attention
        pooling.
        :param window_size: Self-attention window size (None to use
        global, non-windowed attention).
        """
        # (h, w)
        ### assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads

        self.is_tokenwise = is_tokenwise
        if is_tokenwise: pool_size = None

        #_defaults:
        # - "vitdet_b_coco.yml"
        #model:
        # classes: 30
        # detectron2_config: "configs/detectron/vitdet_b_vid.py"
        
        #self.scale = key_dim ** -0.5
        #self.key_dim = key_dim
        #self.nh_kd = nh_kd = key_dim * num_heads
        #self.d = int(attn_ratio * key_dim)
        #self.dh = int(attn_ratio * key_dim) * num_heads
        #self.attn_ratio = attn_ratio
        #h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim, eps=LN_EPS)
        
        #self.qkv = nn.Linear(in_features=dim, out_features=dim * 3)
        # self.qkv = nn.Linear(dim, h)
        self.qkv = Linear(in_features=dim, out_features=dim * 3)
        self.proj = nn.Linear(self.dh, dim)
        # self.projection = CountedLinear(in_features=dim, out_features=dim)

        ######################################################
        # unused now
        #points = list(itertools.product(
        #    range(resolution[0]), range(resolution[1])))
        #N = len(points)
        #attention_offsets = {}
        #idxs = []
        #for p1 in points:
        #    for p2 in points:
        #        offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
        #        if offset not in attention_offsets:
        #            attention_offsets[offset] = len(attention_offsets)
        #        idxs.append(attention_offsets[offset])
        #self.attention_biases = torch.nn.Parameter(
        #    torch.zeros(num_heads, len(attention_offsets)))
        #self.register_buffer('attention_bias_idxs',
        #                     torch.LongTensor(idxs).view(N, N),
        #                     persistent=False)
        ######################################################

        #self.num_heads = num_heads
        self.input_size = tuple(input_size)
        
        if ats_fraction is not None:
            ass is None
            assert window_size is None
            assert not (ats_fraction < 0.0 or ats_fraction > 1.0)
        # assert not (drop_path_rate < 0.0 or drop_path_rate > 1.0)
        assert matmul_2_cast in [None, "float16", "bfloat16"]
        self.ats_fraction = ats_fraction
        self.last_ats_indices = None
        self.matmul_2_cast = matmul_2_cast

        if pool_size is None: self.pool_size = None
        else: self.pool_size = numeric_tuple(pool_size, length=2)

        if window_size is None:
            self.window_size = None
            attention_size = input_size
        else:
            self.window_size = numeric_tuple(window_size, length=2)
            attention_size = self.window_size
            if relative_embedding_size is not None: relative_embedding_size = self.window_size
        self.scale = sqrt(dim // num_heads)

        if relative_embedding_size is not None:
            self.relative_position = RelativePositionEmbedding(
                attention_size,
                relative_embedding_size,
                dim // num_heads,
                pool_size=self.pool_size,
            )
        else:
            self.relative_position = None
        
        ### EventfulTokenwiseBlock
        self.gate_before_ln = gate_before_ln
        # TokenGate = SimpleSTGTGate if stgt else TokenGate
        self.qkv_gate = TokenGate()
        self.qkv_accumulator = TokenBuffer()
        self.projection_gate = TokenGate()
        self.projection_accumulator = TokenBuffer()

        ### EventfulMatmul1Block
        # self._pool_index assumes that the input size is divisible by
        # the pooling size.
        if self.pool_size is not None:
            assert all(s % p == 0 for s, p in zip(self.input_size, self.pool_size))

        # This class only supports non-windowed attention for now.
        assert self.window_size is None

        self.matmul_accumulator_1 = MatmulBuffer()

        ## EventfulBlock
        self.v_gate = TokenDeltaGate()
        self.matmul_gate = TokenDeltaGate(structure="col")
        self.matmul_accumulator_2 = MatmulDeltaAccumulator()

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.register_buffer('ab',
                                 self.attention_biases[:, self.attention_bias_idxs],
                                 persistent=False)

    def _forward_pre_attention(self, x):
        skip_1 = x

        # Gate-accumulator block 1
        if self.gate_before_ln:
            x, index = self.qkv_gate(x)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x, index = self.qkv_gate(x)
        x = self.qkv(x)
        return skip_1, x, index

    def _partition_heads(self, x):
        # (batch, token, dim)

        x = x.view(x.shape[:-1] + (3, self.num_heads, x.shape[-1] // (3 * self.num_heads)))
        q, k, v = x.permute(2, 0, 3, 1, 4)
        # (batch, heads, token, dim / heads)

        return q, k, v

    def _pool_index(self, index):
        if (self.pool_size is None) or (index is None):
            return index
        width = self.input_size[1]
        index_y = index.div(width, rounding_mode="floor")
        index_x = index.remainder(width)
        index_y = index_y.div(self.pool_size[0], rounding_mode="floor")
        index_x = index_x.div(self.pool_size[1], rounding_mode="floor")
        index = index_y * (width // self.pool_size[1]) + index_x

        # Calling .unique() still works if there are multiple items in
        # the batch. However, the output size along dim=-1 will be the
        # largest of the individual output sizes. This could result in
        # some redundant downstream computation.
        index = index.unique(dim=-1)
        return index

    def _pool_tokens(self, x):
        # (batch, heads, token, dim)

        if self.pool_size is None:
            return x
        w = self.input_size if (self.window_size is None) else self.window_size
        s = x.shape

        # Can't use x.view here because of the permutation in
        # _partition_heads.
        x = x.reshape((-1,) + w + x.shape[-1:])
        # (batch * heads, token_y, token_x, dim)

        x = x.permute(0, 3, 1, 2)
        x = func.avg_pool2d(x, self.pool_size)
        # (batch * heads, dim, token_y, token_x)

        x = x.permute(0, 2, 3, 1)
        # (batch * heads, token_y, token_x, dim)

        x = x.view(s[:-2] + (-1,) + s[-1:])
        # (batch, heads, token, dim)

        return x

    def _forward_matmul_1(self, x):
        x, index = x
        q, k, v = self._partition_heads(x)
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)
        index_k = self._pool_index(index)

        # See comment in Block._forward_attention.
        x = self.matmul_accumulator_1(
            q / self.scale, k.transpose(-2, -1), index, index_k
        )

        if self.relative_position is not None:
            # We need inplace=False because x is a direct reference to
            # an accumulator state.
            x = self.relative_position(x, q, inplace=False)
        x = x.softmax(dim=-1)
        return x, v, index_k

    def _cast_matmul_2(self, x, v):
        old_dtype = x.dtype
        if self.matmul_2_cast is not None:
            dtype = getattr(torch, self.matmul_2_cast)
            x = x.to(dtype)
            v = v.to(dtype)
        return x, v, old_dtype

    def _stabilize_ats_indices(self, ats_indices):
        ats_indices = ats_indices.sort(dim=-1)[0]
        if self.last_ats_indices is None:
            return ats_indices

        # Faster on the CPU
        new_indices = ats_indices.flatten(end_dim=-2).cpu()
        old_indices = self.last_ats_indices.flatten(end_dim=-2).cpu()
        stabilized = old_indices.clone()
        for i in range(new_indices.shape[0]):
            old_not_in_new = torch.isin(old_indices[i], new_indices[i], invert=True)
            new_not_in_old = torch.isin(new_indices[i], old_indices[i], invert=True)
            stabilized[i, old_not_in_new] = new_indices[i, new_not_in_old]
        return stabilized.to(ats_indices.device).view(ats_indices.shape)

    # A simple version of the method from
    # "Adaptive Token Sampling for Efficient Vision Transformers"
    # (Fayyaz et al., ECCV 2022)
    # For now we just use the top-k version of ATS (select the tokens
    # with the k highest scores). Using CDF-based token sampling should
    # also be possible, but it would be more complex to implement (we
    # would need a mechanism for masking the K' < K active tokens in
    # gates and buffers).
    def _adaptive_token_sampling(self, a, v):
        if self.ats_fraction is None:
            return a, None

        class_scores = a[..., 0]
        raw_scores = class_scores * torch.linalg.vector_norm(v[...], dim=-1)
        scores = raw_scores / raw_scores[..., 1:].sum(dim=-1, keepdim=True)

        # Always select the class token.
        scores[..., 0] = float("inf")

        # Sum scores over heads.
        scores = scores.sum(dim=-3)

        # Add +1 for the class token
        n_select = int(self.ats_fraction * (scores.shape[-1] - 1)) + 1

        # Select the k tokens with the highest scores.
        ats_indices = scores.topk(n_select, sorted=False)[1]

        # Sort the token indices (for stabilization). This seems to
        # work pretty well, although we could probably come up with
        # better/more sophisticated. E.g., we could try to find the
        # permutation of indices that minimized some norm between the
        # previous and current ats_indices.
        ats_indices = self._stabilize_ats_indices(ats_indices)
        self.last_ats_indices = ats_indices

        return (
            a.gather(dim=-2, index=expand_row_index(ats_indices, a.shape)),
            ats_indices,
        )

    @staticmethod
    def _recombine_heads(x):
        # (batch, heads, token, dim / heads)

        # Can't use x.view here because of the permutation.
        x = x.permute(0, 2, 1, 3)
        x_reshaped = x.reshape(x.shape[:-2] + (-1,))
        # (batch, token, dim)

        # We assume that x.reshape actually copies the data. We can run
        # into problems if this is not the case, i.e., we may end up
        # with a gate being passed a raw reference to an accumulator
        # state. For an example, see EventfulMatmul1Block.
        assert x.data_ptr() != x_reshaped.data_ptr()
        x = x_reshaped

        return x

    def _uncast_matmul_2(self, x, old_dtype):
        if self.matmul_2_cast is not None:
            x = x.to(old_dtype)
        return x

    def _forward_attention(self, a):
        a, v, index_k = self._forward_matmul_1(a)

        a, v, old_dtype = self._cast_matmul_2(a, v)
        a, ats_indices = self._adaptive_token_sampling(a, v)
        if not self.matmul_2_cast:
            # We clone v here because it may be a direct reference to
            # self.qkv_accumulator.a.
            v = v.clone()
        v_n_tilde, v_delta_tilde, index_v = self.v_gate(v, forced_index=index_k)
        a_n_tilde, a_delta_tilde, _ = self.matmul_gate(a, forced_index=index_v)
        a = self.matmul_accumulator_2(
            a_n_tilde, v_n_tilde, a_delta_tilde, v_delta_tilde
        )

        a = self._recombine_heads(a)
        a = self._uncast_matmul_2(a, old_dtype)
        return a, ats_indices

    @staticmethod
    def _gather_ats_skip(skip_1, ats_indices):
        if ats_indices is None:
            return skip_1
        else:
            return skip_1.gather(
                dim=-2, index=expand_row_index(ats_indices, skip_1.shape)
            )

    def reset_self(self):
        self.last_ats_indices = None

    def _forward_post_attention_et(self, x, skip_1):
        # Gate-accumulator block 2
        x, index = self.projection_gate(x)
        x = self.projection(x)
        x = self.projection_accumulator(x, index)

        ###### this is what we add to the other network instead
        x = self.add(self.drop_path(x), skip_1)
        skip_2 = x

        # Gate-accumulator block 3
        if self.gate_before_ln:
            x, index = self.mlp_gate(x)
            x = self.mlp_layer_norm(x)
        else:
            x = self.mlp_layer_norm(x)
            x, index = self.mlp_gate(x)
        x = self._forward_mlp(x)
        x = self.mlp_accumulator(x, index)
        x = self.add(self.drop_path(x), skip_2)

        return x

    def _forward_post_attention(self, x):
        # Gate-accumulator block 2
        x, index = self.projection_gate(x)
        x = self.proj(x)
        x = self.projection_accumulator(x, index)
        return x

    def forward_eventful(self, x):
        skip_1, x, index = self._forward_pre_attention(x)
        x = self.qkv_accumulator(x, index)
        x, ats_indices = self._forward_attention((x, index))
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        x = self._forward_post_attention(x, skip_1)
        return x, skip_1

    ####################################################
    ######## TOKENWISE ONLY
    ####################################################

    def _recombine_windows(self, x):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        s = self.input_size
        total_h = p[0] + s[0]
        total_w = p[1] + s[1]
        # (batch * window, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(-1, total_h // d[0], total_w // d[1], d[0], d[1], x.shape[-1])
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Recombine the window partitions. Can't use x.view here because
        # of the transpose.
        x = x.transpose(-3, -4)
        x = x.reshape(-1, total_h, total_w, x.shape[-1])
        # (batch, height, width, dim)

        # Remove padding.
        if any(p):
            x = x[:, : s[0], : s[1]]
            # (batch, height, width, dim)

        # Re-flatten the spatial dimensions.
        x = x.flatten(start_dim=1, end_dim=2)
        # (batch, token, dim)

        return x

    def _compute_window_padding(self):
        pad_h = -self.input_size[0] % self.window_size[0]
        pad_w = -self.input_size[1] % self.window_size[1]
        return pad_h, pad_w

    def _partition_windows(self, x, in_qkv_domain):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        # (batch, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(x.shape[:1] + self.input_size + x.shape[2:])
        # (batch, height, width, dim)

        if any(p):
            s = x.shape
            pad_tensor = torch.zeros(
                (1,) * (x.ndim - 1) + s[-1:], dtype=x.dtype, device=x.device
            )

            # The attention computation expects padded tokens to equal
            # _forward_qkv(zero). If x has already been mapped to the
            # QKV domain, then we need to transform the padded zero
            # values to the QKV domain. Only the bias portion of the
            # linear transform has an effect on the zero padding vector.
            if in_qkv_domain:
                pad_tensor = self.qkv.forward_bias(pad_tensor)

            # Pad to a multiple of the window size.
            # func.pad seems broken (see the comments in pad_to_size).
            # In the meantime we'll use pad_to_size.
            # x = func.pad(x, (0, 0, 0, p[1], 0, p[0]))
            x = pad_to_size(x, (s[-3] + p[0], s[-2] + p[1], s[-1]), pad_tensor)
            # (batch, height, width, dim)

        # Partition into windows.
        s = x.shape
        x = x.view(-1, s[-3] // d[0], d[0], s[-2] // d[1], d[1], s[-1])
        x = x.transpose(-3, -4)
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Re-flatten the spatial dimensions. Can't use x.view here
        # because of the transpose.
        x = x.reshape(-1, prod(d), s[-1])
        # (batch * window, token, dim)

        return x

    def _forward_attention_tokenwise(self, x):
        # (batch, token, dim)

        # Partition the windows and attention heads. _window_partition
        # is a noop if self.window_size is None. Windows are arranged
        # along the batch dimension.
        x = self._partition_windows(x, in_qkv_domain=True)
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Token pooling is a noop if self.pool_size is None.
        # I think we just don't even do this op.
        #k = self._pool_tokens(k)
        #v = self._pool_tokens(v)

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = q / self.scale @ k.transpose(-2, -1)
        if self.relative_position is not None:
            x = self.relative_position(x, q)
        x = x.softmax(dim=-1)

        # Adaptive token sampling is a noop if self.ats_fraction is None.
        x, ats_indices = self._adaptive_token_sampling(x, v)

        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = x @ v
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)
        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        # (batch, token, dim)

        return x, ats_indices

    def forward_tokenwise(self, x):
        skip_1, x, index = self._forward_pre_attention_tokenwise(x)
        x = self.qkv_accumulator(x, index)
        x, ats_indices = self._forward_attention_tokenwise(x)
        skip_1 = self._gather_ats_skip(skip_1, ats_indices)
        x = self._forward_post_attention_tokenwise(x, skip_1)
        return x, skip_1

    def _forward_post_attention_tokenwise(self, x, skip_1):
        # Gate-accumulator block 2
        x, index = self.projection_gate(x)
        x = self.projection(x)
        x = self.projection_accumulator(x, index)

        #x = self.add(self.drop_path(x), skip_1)
        #skip_2 = x
        ## Gate-accumulator block 3
        #if self.gate_before_ln:
        #    x, index = self.mlp_gate(x)
        #    x = self.mlp_layer_norm(x)
        #else:
        #    x = self.mlp_layer_norm(x)
        #    x, index = self.mlp_gate(x)
        #x = self._forward_mlp(x)
        #x = self.mlp_accumulator(x, index)
        #x = self.add(self.drop_path(x), skip_2)

        return x, skip_1

    def _forward_pre_attention_tokenwise(self, x):
        skip_1 = x

        # Gate-accumulator block 1
        if self.gate_before_ln:
            x, index = self.qkv_gate(x)
            x = self.input_layer_norm(x)
        else:
            x = self.input_layer_norm(x)
            x, index = self.qkv_gate(x)
        x = self.qkv(x)
        return skip_1, x, index

    def forward(self, x):
        if self.is_tokenwise: return self.forward_tokenwise(x)
        else: self.forward_eventful(x)

class EvenfulTinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self, 
        dim, 
        input_size, 
        num_heads=8, 
        window_size=7,
        mlp_ratio=4., 
        drop=0., 
        drop_path=0.,
        local_conv_size=3,
        activation=nn.GELU,

        ats_fraction=None,
        relative_embedding_size=None,
        matmul_2_cast=None,
        pool_size=None,
        gate_before_ln=False,
        is_tokenwise=False

    ):
        super().__init__()
        self.dim = dim
        self.input_size = input_size
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)# window_size
        self.attn = EventfulAttention(
                dim, input_size, num_heads, ats_fraction,
                relative_embedding_size, matmul_2_cast,
                pool_size, window_size, gate_before_ln,
                is_tokenwise,
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

        # add accumulators
        self.mlp_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp_gate = TokenGate()
        self.mlp_accumulator = TokenBuffer()

    def forward(self, x):
        H, W = self.input_size
        B, L, C = x.shape
        if H == self.window_size and W == self.window_size:
            x, skip_1 = self.attn(x)
        else:
            x, skip_1 = self.attn.forward_tokenwise(x)

        x = self.drop_path(x) + skip_1
        skip_2 = x

        # from original TinyVit
        # conv intbetween drops
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        # Gate-accumulator block 3
        if self.gate_before_ln:
            x, index = self.mlp_gate(x)
            x = self.mlp_layer_norm(x)
        else:
            x = self.mlp_layer_norm(x)
            x, index = self.mlp_gate(x)
        # use original variant
        x = self.mlp(x)
        x = self.mlp_accumulator(x, index)

        x = self.drop_path(x) + skip_2

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_size={self.input_size}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

class EventfulBasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self, 
                 dim, 
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4., 
                 drop=0.,
                 drop_path=0., 
                 downsample=None, 
                 use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 # these are going to have to be manually set
                 # some how...
                 ats_fraction=None,
                 relative_embedding_size=None,
                 matmul_2_cast=None,
                 pool_size=None,
                 gate_before_ln=False,
                 is_tokenwise=False,
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            EvenfulTinyViTBlock(dim=dim, input_size=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         ats_fraction=ats_fraction,
                         relative_embedding_size=relative_embedding_size,
                         matmul_2_cast=matmul_2_cast,
                         pool_size=pool_size,
                         gate_before_ln=gate_before_ln,
                         is_tokenwise=is_tokenwise
                        )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

'''
    img_size=img_size[0], 
    in_chans=3, # in_channels
    num_classes=1000,
    embed_dims=[64, 128, 160, 320],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
'''

class EventfulTinyViT(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 in_chans=3, 
                 num_classes=1000,
                 embed_dims=[64, 128, 160, 320], 
                 depths=[2, 2, 6, 2],
                 num_heads=[2, 4, 5, 10],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.0,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=0.8,
                 ats_fraction=None,
                 relative_embedding_size=[64,64], # might be smaller
                 matmul_2_cast=None,
                 pool_size=2,
                 gate_before_ln=False

                 ):
        super().__init__()

        '''
        tinyvit
        window_indices: 


        :vitdet_b_coco:
        model:
        classes: 80
        detectron2_config: "configs/detectron/vitdet_b_coco.py"
        input_shape: [3, 1024, 1024]
        normalize_mean: [123.675, 116.28, 103.53]
        normalize_std: [58.395, 57.12, 57.375]
        output_channels: 256
        patch_size: [16, 16]
        scale_factors: [4.0, 2.0, 1.0, 0.5]
        backbone_config:
            depth: 12
            position_encoding_size: [14, 14]
            # i think these are where h/w != //0
            window_indices: [0, 1, 3, 4, 6, 7, 9, 10]
            # lets get these window indices
            block_config:
            dim: 768
            relative_embedding_size: [64, 64]
            heads: 12
            mlp_ratio: 4
            window_size: [14, 14]
        model:
        backbone_config:
            block_config:
                pool_size: 2
            #TODO: how to set this variable iteratively?
            windowed_overrides:
                pool_size: null
        token_top_k: [512]
        for k in config.get("token_top_k", []):
            set_policies(model, TokenNormTopK, k=k)
            do_evaluation(f"Token top k={k}")

        for i in range(depth):
            block_class_i = block_class # EventfulBlock
            block_config_i = block_config.copy()
            if i in window_indices: # EventfulTokenwiseBlock
                if windowed_class is not None:
                    block_class_i = windowed_class
                if windowed_overrides is not None:
                    block_config_i =  block_config_i or windowed_overrides # |= windowed_overrides 
            else:
                block_config_i["window_size"] = None
            self.blocks.append(
                getattr(blocks, block_class_i)(input_size=input_size, **block_config_i)
            )
        '''

        self.img_size=img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        # window_indices: [0, 1, 3, 4, 6, 7, 9, 10] ? [0, 1, 2, 3] for tinyvit
        indicies = []
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # is_tokenwise = False
            kwargs = dict(dim=embed_dims[i_layer],
                        input_resolution=(patches_resolution[0] // (2 ** (i_layer-1 if i_layer == 3 else i_layer)),
                                patches_resolution[1] // (2 ** (i_layer-1 if i_layer == 3 else i_layer))),
                        #   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                        #                     patches_resolution[1] // (2 ** i_layer)),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          ats_fraction=ats_fraction,
                          relative_embedding_size=relative_embedding_size,
                          is_tokenwise=False,
                          pool_size=pool_size, # default poolsize
                          matmul_2_cast=matmul_2_cast,
                          gate_before_ln=gate_before_ln
                        )
            if kwargs['input_resolution'][0] != kwargs['window_size']:
                indicies.append(i_layer)
                 #is_tokenwise = True
                kwargs['is_tokenwise'] = True

            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = EventfulBasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )
    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]
        #print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B,_,C=x.size()
        x = x.view(B, 64, 64, C)
        x=x.permute(0, 3, 1, 2)
        x=self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.norm_head(x)
        #x = self.head(x)
        return x

    def forward_features_unet(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1
        hidden_states = []
        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            hidden_states.append(x)
        B,_,C=x.size()
        # 64 if self.img_size == 1024 else 32
        out = self.img_size // 16 #-> 1024:64, 512:32, 256:16
        x = x.view(B, out, out, C) ## this //2 if 
        x=x.permute(0, 3, 1, 2)
        x=self.neck(x)
        return x, hidden_states

    def forward_unet(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states
    
class SAMUNETR(nn.Module):
    
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        feature_size: int = 20,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        spatial_dims: int = 2,
        embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        vit_patch_size = 16,
        encoder_out_channels = 256,
        pretrained="./weights/mobile_sam_encoder.pt",
        trainable_encoder=True
        
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            spatial_dims: number of spatial dims.
            embed_dim: embeding dimensions for sam encoder.
            encoder_depth= number of attention blocks for the encoder
            encoder_num_heads: number of attention heads.
            encoder_global_attn_indexes: Indexes for blocks using global attention.
            vit_patch_size: Patch size
            encoder_out_channels: number of output channels for the encoder.
            pretrained: Wheter to use pretrained model or not
        """

        super().__init__()
        print('Using SAMUNETR_V2')

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(vit_patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if embed_dim % feature_size != 0:
            raise ValueError("embed_dim should be divisible by feature_size.")

        self.normalize = normalize

        # Image Encoder using Vision Transformer (ViT)
        '''
        self.image_encoder_vit=ImageEncoderViT(
                    depth=encoder_depth,
                    embed_dim=embed_dim,
                    img_size=img_size[0],
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=encoder_out_channels,
                    in_chans=in_channels
        )#.to('cuda')
        '''
        # going to save the encoder alone
        self.image_encoder_vit = EventfulTinyViT(
                img_size=img_size[0], 
                in_chans=3, # in_channels
                num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            )

        # img_size: Tuple[int, int] = to_2tuple(resolution)
        # self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        # vit_patch_size = self.patches_resolution

        if pretrained:
            model_state=torch.load(pretrained)

            if img_size!=1024:
                pass

            self.image_encoder_vit.load_state_dict(state_dict=model_state)
            
            if not trainable_encoder:
                for param in self.image_encoder_vit.parameters():
                    param.requires_grad = False
                print('Image encoder no trainable')
        
        # Encoder blocks        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=160,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=320,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=320,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=True,
            res_block=True,
        )
        
        # Decoder blocks
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=320,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [embed_dim]


    def proj_feat(self, x, proj_view_shape=None):
        new_view = [x.size(0)] + proj_view_shape if proj_view_shape else self.proj_view_shape 
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    # reduce tensor size (weight distill)
    def resize_tensor(self,tensor, output_shape):
        # check if the input tensor has a batch dimension
        has_batch_dim = tensor.dim() == 4 
        # add a batch dimension if needed
        if not has_batch_dim:
            tensor=tensor.view(1,*tensor.shape,1)
        # reshape the tensor to (batch_size, channel, *spatial)
        tensor = tensor.permute(0, -1, *range(1, len(tensor.shape) - 1))
        # resize the tensor using interpolate
        resized_tensor = F.interpolate(tensor, size=output_shape, mode='bilinear')
        
        resized_tensor=resized_tensor.permute(0,2,3,1)
        # remove the batch dimension if needed
        if not has_batch_dim:
            resized_tensor = resized_tensor.squeeze()

        return resized_tensor
    

    def forward(self, x_in):
        x, hidden_states_out = self.image_encoder_vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(self.proj_feat(x2,  list(self.feat_size) + [160]))
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(self.proj_feat(x3,  list(self.feat_size) + [320]))
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(self.proj_feat(x4,  list(self.feat_size) + [320]))
        dec4 = self.proj_feat(hidden_states_out[2],  list(self.feat_size) + [320])
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: EventfulTinyViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

