import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V



def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):  # Patch Partition + Linear Embedding
    def __init__(self, patch_size=4, in_channels=3, emb_dim=96):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)  # 4x4卷积实现Patch Partition

    def forward(self, x):
        # (B,C,H,W)
        x = self.conv(x)
        _, _, H, W = x.shape
        x = rearrange(x, "B C H W -> B (H W) C")  # Linear Embedding
        return x, H, W


class MLP(nn.Module):  # MLP
    def __init__(self, in_dim, hidden_dim=None, drop_ratio=0.):
        super(MLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim * 4  # linear的hidden_dims默认为in_dims的4倍

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        # Linear + GELU + Dropout + Linear + Dropout
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class WindowMultiHeadSelfAttention(nn.Module):  # W-MSA / SW-MSA
    def __init__(self, dim, window_size, num_heads,
                 attn_drop_ratio=0., proj_drop_ratio=0.):
        super(WindowMultiHeadSelfAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        # 创建Relative position bias matrix，其参数可训练，根据Relative position index取其中的值作为B
        self.relative_position_bias_matrix = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        # 使用register_buffer，使得relative_position_index可以随model.state_dict()保存，并可以随model.cuda()加载至GPU
        self.register_buffer("relative_position_index", self._get_relative_position_index())

    def _get_relative_position_index(self):  # 创建Relative position index
        coords = torch.flatten(
            torch.stack(
                torch.meshgrid([torch.arange(self.window_size), torch.arange(self.window_size)], indexing="ij"), dim=0
            ), 1
        )
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords += self.window_size - 1
        relative_coords[0, :, :] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(0)
        return relative_position_index.view(-1)

    def forward(self, x, mask=None):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "B P (C H d) -> C B H P d", C=3, H=self.num_heads, d=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k = rearrange(k, "B H P d -> B H d P")
        # Attention(Q, K, V ) = softmax(QKT/dk)V （T表示转置)
        attn = torch.matmul(q, k) * self.head_dim ** -0.5  # QKT/dk

        bias = self.relative_position_bias_matrix[self.relative_position_index]
        bias = rearrange(bias, "(P1 P2) H -> 1 H P1 P2", P1=self.window_size ** 2, P2=self.window_size ** 2)
        attn += bias  # QKT/dk + B

        if mask is not None:
            # 如果mask不为None，对attn进行加和，使得在原图上不相邻的token对应的attn-100，经过softmax后趋近于0
            attn = rearrange(attn, "(B NW) H P1 P2 -> B NW H P1 P2", NW=mask.shape[0])
            mask = rearrange(mask, "NW P1 P2 -> 1 NW 1 P1 P2")
            attn += mask
            attn = rearrange(attn, "B NW H P1 P2 -> (B NW) H P1 P2")

        attn = F.softmax(attn)  # softmax(QKT/dk + B)

        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)  # softmax(QKT/dk + B)V
        x = rearrange(x, "B H P d -> B P (H d)")
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):  # Swin Transformer Block
    def __init__(self, dim, num_heads, window_size=7, shift=True,
                 attn_drop_ratio=0., proj_drop_ratio=0., drop_path_ratio=0.):
        super(SwinTransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0.  # 不进行shift时，shift_size取0
        self.layernorm1 = nn.LayerNorm(dim)
        self.attn = WindowMultiHeadSelfAttention(dim, self.window_size, self.num_heads,
                                                 attn_drop_ratio=attn_drop_ratio,
                                                 proj_drop_ratio=proj_drop_ratio)
        self.droppath = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.layernorm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def _create_mask(self, H, W, device):  # 创建mask
        mask = torch.zeros((1, 1, H, W), device=device)
        slices = (slice(0, -self.window_size),
                  slice(-self.window_size, -self.shift_size),
                  slice(-self.shift_size, None))
        count = 0
        for h in slices:
            for w in slices:
                mask[:, :, h, w] = count
                count += 1

        mask = rearrange(mask, "1 1 (H Hs) (W Ws) -> (H W) (Hs Ws)", Hs=self.window_size, Ws=self.window_size)
        attn_mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.))  # 在原图上不相邻的token，mask为-100.
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.))  # 在原图上相邻的token，mask为0.
        return attn_mask

    def forward(self, input: tuple):
        x, H, W = input
        shortcut = x
        x = self.layernorm1(x)
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        if self.shift_size > 0.:  # 如果偏移量shift_size>0.，则对x进行偏移，同时创建对应的mask
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            mask = self._create_mask(H, W, device=x.device)
        else:
            mask = None
        num_windows = (x.shape[2] // self.window_size, x.shape[3] // self.window_size)
        # x = rearrange(x, "B C (H Hs) (W Ws) -> (B H W) C Hs Ws", Hs=self.window_size, Ws=self.window_size)
        x = rearrange(x, "B C (H Hs) (W Ws) -> (B H W) (Hs Ws) C", Hs=self.window_size, Ws=self.window_size)
        x = self.attn(x, mask)
        # x = rearrange(x, "(B H W) C Hs Ws -> B C (H Hs) (W Ws)", Hs=self.window_size, Ws=self.window_size)
        x = rearrange(x, "(B H W) (Hs Ws) C -> B C (H Hs) (W Ws)", H=num_windows[0], W=num_windows[1],
                      Hs=self.window_size, Ws=self.window_size)
        if self.shift_size > 0.:  # 如果偏移量shift_size>0.，则将偏移过的x调整回原来的位置
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = rearrange(x, "B C H W -> B (H W) C", H=H, W=W)
        x = shortcut + self.droppath(x)  # 残差连接
        shortcut = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = shortcut + self.droppath(x)  # 残差连接
        return x, H, W


class PatchMerging(nn.Module):  # Patch Merging
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.layernorm = nn.LayerNorm(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, input: tuple):
        # (B,L,C) --> (B,C,H,W) --> (B,4*C,H/2,W/2) --> (B,L/4,4*C) --> (B,L/4,2*C)
        x, H, W = input
        x = rearrange(x, "B (H W) C -> B C H W", H=H, W=W)
        x = torch.cat([x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2], x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]], dim=1)
        _, _, H, W = x.shape
        x = rearrange(x, "B C H W -> B (H W) C")
        x = self.layernorm(x)
        x = self.linear(x)
        return x, H, W


class SwinHead(nn.Module):  # Swin Head，分类任务的Head
    def __init__(self, dim, num_classes):
        super(SwinHead, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.mlphead = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.layernorm(x)
        x = rearrange(x, "B L C -> B C L")
        x = self.avgpool(x)
        return self.mlphead(x.squeeze())

class HierarchyAttention(nn.Module):
    def __init__(self, channel_list=[96, 192, 384, 768], ratio=16):
        super(HierarchyAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.input_channel = sum(channel_list)
        self.fc = nn.Sequential(
            nn.Linear(self.input_channel, self.input_channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.input_channel // ratio, channel_list[-1], bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        b1, c1, _, _ = x1.size()
        y1 = self.avg_pool(x1)

        b2, c2, _, _ = x2.size()
        y2 = self.avg_pool(x2)

        b3, c3, _, _ = x3.size()
        y3 = self.avg_pool(x3)

        b4, c4, _, _ = x4.size()
        y4 = self.avg_pool(x4)

        y = torch.concat([y1, y2, y3, y4], dim=1)
        y = y.view(b1, c1 + c2 + c3 + c4)
        y = self.fc(y).view(b1, c4, 1, 1)
        return x4 * y



class SwinTransformer(nn.Module):  # Swin Transformer
    def __init__(self, dims=(96, 192, 384, 768), num_blocks=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 num_classes=1000,
                 pos_drop_ratio=0., attn_drop_ratio=0., proj_drop_ratio=0., drop_path_ratio_max=0.1):
        super(SwinTransformer, self).__init__()
        self.patchembedding = PatchEmbedding(emb_dim=dims[0])
        self.pos_drop = nn.Dropout(pos_drop_ratio)
        # #drop path ratio从0递增至drop_path_ratio_max
        drop_path_ratio = [i.item() for i in torch.linspace(0, drop_path_ratio_max, sum(num_blocks))]
        self.blocks1 = nn.Sequential(
            *[SwinTransformerBlock(dims[0], num_heads[0], shift=(i % 2 != 0),
                                   attn_drop_ratio=attn_drop_ratio,
                                   proj_drop_ratio=proj_drop_ratio,
                                   drop_path_ratio=drop_path_ratio[i + sum(num_blocks[:0])])
              for i in range(num_blocks[0])]
        )
        self.patchmerging2 = PatchMerging(dims[0])
        self.blocks2 = nn.Sequential(
            *[SwinTransformerBlock(dims[1], num_heads[1], shift=(i % 2 != 0),
                                   attn_drop_ratio=attn_drop_ratio,
                                   proj_drop_ratio=proj_drop_ratio,
                                   drop_path_ratio=drop_path_ratio[i + sum(num_blocks[:1])])
              for i in range(num_blocks[1])]
        )
        self.patchmerging3 = PatchMerging(dims[1])
        self.blocks3 = nn.Sequential(
            *[SwinTransformerBlock(dims[2], num_heads[2], shift=(i % 2 != 0),
                                   attn_drop_ratio=attn_drop_ratio,
                                   proj_drop_ratio=proj_drop_ratio,
                                   drop_path_ratio=drop_path_ratio[i + sum(num_blocks[:2])])
              for i in range(num_blocks[2])]
        )
        self.patchmerging4 = PatchMerging(dims[2])
        self.blocks4 = nn.Sequential(
            *[SwinTransformerBlock(dims[3], num_heads[3], shift=(i % 2 != 0),
                                   attn_drop_ratio=attn_drop_ratio,
                                   proj_drop_ratio=proj_drop_ratio,
                                   drop_path_ratio=drop_path_ratio[i + sum(num_blocks[:3])])
              for i in range(num_blocks[3])]
        )
        self.hierarchy_attention = HierarchyAttention()
        self.skattention = SKAttention(channel=768)
        self.head = SwinHead(dims[-1], num_classes)

    def forward(self, x):
        b, _, _, _ = x.size()
        # Patch Partition + Stage1
        x, H, W = self.patchembedding(x)
        x = self.pos_drop(x)
        x, H, W = self.blocks1((x, H, W))
        x1 = x.reshape(b, 96, 56, 56)

        # Stage2
        x, H, W = self.patchmerging2((x, H, W))
        x, H, W = self.blocks2((x, H, W))
        x2 = x.reshape(b, 192, 28, 28)

        # Stage3
        x, H, W = self.patchmerging3((x, H, W))
        x, H, W = self.blocks3((x, H, W))
        x3 = x.reshape(b, 384, 14, 14)

        # Stage4
        x, H, W = self.patchmerging4((x, H, W))
        x, H, W = self.blocks4((x, H, W))
        x4 = x.reshape(b, 768, 7, 7)
        x = self.hierarchy_attention(x1, x2, x3, x4)
        x = self.skattention(x)
        x = x.reshape(b, 49, 768)
        return self.head(x)




def AllSwin_T(num_classes=1000):  # Swin Tiny
    return SwinTransformer(dims=(96, 192, 384, 768),
                           num_blocks=(2, 2, 6, 2),
                           num_heads=(3, 6, 12, 24),
                           num_classes=num_classes)


def AllSwin_S(num_classes=1000):  # Swin Small
    return SwinTransformer(dims=(96, 192, 384, 768),
                           num_blocks=(2, 2, 18, 2),
                           num_heads=(3, 6, 12, 24),
                           num_classes=num_classes)


def AllSwin_B(num_classes=1000):  # Swin Base
    return SwinTransformer(dims=(128, 256, 512, 1024),
                           num_blocks=(2, 2, 18, 2),
                           num_heads=(4, 8, 16, 32),
                           num_classes=num_classes)


def AllSwin_L(num_classes=1000):  # Swin Large
    return SwinTransformer(dims=(192, 384, 768, 1536),
                           num_blocks=(2, 2, 18, 2),
                           num_heads=(6, 12, 24, 48),
                           num_classes=num_classes)


if __name__ == "__main__":
    cuda = True if torch.cuda.is_available() else False
    images = torch.randn(8, 3, 224, 224)
    swin_t = Swin_T()
    swin_s = Swin_S()
    swin_b = Swin_B()
    swin_l = Swin_L()
    if cuda:
        images = images.cuda()
        swin_t.cuda()
        swin_s.cuda()
        swin_b.cuda()
        swin_l.cuda()
    print(swin_t(images).shape)
    print(swin_s(images).shape)
    print(swin_b(images).shape)
    print(swin_l(images).shape)

