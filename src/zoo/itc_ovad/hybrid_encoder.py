'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from .utils import get_activation

from src.core import register

__all__ = ['HybridEncoder']

class Text_Guided_Feature_Enhance(nn.Module):
    def __init__(self,
                 class_embeddings_channels: int=1024,
                 image_channels: int=256,
                 num_heads: int = 8,
                 dual_align: bool=False
                ):
        super(Text_Guided_Feature_Enhance, self).__init__()
        self.num_heads = num_heads
        self.class_embeddings_channels = class_embeddings_channels
        self.image_channels = image_channels
        self.dual_align = dual_align
        
        if dual_align:
            self.img_proj = nn.Linear(self.image_channels, self.image_channels*2)
            self.text_proj = nn.Linear(self.class_embeddings_channels, self.image_channels*2)
            self.hidden_channel = self.image_channels*2
            self.head_channels = self.hidden_channel // num_heads
        else:
            self.img_proj = nn.Linear(self.image_channels, self.class_embeddings_channels)
            self.hidden_channel = self.class_embeddings_channels
            self.head_channels = self.hidden_channel // num_heads

        
        self.query = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   nn.Linear(self.hidden_channel, self.hidden_channel))
        self.key = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                 nn.Linear(self.hidden_channel, self.hidden_channel))
        self.value = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   nn.Linear(self.hidden_channel, self.hidden_channel))
        self.proj = nn.Linear(self.hidden_channel, self.image_channels)
    
    def forward(self, img_feat, class_embeddings):  
        B, C, token_num = img_feat.shape
        self.img_feat = img_feat
        # class_embeddings = class_embeddings[:, :-1, :] 
        pad_mask = (class_embeddings.abs().sum(dim=-1) == 0)  
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)         
        pad_mask = pad_mask.expand(B, self.num_heads, token_num, -1)
        
        # B, _, H, W = img_feat.shape
        # img_feat_tmp = img_feat.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        img_feat_tmp = img_feat.permute(0, 2, 1)
        img_feat_tmp = self.img_proj(img_feat_tmp)
        
        if self.dual_align:
            class_embeddings = self.text_proj(class_embeddings)
        
        q = self.query(img_feat_tmp)
        k = self.key(class_embeddings)
        v = self.value(class_embeddings)
        
        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)

        attn_weight = torch.matmul(q, k)
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight.masked_fill(pad_mask, float('-inf'))
        
        B_F_weight = attn_weight.clone()
        attn_weight = F.softmax(attn_weight, dim=-1)
        
        B_F_weight = B_F_weight[:,:,:,:-1].max(dim=-1)[0]
        B_F_weight = B_F_weight.sigmoid()

        v = v.permute(0, 2, 1, 3)
        aug_v = torch.matmul(attn_weight, v)
        aug_v = aug_v.permute(0, 2, 1, 3).reshape(B, -1, self.hidden_channel)

        aug_v = self.proj(aug_v)
        
        aug_v = aug_v.permute(0, 2, 1).reshape(B, self.num_heads, self.image_channels//self.num_heads, -1)
        aug_v = aug_v * (B_F_weight.reshape(B, self.num_heads, -1).unsqueeze(2))
        aug_v = aug_v.reshape(B, self.image_channels, token_num)
        return img_feat + aug_v
    
    # def forward(self, img_feat, class_embeddings):  
    #     B, _, H, W = img_feat.shape

    #     pad_mask = (class_embeddings.abs().sum(dim=-1) == 0)  
    #     pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)         
    #     pad_mask = pad_mask.expand(B, self.num_heads, H*W, -1)

    #     img_feat_tmp = img_feat.permute(0, 2, 3, 1).reshape(B, H*W, -1)
    #     img_feat_tmp = self.img_proj(img_feat_tmp)
        
    #     if self.dual_align:
    #         class_embeddings = self.text_proj(class_embeddings)
        
    #     q = self.query(img_feat_tmp)
    #     k = self.key(class_embeddings)
    #     v = self.value(class_embeddings)
        
    #     q = q.reshape(B, -1, self.num_heads, self.head_channels)
    #     k = k.reshape(B, -1, self.num_heads, self.head_channels)
    #     v = v.reshape(B, -1, self.num_heads, self.head_channels)
        
    #     q = q.permute(0, 2, 1, 3)
    #     k = k.permute(0, 2, 3, 1)

    #     attn_weight = torch.matmul(q, k)
    #     attn_weight = attn_weight / (self.head_channels**0.5)
    #     attn_weight = attn_weight.masked_fill(pad_mask, float('-inf'))
        
    #     B_F_weight = attn_weight.clone()
    #     attn_weight = F.softmax(attn_weight, dim=-1)

    #     B_F_weight = B_F_weight[:,:,:,:-1].max(dim=-1)[0]
    #     B_F_weight = B_F_weight.sigmoid()
        
    #     v = v.permute(0, 2, 1, 3)
    #     aug_v = torch.matmul(attn_weight, v)
    #     aug_v = aug_v.permute(0, 2, 1, 3).reshape(B, -1, self.hidden_channel)

    #     aug_v = self.proj(aug_v)
        
    #     aug_v = aug_v.permute(0, 2, 1).reshape(B, self.num_heads, self.image_channels//self.num_heads, -1)
    #     aug_v = aug_v * (B_F_weight.reshape(B, self.num_heads, -1).unsqueeze(2))
    #     aug_v = aug_v.reshape(B, self.image_channels, H, W)
        
    #     return img_feat + aug_v  
       

class Visual_Guided_Text_Refine(nn.Module):
    def __init__(self,
                 class_embeddings_channels: int=1024,
                 image_channels: int=256,
                 num_heads: int = 8,
                 dual_align: bool = False):
        super(Visual_Guided_Text_Refine, self).__init__()
        self.num_heads = num_heads
        self.class_embeddings_channels = class_embeddings_channels
        self.image_channels = image_channels
        self.dual_align = dual_align
        
        if dual_align:
            print('双向对齐')
            self.img_proj = nn.Linear(self.image_channels, self.image_channels*2)
            self.text_proj = nn.Linear(self.class_embeddings_channels, self.image_channels*2)
            self.hidden_channel = self.image_channels*2
            self.head_channels = self.hidden_channel // num_heads
        else:
            self.hidden_channel = class_embeddings_channels
            self.head_channels = self.hidden_channel // num_heads
            self.img_proj = nn.Linear(self.image_channels, self.class_embeddings_channels)

        self.query = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   nn.Linear(self.hidden_channel, self.hidden_channel))
        self.key = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                 nn.Linear(self.hidden_channel, self.hidden_channel))
        self.value = nn.Sequential(nn.LayerNorm(self.hidden_channel),
                                   nn.Linear(self.hidden_channel, self.hidden_channel))
        self.proj = nn.Linear(self.hidden_channel, self.class_embeddings_channels)
    
    def forward(self, img_feat, class_embeddings):
        class_embeddings_ori = class_embeddings
        
        _, num_q, _ = class_embeddings.shape
        B, num_k, _ = img_feat.shape
        
        # pad_mask = (class_embeddings.abs().sum(dim=-1) == 0)   
        # pad_mask = pad_mask.unsqueeze(1).unsqueeze(-1)         
        # pad_mask = pad_mask.expand(B, self.num_heads, num_q, -1)
        
        img_feat_tmp = self.img_proj(img_feat)
        
        if self.dual_align:
            class_embeddings = self.text_proj(class_embeddings)
            
        q = self.query(class_embeddings)
        k = self.key(img_feat_tmp)
        v = self.value(img_feat_tmp)
        
        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        

        attn_weight = torch.matmul(q, k)
        attn_weight = attn_weight / (self.head_channels**0.5)
        # attn_weight = attn_weight.masked_fill(pad_mask, float('-inf'))
        
        attn_weight = F.softmax(attn_weight, dim=-1)

        
        v = v.permute(0, 2, 1, 3)
        aug_v = torch.matmul(attn_weight, v)
        aug_v = aug_v.permute(0, 2, 1, 3).reshape(B, -1, self.hidden_channel)

        aug_v = self.proj(aug_v)

        pad_mask = (class_embeddings.abs().sum(dim=-1) == 0)   
        aug_v = aug_v.masked_fill(pad_mask.unsqueeze(-1), 0.)
        return class_embeddings_ori + aug_v


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 TGFE_Flag=True,
                 TGFE_dict=None,
                 VGTR_Flag=False,
                 VGTR_dict=None,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.TGFE_Flag = TGFE_Flag
        if TGFE_Flag:
            print("TGFE_Flag=True")
            self.TGFE = Text_Guided_Feature_Enhance(**TGFE_dict)
            
        self.VGTR_Flag = VGTR_Flag
        if VGTR_Flag:
            print("VGTR_Flag=True")
            self.VGTR = Visual_Guided_Text_Refine(**VGTR_dict)
            
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, class_embeddings):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        if self.TGFE_Flag:
            # proj_feats = [self.TGFE(feat, class_embeddings)  for feat in proj_feats]
            B, C = proj_feats[0].shape[:2]
            shapes = []   # 存储 (H_i, W_i)
            lengths = []  # 存储 H_i * W_i
            proj_feats_flat = []

            for feat in proj_feats:
                _, _, H, W = feat.shape
                shapes.append((H, W))
                lengths.append(H * W)
                proj_feats_flat.append(feat.flatten(2))

            concatenated_feats = torch.cat(proj_feats_flat, dim=2)

            output = self.TGFE(concatenated_feats, class_embeddings)
            
            if self.VGTR_Flag:
                v_aug_class_embeddings = self.VGTR(output.permute(0,2,1), class_embeddings)

            outputs_split = torch.split(output, lengths, dim=2)
            proj_feats = [feat.view(B, C, H, W) for feat, (H, W) in zip(outputs_split, shapes)]
        
        else:
            # proj_feats = [self.TGFE(feat, class_embeddings)  for feat in proj_feats]
            B, C = proj_feats[0].shape[:2]
            shapes = []   # 存储 (H_i, W_i)
            lengths = []  # 存储 H_i * W_i
            proj_feats_flat = []

            for feat in proj_feats:
                _, _, H, W = feat.shape
                shapes.append((H, W))
                lengths.append(H * W)
                proj_feats_flat.append(feat.flatten(2))

            concatenated_feats = torch.cat(proj_feats_flat, dim=2)

            # output = self.TGFE(concatenated_feats, class_embeddings)
            
            if self.VGTR_Flag:
                v_aug_class_embeddings = self.VGTR(concatenated_feats.permute(0,2,1), class_embeddings)

            outputs_split = torch.split(output, lengths, dim=2)
            proj_feats = [feat.view(B, C, H, W) for feat, (H, W) in zip(outputs_split, shapes)]
            
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs, v_aug_class_embeddings
