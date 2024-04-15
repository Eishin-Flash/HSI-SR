import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from .VQ_Restormer_v2 import VQ_AE_Restormer

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class CodeFormer(VQ_AE_Restormer):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, in_channel=3,
                fix_modules=['quantize','Decoder','refinement','output'], vqgan_path='vq_s1/net_300epoch.pth'):
        super(CodeFormer, self).__init__(inp_channels=31,out_channels=31,dim=48,
                                        num_blocks=[2,3,3,4],num_refinement_blocks=3,
                                        heads=[1,2,4,8],ffn_expansion_factor=2.66,bias=False,
                                        LayerNorm_type='WithBias',codebook_size=1024,beta=0.25)

        if vqgan_path is not None:
            checkpoint = torch.load(vqgan_path)
            self.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},strict=True)

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2

        self.position_emb = PositionEmbeddingSine(num_pos_feats=256)
        self.feat_emb = nn.Linear(384, self.dim_embd)

        self.expand_conv=nn.Conv2d(in_channel,31,kernel_size=1)
        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))
    

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, code_only=False):
        # ################### Encoder #####################

        x = self.expand_conv(x)
        x = self.Encoder(x)
        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        pos_emb = self.position_emb(x)
        #b,512,x,y
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        pos_emb = pos_emb.flatten(2).permute(2,0,1)
        #xy,b,512
        query_emb = feat_emb
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n

        if code_only: # for training stage II
          # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0],61,64,384])
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

        quant_feat = quant_feat.detach() # for training stage III

        # ################## Generator ####################
        x = quant_feat
        out = self.Decoder(x)
        # logits doesn't need softmax before cross_entropy loss
        out = self.refinement(out)
        out = self.output(out)
        return out[:,:,:482,:512]