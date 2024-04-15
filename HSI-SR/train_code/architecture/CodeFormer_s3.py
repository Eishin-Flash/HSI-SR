import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from .CodeFormer import CodeFormer

class CodeFormer_s3(CodeFormer):
    def __init__(self, dim_embd=512, n_head=8, n_layers=9, 
                codebook_size=1024, in_channel=3,
                fix_modules=['quantize','Decoder','refinement','output'],dim=48,bias=False,
                CodeFormer_path='vq_s1/codeformer.pth'):
        super(CodeFormer_s3,self).__init__(
            dim_embd=dim_embd,
            n_head=n_head,
            n_layers=n_layers,
            codebook_size=codebook_size,
            in_channel=in_channel,
            fix_modules=fix_modules,
            vqgan_path=None
        )
        
        if CodeFormer_path is not None:
            checkpoint = torch.load(CodeFormer_path)
            self.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},strict=True)
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2 ** 1), dim, kernel_size=1, bias=bias)
        self.skip_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self,x):
        #b,3,128,128
        x = self.expand_conv(x)
        #b,31,128,128
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        inp_img = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        h_inp_n=inp_img.shape[2]
        w_inp_n=inp_img.shape[3]
        #通过padding避免出现输出图像与输入尺寸不一致的情况
        inp_enc_level1 = self.Encoder.patch_embed(inp_img)
        #b,48,128,128
        out_enc_level1 = self.Encoder.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.Encoder.down1_2(out_enc_level1)
        #b,96,64,64
        out_enc_level2 = self.Encoder.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.Encoder.down2_3(out_enc_level2)
        #b,192,32,32
        out_enc_level3 = self.Encoder.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.Encoder.down3_4(out_enc_level3)
        #b,384,16,16
        latent = self.Encoder.latent(inp_enc_level4)
        lq_feat = latent
        
        pos_emb = self.position_emb(latent)
        #b,512,16,16
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        #256,b,512
        pos_emb = pos_emb.flatten(2).permute(2,0,1)
        #256,b,512
        query_emb = feat_emb
        
        # Transformer
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # output logits
        logits = self.idx_pred_layer(query_emb) # (hw)bn
        logits = logits.permute(1,0,2) # (hw)bn -> b(hw)n
        
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0],int(h_inp_n/8),int(w_inp_n/8),384])
        #b,384,16,16
        inp_dec_level3 = self.Decoder.up4_3(quant_feat)
        #b,192,32,32
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.Decoder.decoder_level3(inp_dec_level3)
        #b,192,32,32

        inp_dec_level2 = self.Decoder.up3_2(out_dec_level3)
        #b,96,64,64
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.Decoder.decoder_level2(inp_dec_level2)
        #b,96,64,64
        inp_dec_level1 = self.Decoder.up2_1(out_dec_level2)
        #b,48,128,128
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1 = self.Decoder.decoder_level1(inp_dec_level1)
        #b,48,128,128

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1[:, :, :h_inp, :w_inp], logits ,lq_feat