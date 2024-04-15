import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .vq import VectorQuantize

class ResBlockX(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.groupNorm = nn.GroupNorm(num_groups=in_channels,num_channels=in_channels)
        self.swish = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, in_channels, (3, 3, 3), padding=1,padding_mode='reflect')
        self.conv2 = nn.Conv3d(in_channels, in_channels, (3, 3, 3), padding=1,padding_mode='reflect')

    def forward(self, x):
        residual = x
        x = self.swish(self.groupNorm(x))
        x = self.conv1(x)
        x = self.swish(self.groupNorm(x))
        x = self.conv2(x)

        return x + residual
    
class ResBlockXY(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupNorm1 = nn.GroupNorm(num_groups=in_channels,num_channels=in_channels)
        self.swish = nn.SiLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=1,padding_mode='reflect')

        self.groupNorm2 = nn.GroupNorm(num_groups=out_channels,num_channels=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, (3, 3, 3), padding=1,padding_mode='reflect')

        self.resConv = nn.Conv3d(in_channels, out_channels, (1, 1, 1))

    def forward(self, x):
        residual = self.resConv(x)

        x = self.swish(self.groupNorm1(x))
        x = self.conv1(x)
        x = self.swish(self.groupNorm2(x))
        x = self.conv2(x)

        return x + residual
    
class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.leakyRELU = nn.LeakyReLU()
        self.pool = nn.AvgPool3d((2, 2, 2), padding=1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, (3, 3, 3), padding=1,padding_mode='reflect')
        self.conv2 = nn.Conv3d(out_channels, out_channels, (3, 3, 3), padding=1,padding_mode='reflect')

        self.resConv = nn.Conv3d(in_channels, out_channels, (1, 1, 1))

    def forward(self, x):
        residual = self.resConv(self.pool(x))

        x = self.leakyRELU(self.conv1(x))
        x = self.pool(x)
        x = self.leakyRELU(self.conv2(x))

        return x + residual

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.beta = beta  # commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        #1024,64
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, z):
        # z:b,c,t,h,w
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 4, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)
        #b,t,h,w,c->(b*t*h*w),c
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + (self.embedding.weight**2).sum(1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())

        mean_distance = torch.mean(d)
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        # min_encoding_scores, min_encoding_indices = torch.topk(d, 1, dim=1, largest=False)
        # [0-1], higher score, higher confidence
        # min_encoding_scores = torch.exp(-min_encoding_scores/10)

        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.codebook_size).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        loss = torch.mean((z_q-z.detach())**2) + self.beta * torch.mean((z_q.detach() - z) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # reshape back to match original input shape
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()

        return z_q, loss, {
            "perplexity": perplexity,
            "min_encodings": min_encodings,
            "min_encoding_indices": min_encoding_indices,
            "mean_distance": mean_distance
            }

    def get_codebook_feat(self, indices, shape):
        #根据离散序列取codebook向量
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 4, 1, 2, 3).contiguous()

        return z_q
    
class VectorQuantizer_v2(nn.Module):
    def __init__(self, codebook_size=1024, emb_dim=256, decay=0.8, commitment_weight=0.25):
        super(VectorQuantizer_v2, self).__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.vq = VectorQuantize(
            dim=self.emb_dim,
            codebook_size=self.codebook_size,
            decay=self.decay,
            commitment_weight=self.commitment_weight,
        )

    def forward(self, z):
        frames = z.shape[2]
        height = z.shape[3]
        width = z.shape[4]
        z = rearrange(z, 'b c f h w -> b (f h w) c')
        zq, indices, commit_loss = self.vq(z)
        zq = rearrange(zq, 'b (f h w) c -> b c f h w', f=frames, h=height, w=width)
        indices = indices.view(-1, 1)
        return zq, commit_loss[0], {
            "min_encoding_indices": indices
        }

    def get_codebook_feat(self, indices, shape):
        # shape:b,c,h,w
        # 根据离散序列取Codebook向量
        zq = self.vq.get_output_from_indices(indices)
        # (b,h,w),c
        zq = zq.view(shape)
        return zq
    
class Encoder(nn.Module):
    def __init__(self,nf):
        super(Encoder,self).__init__()
        self.first_conv=nn.Conv3d(3,8*nf,(3,3,3),padding=1,padding_mode='reflect')
        self.Res_seq_128=nn.Sequential(
            ResBlockX(8*nf),
            ResBlockX(8*nf)
        )
        self.Res_seq_64=nn.Sequential(
            ResBlockXY(8*nf,16*nf),
            ResBlockX(16*nf)
        )
        self.Res_seq_32=nn.Sequential(
            ResBlockX(16*nf),
            ResBlockX(16*nf)
        )
        self.Res_seq_16=nn.Sequential(
            ResBlockXY(16*nf,32*nf),
            ResBlockX(32*nf),
            ResBlockX(32*nf),
            ResBlockX(32*nf)
        )
        self.encoder_end=nn.Sequential(
            nn.GroupNorm(num_groups=32*nf,num_channels=32*nf),
            nn.SiLU(),
            nn.Conv3d(32*nf,256,(1,1,1))
        )
        self.pool3d=nn.AvgPool3d((2, 2, 2))
        self.pool2d=nn.AvgPool3d((1, 2, 2))

    def forward(self,x):
        #b,1,31,128,128-->b,3,31,128,128
        x=torch.cat([x,x,x],dim=1)
        b, c, t_inp, h_inp, w_inp = x.shape
        tb, hb, wb =4, 8, 8
        pad_t = (tb - t_inp % tb) % tb 
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb

        x = F.pad(x, [0, pad_w, 0, pad_h, 0, pad_t], mode='reflect')
        x=self.first_conv(x)
        x=self.Res_seq_128(x)
        x=self.pool3d(x)
        x=self.Res_seq_64(x)
        x=self.pool3d(x)
        x=self.Res_seq_32(x)
        x=self.pool2d(x)
        x=self.Res_seq_16(x)
        x=self.encoder_end(x)
        return x
                
        
class Decoder(nn.Module):
    def __init__(self,nf):
        super(Decoder,self).__init__()
        self.first_conv=nn.Conv3d(256, 32*nf, (3, 3, 3), padding=1,padding_mode='reflect')
        self.Res_seq_16=nn.Sequential(
            ResBlockX(32*nf),
            ResBlockX(32*nf),
            ResBlockX(32*nf),
            ResBlockX(32*nf)
        )
        self.upsample_16to32=nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(32*nf, 32*nf, (3, 3, 3), padding=1,padding_mode='reflect')
        )
        self.Res_seq_32=nn.Sequential(
            ResBlockXY(32*nf,16*nf),
            ResBlockX(16*nf)
        )
        self.upsample_32to64=nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(16*nf, 16*nf,(3, 3, 3), padding=1,padding_mode='reflect')
        )
        self.Res_seq_64=nn.Sequential(
            ResBlockX(16*nf),
            ResBlockX(16*nf)
        )
        self.upsample_64to128=nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='nearest'),
            nn.Conv3d(16*nf, 16*nf,(3, 3, 3), padding=1,padding_mode='reflect')
        )
        self.Res_seq_128=nn.Sequential(
            ResBlockXY(16*nf,8*nf),
            ResBlockX(8*nf)
        )
        self.decoder_end=nn.Sequential(
            nn.GroupNorm(num_groups=8*nf,num_channels=8*nf),
            nn.SiLU(),
            nn.Conv3d(8*nf, 3 ,(3, 3, 3), padding=1,padding_mode='reflect')
        )
        self.w = nn.Parameter(torch.ones(3))
        
    def forward(self,x):
        x=self.first_conv(x)
        x=self.Res_seq_16(x)
        x=self.upsample_16to32(x)
        x=self.Res_seq_32(x)
        x=self.upsample_32to64(x)
        x=self.Res_seq_64(x)
        x=self.upsample_64to128(x)
        x=self.Res_seq_128(x)
        x=self.decoder_end(x)
        #b,3,31,128,128
        w1=torch.exp(self.w[0])/torch.sum(torch.exp(self.w))
        w2=torch.exp(self.w[1])/torch.sum(torch.exp(self.w))
        w3=torch.exp(self.w[2])/torch.sum(torch.exp(self.w))
        x=w1*x[:,0,:,:,:]+w2*x[:,1,:,:,:]+w3*x[:,2,:,:,:]
        return x

               
class Spectral_VQAutoEncoder(nn.Module):
    def __init__(self,nf=3,codebook_size=1024,dim=256,beta=0.25):        
        super(Spectral_VQAutoEncoder,self).__init__()
        self.Encoder=Encoder(nf)
        self.vq=VectorQuantizer(codebook_size=codebook_size,emb_dim=dim,beta=beta)
        #self.vq=VectorQuantizer_v2(codebook_size=codebook_size, emb_dim=dim, decay=0.8, commitment_weight=beta)
        self.Decoder=Decoder(nf)

    def forward(self,x):
        #b,31,128,128-->b,1,31,128,128
        x=x.unsqueeze(1)
        t,h,w=x.shape[2],x.shape[3],x.shape[4]
        x=self.Encoder(x)
        zq,codebook_loss,_ = self.vq(x)
        out=self.Decoder(zq)[:,:t,:h,:w]
        return out,codebook_loss
    
    
    
        
class Spectral_Discrminator(nn.Module):
    def __init__(self, c):
        super(Spectral_Discrminator,self).__init__()

        self.image_process = nn.Sequential(
            nn.Conv3d(1, 64*c, (3, 3, 3), padding=1,padding_mode='reflect'),
            nn.LeakyReLU(),
            ResBlockDown(64*c, 128*c),
            ResBlockDown(128*c, 256*c),
            ResBlockDown(256*c, 256*c),
            ResBlockDown(256*c, 256*c),
            ResBlockDown(256*c, 256*c),
            nn.Conv3d(256*c, 256*c, (3, 3, 3), padding=1,padding_mode='reflect'),
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(256*c, 256*c),
            nn.LeakyReLU(),
            nn.Linear(256*c, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.image_process(x)
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        x = self.classifier(x)
        return rearrange(x, 'b c n -> b (c n)').sum(dim=1)
        #(b)
    