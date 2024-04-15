import torch
from .edsr import EDSR
from .HDNet import HDNet
from .hinet import HINet
from .hrnet import SGN
from .HSCNN_Plus import HSCNN_Plus
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .AWAN import AWAN
from .VQ_Restormer_v2 import VQ_AE_Restormer
from .CodeFormer import CodeFormer
from .CodeFormer_s3 import CodeFormer_s3
from .Spectral_VQGAN import Spectral_VQAutoEncoder,Spectral_Discrminator

def model_generator(method, pretrained_model_path=None):
    if method == 'mirnet':
        model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1).cuda()
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus().cuda()
    elif method == 'mst':
        model = MST(dim=31, stage=2, num_blocks=[4, 7, 5]).cuda()
    elif method == 'hinet':
        model = HINet(depth=4).cuda()
    elif method == 'mprnet':
        model = MPRNet(num_cab=4).cuda()
    elif method == 'restormer':
        model = Restormer().cuda()
    elif method == 'edsr':
        model = EDSR().cuda()
    elif method == 'hdnet':
        model = HDNet().cuda()
    elif method == 'hrnet':
        model = SGN().cuda()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus().cuda()
    elif method == 'awan':
        model = AWAN().cuda()
    elif method == 'vqre':
        model =VQ_AE_Restormer().cuda()
    elif method == 'codeformer':
        model =CodeFormer().cuda()
    elif method == 'codeformer_s3':
        model =CodeFormer_s3().cuda()
    elif method == '3d_vq':
        model =Spectral_VQAutoEncoder().cuda()
    elif method == '3d_gan':
        model =Spectral_Discrminator().cuda()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
