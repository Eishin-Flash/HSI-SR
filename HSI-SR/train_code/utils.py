from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
import torch.nn.functional as F
import lpips

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))
    
class Loss_LPIPS(nn.Module):
    def __init__(self, 
            loss_weight=1.0, 
            use_input_norm=True,
            range_norm=False,):
        super(Loss_LPIPS, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        assert pred.shape == target.shape
        #b,31,128,128
        pred=pred.unsqueeze(1)
        target=target.unsqueeze(1)
        pred=torch.cat([pred,pred,pred],dim=1)
        target=torch.cat([target,target,target],dim=1)
        #b,3,31,128,128
        lpips_loss=0
        for i in range(target.shape[2]):
            target_per_c = target[:,:,i,:,:]
            pred_per_c = pred[:,:,i,:,:]
            if self.range_norm:
                pred_per_c   = (pred_per_c + 1) / 2
                target_per_c = (target_per_c + 1) / 2
            if self.use_input_norm:
                pred_per_c   = (pred_per_c - self.mean) / self.std
                target_per_c = (target_per_c - self.mean) / self.std
            lpips_loss += self.perceptual(target_per_c.contiguous(), pred_per_c.contiguous())
        return self.loss_weight * lpips_loss.mean() / target.shape[2]

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.contiguous().view(-1))
        return mrae
    
class Loss_L1(nn.Module):
    def __init__(self):
        super(Loss_L1, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label)
        L1 = torch.mean(error.contiguous().view(-1))
        return L1

class Loss_CE(nn.Module):
    def __init__(self):
        super(Loss_CE,self).__init__()
    
    def forward(self,logits,idx_gt):
        #logits:b,n,hw
        #idx_gt:b,hw
        ce_loss=F.cross_entropy(logits,idx_gt)
        return ce_loss
        
        

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        Itrue = im_true.clamp(0., 1.).mul_(data_range).reshape(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close