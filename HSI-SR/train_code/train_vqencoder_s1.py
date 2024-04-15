import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset_vq_s1 import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR, Loss_L1,Loss_LPIPS
import datetime

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='3d_vq')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=8e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/3dvq/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')
parser.add_argument("--use_per",type=int, default=1, help='whether use perceptual loss')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 10000
total_iteration = per_epoch_iteration*opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()
if opt.use_per==1:
    criterion_percep = Loss_LPIPS()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path).cuda()
print('Parameters number is ', sum(param.numel() for param in model.parameters()))

# output path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
opt.outf = opt.outf + date_time

if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()
    if opt.use_per==1:
        criterion_percep.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=2e-6)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = opt.pretrained_model_path
if resume_file is not None:
    if os.path.isfile(resume_file):
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        iteration = checkpoint['iter']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

def main():
    cudnn.benchmark = True
    iteration = 0
    record_mrae_loss = 1000
    while iteration<total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']
            optimizer.zero_grad()
            output,codebook_loss= model(images)
            mrae_loss = criterion_mrae(output, labels)
            if opt.use_per==1:
                percep_loss=criterion_percep(output,labels)
                loss=10*codebook_loss+mrae_loss+percep_loss
            else:
                loss=10*codebook_loss+mrae_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.data)
            iteration = iteration+1
            if iteration % 100 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))
                if opt.use_per:
                    print('codebook_losses:%.4f'%codebook_loss.item(),' mrae_losses:%.4f'%mrae_loss.item(),' percep_losses:%.4f'%percep_loss.item())
                else:
                    print('codebook_losses:%.4f'%codebook_loss.item(),' mrae_losses:%.4f'%mrae_loss.item())
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')
                # Save model
                if (mrae_loss < record_mrae_loss and iteration>20000) or  iteration % 50000 == 0:
                    print(f'Saving to {opt.outf}')
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)
                if mrae_loss < record_mrae_loss:
                    record_mrae_loss = mrae_loss
                print(" the min mrae is:",record_mrae_loss)
                # print loss
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, mrae_loss, rmse_loss, psnr_loss))
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (iteration, iteration//1000, lr, mrae_loss, rmse_loss, psnr_loss))
    return 0

# Validate
def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)[0]
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
        # record loss
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

if __name__ == '__main__':
    main()
    print(torch.__version__)