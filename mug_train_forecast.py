from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
from torchvision import transforms
from torchvision.utils import save_image

from common.models.analogy import ResUNetG, NetD
from common.mug_dataset import ImageFolder
from common.io import save_checkpoint
from common.utils import (AverageMeter, init_weights, loss_norm_l1, loss_l1, cat_triplet)
from common.progress.bar import Bar
from common.logger import (Logger, savefig)


def main(args):
    print("==> using settings {}".format(args))

    num_workers = 8
    num_epochs = args.num_epochs
    img_dir_path = args.img_dir_path

    cudnn.benchmark = True
    device = torch.device("cuda")

    h_dim = args.h_dim
    img_size = args.img_size
    batch_size = args.batch_size
    lr = 0.0001
    betas = (0.0, 0.9)

    transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = torch.utils.data.DataLoader(ImageFolder(img_dir_path, 'train.txt', transform), batch_size=batch_size,
        num_workers=int(num_workers), shuffle=True, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(ImageFolder(img_dir_path, 'test.txt', transform, step=32),
        batch_size=batch_size, num_workers=int(num_workers), shuffle=False, pin_memory=True, drop_last=True)

    model_gen = ResUNetG(img_size, h_dim, img_dim=3, norm_dim=3)
    model_dis = NetD(img_size, input_dim=6)

    model_gen = torch.nn.DataParallel(model_gen).to(device)
    model_dis = torch.nn.DataParallel(model_dis).to(device)

    model_gen.apply(init_weights)
    model_dis.apply(init_weights)

    optim_gen = optim.Adam(model_gen.parameters(), lr=lr, betas=betas)
    optim_dis = optim.Adam(model_dis.parameters(), lr=lr, betas=betas)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model_gen.load_state_dict(checkpoint['gen_state_dict'])
            model_dis.load_state_dict(checkpoint['dis_state_dict'])
            optim_gen.load_state_dict(checkpoint['gen_optim'])
            optim_dis.load_state_dict(checkpoint['dis_optim'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            out_dir_path = os.path.dirname(args.resume)
            logger = Logger(os.path.join(out_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0
        out_dir_path = os.path.join('checkpoints', datetime.datetime.now().isoformat())

        if not os.path.exists(out_dir_path):
            os.makedirs(out_dir_path)
            print('Make output dir: {}'.format(out_dir_path))

        logger = Logger(os.path.join(out_dir_path, 'log.txt'))
        logger.set_names(['Epoch', 'Train Loss G', 'Train Loss D'])

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        loss_gen, loss_dis = train(train_loader, model_gen, model_dis, optim_gen, optim_dis, device)

        # append logger file
        logger.append([epoch + 1, loss_gen, loss_dis])

        if (epoch + 1) % args.snapshot == 0:
            # validate
            validate(val_loader, model_gen, device, os.path.join(out_dir_path, 'epoch_{:04d}'.format(epoch + 1)))

            # save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'gen_state_dict': model_gen.state_dict(),
                'dis_state_dict': model_dis.state_dict(),
                'gen_optim': optim_gen.state_dict(),
                'dis_optim': optim_dis.state_dict()
            }, checkpoint=out_dir_path)

    logger.close()
    logger.plot(['Train Loss G', 'Train Loss D'])
    savefig(os.path.join(out_dir_path, 'log.eps'))


def train(train_loader, model_gen, model_dis, optim_gen, optim_dis, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    gen_losses = AverageMeter()
    dis_losses = AverageMeter()

    # switch to train mode
    torch.set_grad_enabled(True)
    model_gen.train()
    model_dis.train()
    end = time.time()

    bar = Bar('Train', max=len(train_loader))
    for i, (img_src, norm_src, img_dst, norm_dst) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_src, x_dst, n_src, n_dst = img_src.to(device), img_dst.to(device), norm_src.to(device), norm_dst.to(device)
        batch_size = x_src.size(0)

        ######################
        # (1) Update D network
        ######################

        x_fake, w = model_gen(x_src, n_src, n_dst)

        eps = torch.rand(batch_size, 1).to(device)
        eps = eps.expand(-1, int(x_src.numel() / batch_size)).view_as(x_src)

        x_rand = eps * x_dst.detach() + (1 - eps) * x_fake.detach()
        x_rand.requires_grad_()
        x_rand = torch.cat([x_rand, n_dst], dim=1)
        loss_rand_x = model_dis(x_rand)

        grad_outputs = torch.ones(loss_rand_x.size())
        grads = autograd.grad(loss_rand_x, x_rand, grad_outputs=grad_outputs.to(device), create_graph=True)[0]
        loss_gp = torch.mean((grads.view(batch_size, -1).pow(2).sum(1).sqrt() - 1).pow(2))

        loss_real_x = model_dis(torch.cat([x_dst, n_dst], dim=1))
        loss_fake_x = model_dis(torch.cat([x_fake.detach(), n_dst], dim=1))
        loss_dis = loss_fake_x.mean() - loss_real_x.mean() + 10.0 * loss_gp

        # compute gradient and bp
        optim_dis.zero_grad()
        loss_dis.backward()
        optim_dis.step()

        dis_losses.update(float(loss_dis.item()))

        ######################
        # (2) Update G network
        ######################

        loss_fake_x = model_dis(torch.cat([x_fake, n_dst], dim=1))
        loss_gen = -loss_fake_x.mean() + 3.0 * loss_l1(x_fake, x_dst) + 0.05 * loss_norm_l1(w)

        # compute gradient and bp
        optim_gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()

        gen_losses.update(float(loss_gen.item()))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss G: {loss_g:.4f} | Loss D: {loss_d: .4f}'.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss_g=gen_losses.avg,
            loss_d=dis_losses.avg
        )
        bar.next()

    bar.finish()
    return gen_losses.avg, dis_losses.avg


def validate(val_loader, model_gen, device, out_dir_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    os.makedirs(out_dir_path)

    # switch to evaluate mode
    torch.set_grad_enabled(False)
    model_gen.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(val_loader))
    for i, (img_src, norm_src, img_dst, norm_dst) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_src, x_dst, n_src, n_dst = img_src.to(device), img_dst.to(device), norm_src.to(device), norm_dst.to(device)
        x_fake, _ = model_gen(x_src, n_src, n_dst)
        num_rows, x_out = cat_triplet(x_src, x_fake, x_dst)
        save_image(x_out, os.path.join(out_dir_path, 'eval_batch_{:04d}.jpg'.format(i + 1)), normalize=True, nrow=num_rows)

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
            batch=i + 1,
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td
        )

        bar.next()

    bar.finish()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training the forecasting networks on MUG dataset.')
    parser.add_argument("--img_dir_path", type=str, required=True, help='folder path including input training images')
    parser.add_argument("--resume", type=str, required=False, help='resume checkpoint path')
    parser.add_argument('--snapshot', default=1, type=int, help='save models for every #snapshot epochs (default: 1)')
    parser.add_argument("--batch_size", type=int, default=32, help='batch size of each epoch for training')
    parser.add_argument("--num_epochs", type=int, default=20, help='number of epochs to run for training')
    parser.add_argument("--img_size", type=int, default=64, help='input image size (should be 64, 96)')
    parser.add_argument("--h_dim", type=int, default=128, help="dimension of the auto-encoder's hidden state")
    args = parser.parse_args()

    main(parser.parse_args())
