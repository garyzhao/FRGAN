from __future__ import division
from __future__ import print_function

import os
import time
import argparse

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from common.models.analogy import ResUNetG
from common.mug_dataset import VideoFolder
from common.io import (torch_to_pil_image, dump_gif)
from common.utils import (AverageMeter, cat_triplet)
from common.progress.bar import Bar


def main(args):
    print("==> using settings {}".format(args))

    num_workers = 8
    img_dir_path = args.img_dir_path
    out_dir_path = args.out_dir_path

    cudnn.benchmark = True
    device = torch.device("cuda")

    h_dim = args.h_dim
    video_length = args.length
    img_size = args.img_size
    batch_size = args.batch_size

    transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_loader = torch.utils.data.DataLoader(VideoFolder(img_dir_path, 'test.txt', video_length, transform),
        batch_size=batch_size, num_workers=int(num_workers), shuffle=True, pin_memory=True, drop_last=True)

    model_gen = ResUNetG(img_size, h_dim, img_dim=3, norm_dim=3)
    model_gen = torch.nn.DataParallel(model_gen).to(device)

    # resume from a checkpoint
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model_gen.load_state_dict(checkpoint['gen_state_dict'])
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    # test
    test(test_loader, model_gen, device, out_dir_path)


def test(test_loader, model_gen, device, out_dir_path):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    if not os.path.exists(out_dir_path):
        os.makedirs(out_dir_path)
        print('Make output dir: {}'.format(out_dir_path))

    # switch to evaluate mode
    torch.set_grad_enabled(False)
    model_gen.eval()
    end = time.time()

    bar = Bar('Test ', max=len(test_loader))
    for i, (img_src, norm_src, img_t, norm_t) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        output = []
        for j in range(img_t.size(2)):
            x_src, n_src = img_src.to(device), norm_src.to(device)
            x_dst, n_dst = img_t[:, :, j, :, :].to(device), norm_t[:, :, j, :, :].to(device)
            x_fake, _ = model_gen(x_src, n_src, n_dst)

            _, x_out = cat_triplet(n_dst, x_fake, x_dst)
            output.append(torch_to_pil_image(x_out))

        # save videos
        dump_gif(output, os.path.join(out_dir_path, 'test_batch_{:04d}.gif'.format(i + 1)))

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
            batch=i + 1,
            size=len(test_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td
        )

        bar.next()

    bar.finish()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing the networks on MUG dataset.')
    parser.add_argument("--img_dir_path", type=str, required=True, help='folder path including input testing images')
    parser.add_argument("--out_dir_path", type=str, required=True, help='folder path including output videos')
    parser.add_argument("--checkpoint", type=str, required=True, help='checkpoint path')
    parser.add_argument("--batch_size", type=int, default=8, help='batch size of for testing')
    parser.add_argument("--img_size", type=int, default=64, help='input image size (should be 64, 96)')
    parser.add_argument("--length", type=int, default=96, help='video length of the output')
    parser.add_argument("--h_dim", type=int, default=128, help="dimension of the auto-encoder's hidden state")
    args = parser.parse_args()

    main(parser.parse_args())
