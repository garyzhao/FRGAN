import torch
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def visualize_norm(norm):
    (h, w, _) = norm.shape
    norm = norm.reshape((h * w, 3))
    x, y, z = norm[:, 0], norm[:, 1], norm[:, 2]
    mask = (x == 0) & (y == 0) & (z == 0)

    norm = (norm - (-1)) / (1 - (-1))
    norm[:, :3][mask] = [0, 0, 0]
    norm = norm.reshape((h, w, 3))
    return norm


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


def add_sn(m):
    for name, c in m.named_children():
        m.add_module(name, add_sn(c))

    # apply to every conv and conv transpose module in a model
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return nn.utils.spectral_norm(m)
    else:
        return m


def loss_norm_l1(a):
    return torch.mean(torch.abs(a))


def loss_l1(a, b):
    return torch.mean(torch.abs(a - b))


def cat_triplet(x_src, x_dst, x_fake):
    x_src = x_src.cpu()
    x_dst = x_dst.cpu()
    x_fake = x_fake.cpu()

    size = x_src.size(0)
    nrow = (min(8, size) if size <= 16 else 16)
    i_start = 0
    i_end = nrow

    out = []
    while i_end <= size:
        out.append(x_src[i_start:i_end, :, :, :])
        out.append(x_dst[i_start:i_end, :, :, :])
        out.append(x_fake[i_start:i_end, :, :, :])
        i_start = i_end
        i_end = i_end + nrow

    return nrow, torch.cat(out).clamp_(-1.0, 1.0)
