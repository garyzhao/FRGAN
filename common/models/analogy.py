import torch
import torch.nn as nn


def _compute_layer_config(img_size):
    min_size = img_size
    block_num = 0
    while min_size >= 8:
        min_size /= 2
        block_num += 1
    return block_num, min_size


def _make_upsample_layer(in_channels, out_channels, filter_size):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, out_channels, filter_size, 1, 1))


class ResUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, relu=nn.ReLU()):
        super(ResUpsampleBlock, self).__init__()
        self.shortcut = _make_upsample_layer(in_channels, out_channels, filter_size)
        self.conv1 = _make_upsample_layer(in_channels, out_channels, filter_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, filter_size, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = relu

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return shortcut + out


class ResSubsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, filter_size, relu=nn.ReLU()):
        super(ResSubsampleBlock, self).__init__()
        self.shortcut = nn.Sequential(nn.AvgPool2d(2), nn.Conv2d(in_channels, out_channels, filter_size, 1, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels, filter_size, 1, 1)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, filter_size, 1, 1), nn.AvgPool2d(2))
        self.relu = relu

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return shortcut + out


class EncoderU(nn.Module):

    def __init__(self, input_size, z_dim, input_dim=3, filter_size=3, min_filter_num=64, max_filter_num=512):
        super(EncoderU, self).__init__()
        self.block_num, self.min_size = _compute_layer_config(input_size)

        res_blocks = []
        current_filter_num = min_filter_num
        for i in range(self.block_num):
            next_filter_num = min(current_filter_num * 2, max_filter_num)
            res_blocks.append(ResSubsampleBlock(current_filter_num, next_filter_num, filter_size))
            current_filter_num = next_filter_num

        self.res_blocks = nn.ModuleList(res_blocks)
        self.conv = nn.Conv2d(input_dim, min_filter_num, filter_size, 1, 1)
        self.fc_z_dim = current_filter_num * self.min_size * self.min_size
        self.fc_z = nn.Linear(self.fc_z_dim, z_dim)

    def forward(self, x):
        out = [self.conv(x)]
        for res_block in self.res_blocks:
            out.append(res_block(out[-1]))
        z = out[-1].view(-1, self.fc_z_dim)
        z = self.fc_z(z)
        return z, out


class DecoderU(nn.Module):

    def __init__(self, input_size, z_dim, input_dim=3, filter_size=3, min_filter_num=64, max_filter_num=512):
        super(DecoderU, self).__init__()
        self.block_num, self.min_size = _compute_layer_config(input_size)

        res_blocks = []

        filters = []
        current_filter_num = min_filter_num
        for i in range(self.block_num):
            next_filter_num = min(current_filter_num * 2, max_filter_num)
            filters.append([next_filter_num, current_filter_num, 0])
            current_filter_num = next_filter_num

        for i in range(self.block_num):
            if i == 0:
                continue
            filter_num = filters[i][0]
            for j in range(i):
                filters[j][2] += filter_num

        for i in range(self.block_num):
            res_blocks.append(ResUpsampleBlock(filters[i][0] * 4 + filters[i][2], filters[i][1], filter_size))
        res_blocks.reverse()

        self.res_blocks = nn.ModuleList(res_blocks)
        self.max_filter_num = current_filter_num
        self.fc = nn.Linear(z_dim, self.max_filter_num * self.min_size * self.min_size)
        self.bn = nn.BatchNorm2d(min_filter_num)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(min_filter_num * 4, input_dim, filter_size, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.tanh = nn.Tanh()
        self.sigmod = nn.Sigmoid()
        self.w = nn.Conv2d(min_filter_num * 4, 1, filter_size, 1, 1)

    def forward(self, z, outs):
        h0 = [self.fc(z).view(-1, self.max_filter_num, self.min_size, self.min_size)]
        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](torch.cat([out.pop() for out in outs] + h0, 1))
            for j in range(len(h0)):
                h0[j] = self.up(h0[j])
            h0.append(x)
        x = self.bn(x)
        x = self.relu(x)

        o = torch.cat([out.pop() for out in outs] + [x], 1)
        w = self.w(o)
        w = self.sigmod(w)

        x = self.conv(o)
        x = self.tanh(x)
        return x, w


class NetD(nn.Module):

    def __init__(self, input_size, input_dim=3, filter_size=3, min_filter_num=64, max_filter_num=512):
        super(NetD, self).__init__()
        self.block_num, self.min_size = _compute_layer_config(input_size)

        res_blocks = []
        current_filter_num = min_filter_num
        for i in range(self.block_num):
            next_filter_num = min(current_filter_num * 2, max_filter_num)
            res_blocks.append(ResSubsampleBlock(current_filter_num, next_filter_num, filter_size, relu=nn.LeakyReLU()))
            current_filter_num = next_filter_num

        self.res_blocks = nn.ModuleList(res_blocks)
        self.conv = nn.Conv2d(input_dim, min_filter_num, filter_size, 1, 1)
        self.fc_dim = current_filter_num * self.min_size * self.min_size
        self.fc = nn.Linear(self.fc_dim, 1)

    def forward(self, x):
        out = self.conv(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        out = out.view(-1, self.fc_dim)
        out = self.fc(out)
        return out


class ResUNetG(nn.Module):

    def __init__(self, input_size, h_dim, img_dim=3, norm_dim=3, filter_size=3, min_filter_num=64, max_filter_num=512):
        super(ResUNetG, self).__init__()
        self.img_encoder = EncoderU(input_size, h_dim, input_dim=img_dim, filter_size=filter_size, min_filter_num=min_filter_num, max_filter_num=max_filter_num)
        self.norm_encoder = EncoderU(input_size, h_dim, input_dim=norm_dim, filter_size=filter_size, min_filter_num=min_filter_num, max_filter_num=max_filter_num)
        self.decoder = DecoderU(input_size, h_dim, input_dim=img_dim, filter_size=filter_size, min_filter_num=min_filter_num, max_filter_num=max_filter_num)

    def forward(self, x, m, n):
        h, u = self.img_encoder(x)
        n0, u0 = self.norm_encoder(n)
        n1, u1 = self.norm_encoder(m)
        d = n0 - n1
        out, w = self.decoder(h + d, [u, u0, u1])

        m = w.repeat(1, 3, 1, 1)
        return out * m + x * (1 - m), w
