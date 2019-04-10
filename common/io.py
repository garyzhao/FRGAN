import os
import glob
import torch
import numpy as np
import scipy.io
from PIL import Image
from torchvision.utils import make_grid


def pil_load_image(img_path):
    return Image.open(img_path).convert('RGB')


def torch_to_pil_image(x):
    x_out = make_grid(x, normalize=True, scale_each=True)
    x_out = (x_out.numpy() * 255.0).astype('uint8')
    x_out = np.transpose(x_out, (1, 2, 0))
    x_out = Image.fromarray(x_out, 'RGB')
    return x_out


def load_dataset_split(data_fp):
    videos = {}
    with open(data_fp) as f:
        for line in f:
            k, img_name = line.rsplit('\n', 1)[0].rsplit(' ', 1)
            if k in videos:
                videos[k].append(img_name)
            else:
                videos[k] = [img_name]

    subjects = {}
    for k in videos:
        info = k.rsplit(' ')
        d = [info[0], info[1], info[2], videos[k]]
        if info[0] in subjects:
            subjects[info[0]].append(d)
        else:
            subjects[info[0]] = [d]
    return subjects


def load_face_data(img_path, transform=None):
    img = pil_load_image(img_path)
    if transform is not None:
        img = transform(img)

    params = scipy.io.loadmat(img_path[0:-4], variable_names=['norm'])
    norm = params['norm'].astype(np.float32)
    return img, np.transpose(norm, (2, 0, 1))


def save_checkpoint(state, checkpoint='checkpoint'):
    file_path = os.path.join(checkpoint, 'checkpoint_{:04d}.pth.tar'.format(state['epoch']))
    torch.save(state, file_path)


def dump_gif(imgs, output_path):
    for i in range(len(imgs)):
        imgs[i] = imgs[i].convert('P', dither=Image.NONE, palette=Image.ADAPTIVE)
    img = imgs[0]
    img.save(output_path, save_all=True, append_images=imgs[1:], duration=50, loop=0)


def dump_gif_from_folder(input_path, output_path, img_ext='.jpg'):
    imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(input_path, '*' + img_ext)))]
    dump_gif(imgs, output_path)
