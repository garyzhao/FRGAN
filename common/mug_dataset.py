from __future__ import print_function

import os
import torch
import torch.utils.data as data
from torchvision import transforms
from common.io import (load_face_data, load_dataset_split)


class ImageFolder(data.Dataset):

    def __init__(self, root_path, split_file, transform=transforms.ToTensor(), step=1):
        imgs = self._make_dataset(root_path, split_file, step)
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in the folders of: " + root_path + ".\n")

        self.imgs = imgs
        self.transform = transform
        print('Found %d images in %s.' % (len(self.imgs), root_path))

    @staticmethod
    def _make_dataset(dir_path, split_file, sample_step):
        img_pairs = []
        inp_fp = load_dataset_split(os.path.join(dir_path, split_file))
        for sub in inp_fp:
            videos_fp = inp_fp[sub]
            for _, exp, tak, img_list in videos_fp:
                init_img_fp = os.path.join(dir_path, sub, exp, tak, img_list[0])
                for img_name in img_list[::sample_step]:
                    img_fp = os.path.join(dir_path, sub, exp, tak, img_name)
                    img_pairs.append([init_img_fp, img_fp])
        return img_pairs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgs_fp = self.imgs[index]
        img_0, norm_0 = load_face_data(imgs_fp[0], self.transform)
        img_1, norm_1 = load_face_data(imgs_fp[1], self.transform)
        return img_0, norm_0, img_1, norm_1


class VideoFolder(data.Dataset):

    def __init__(self, root_path, split_file, video_length, transform=transforms.ToTensor()):
        videos = self._make_dataset(root_path, split_file, video_length)
        if len(videos) == 0:
            raise RuntimeError("Found 0 videos in the folders of: " + root_path + ".\n")

        self.videos = videos
        self.transform = transform
        print('Found %d videos in %s.' % (len(self.videos), root_path))

    @staticmethod
    def _make_dataset(dir_path, split_file, length):
        videos = []
        inp_fp = load_dataset_split(os.path.join(dir_path, split_file))
        for sub in inp_fp:
            videos_fp = inp_fp[sub]
            for _, exp, tak, img_list in videos_fp:
                imgs = []
                init_img_fp = os.path.join(dir_path, sub, exp, tak, img_list[0])

                for img_name in img_list:
                    img_fp = os.path.join(dir_path, sub, exp, tak, img_name)
                    imgs.append(img_fp)

                    if len(imgs) == length:
                        break

                while len(imgs) < length:
                    imgs.append(imgs[-1])

                videos.append([init_img_fp, imgs])
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_fp = self.videos[index]
        img_0, norm_0 = load_face_data(video_fp[0], self.transform)
        img_t, norm_t = [], []

        for img_fp in video_fp[1]:
            img, norm = load_face_data(img_fp, self.transform)
            img_t.append(torch.unsqueeze(img, 0))
            norm_t.append(torch.unsqueeze(torch.from_numpy(norm), 0))

        img_t = torch.transpose(torch.cat(img_t, 0), 0, 1)
        norm_t = torch.transpose(torch.cat(norm_t, 0), 0, 1)

        return img_0, norm_0, img_t, norm_t
