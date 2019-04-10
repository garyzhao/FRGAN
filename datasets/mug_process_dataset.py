from __future__ import division
from __future__ import print_function

import os
import cv2
import dlib
import numpy as np
import argparse

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from scipy.io import savemat
from shutil import copyfile

from dfa.params import (tri, std_size)
from dfa.face import (load_model, crop_img, ToTensorGjz, NormalizeGjz, compute_face_norm)
from dfa.inference import (parse_param, parse_roi_box_from_landmark)
from common.io import load_dataset_split
from common.progress.bar import Bar


def main(args):
    print("==> using settings {}".format(args))

    out_size = args.out_size
    inp_path = args.inp_path
    out_path = args.out_path
    out_path = os.path.join(out_path, 'mug' + str(out_size))

    cudnn.benchmark = True
    model = load_model('models/phase1_wpdc_vdc_v2.pth.tar')
    model = model.cuda()
    model.eval()

    face_regressor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
    face_detector = dlib.get_frontal_face_detector()
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    process_split(inp_path, out_path, 'train.txt', face_detector, face_regressor, model, transform, out_size)
    process_split(inp_path, out_path, 'val.txt', face_detector, face_regressor, model, transform, out_size)
    process_split(inp_path, out_path, 'test.txt', face_detector, face_regressor, model, transform, out_size)


def process_split(inp_path, out_path, split_file, face_detector, face_regressor, model, transform, out_size):
    print("==> processing split {}".format(split_file))
    inp_fp = load_dataset_split(os.path.join('datasets/mug', split_file))

    os.makedirs(out_path)
    copyfile(os.path.join('datasets/mug', split_file), os.path.join(out_path, split_file))

    for sub in inp_fp:
        i = 0
        videos_fp = inp_fp[sub]
        bar = Bar('processing subject #{}'.format(sub), max=len(videos_fp))

        for _, exp, tak, img_list in videos_fp:
            roi_box = None
            out_fp = os.path.join(out_path, sub, exp, tak)
            os.makedirs(out_fp)

            for img_name in img_list:
                img_fp = os.path.join(inp_path, sub, exp, tak, img_name)

                img_ori = cv2.imread(img_fp, cv2.IMREAD_COLOR)
                if roi_box is None:
                    rects = face_detector(img_ori, 1)

                    if len(rects) == 0:
                        raise Exception('No faces detected in {}'.format(img_fp))

                    rect = rects[0]

                    # use landmark for cropping
                    pts = face_regressor(img_ori, rect).parts()
                    pts = np.array([[pt.x, pt.y] for pt in pts]).T
                    roi_box = parse_roi_box_from_landmark(pts)

                    # compute box for cropping
                    llength = ((roi_box[2] - roi_box[0]) + (roi_box[3] - roi_box[1])) / 2.0 * 0.95
                    center_x = (roi_box[2] + roi_box[0]) / 2.0
                    center_y = (roi_box[3] + roi_box[1]) / 2.0 - 0.04 * (roi_box[3] + roi_box[1])

                    out_box = [0] * 4
                    out_box[0] = center_x - llength / 2.0
                    out_box[1] = center_y - llength / 2.0
                    out_box[2] = out_box[0] + llength
                    out_box[3] = out_box[1] + llength

                img = crop_img(img_ori, roi_box)

                # forward: one step
                img = cv2.resize(img, dsize=(std_size, std_size), interpolation=cv2.INTER_LINEAR)
                inp = transform(img).unsqueeze(0)
                with torch.no_grad():
                    inp = inp.cuda()
                    param = model(inp)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

                p, offset, alpha_shp, alpha_exp = parse_param(param, whitening=True)
                norm = compute_face_norm(p, offset, alpha_shp, alpha_exp, tri, roi_box, out_box, out_size)
                img_out = crop_img(img_ori, out_box)
                img_out = cv2.resize(img_out, dsize=(out_size, out_size), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(os.path.join(out_fp, img_name), img_out)
                mat_fp = os.path.join(out_fp, img_name[:-4])
                mat = {'p': p, 'offset': offset, 'shp': alpha_shp, 'exp': alpha_exp, 'norm': norm, 'roi_box': roi_box, 'out_box': out_box}
                savemat(mat_fp, mat)

            bar.suffix = '({i}/{size}) | Total: {total:} | ETA: {eta:}'.format(
                i=i + 1,
                size=len(videos_fp),
                total=bar.elapsed_td,
                eta=bar.eta_td
            )
            bar.next()
            i += 1

        bar.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing MUG dataset.')
    parser.add_argument('--inp_path', type=str, required=True, help='input path of MUG.')
    parser.add_argument('--out_path', type=str, required=True, help='ouput path of processed MUG.')
    parser.add_argument('--out_size', type=int, default=64, help='output image size.')

    main(parser.parse_args())
