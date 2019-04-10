from __future__ import print_function

import argparse
import os
import glob
import cv2
import numpy as np

from scipy.io import loadmat
from common.utils import visualize_norm


def main(args):
    print("==> using settings {}".format(args))

    inp_path = args.inp_path
    img_fmt = args.img_fmt

    for f in sorted(glob.glob(os.path.join(inp_path, '*.' + img_fmt))):
        img = cv2.imread(f)

        norm = loadmat(f[:-4], variable_names=['norm'])['norm']
        norm = visualize_norm(norm)

        blend = cv2.addWeighted(cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX), 0.5, norm, 0.5, 0)
        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        cv2.imshow(inp_path, np.concatenate((img, norm, blend), axis=1))
        cv2.waitKey(90)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Processing MUG dataset.')
    parser.add_argument('--inp_path', type=str, required=True, help='input path of the video for visualization.')
    parser.add_argument('--img_fmt', type=str, default='jpg', help='input image format.')

    main(parser.parse_args())