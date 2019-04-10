from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import mobilenet_v1

from math import sqrt
from z_buffer import z_buffer_c
from .inference import predict_dense


def load_model(file_path, arch='mobilenet_1'):
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) + 10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    return model


def compute_bbox_size(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


def compute_face_norm(p, offset, alpha_shp, alpha_exp, tri, roi_box, out_box, out_size):
    face_n = np.zeros((out_size, out_size, 3))
    p_vertex, vertex = predict_dense(p, offset, alpha_shp, alpha_exp, roi_box, out_box, out_size)

    tri_idx = _z_buffer(p_vertex[0:2, :], vertex, tri, out_size, out_size)
    tri_n = _triangle_norm(vertex, tri)
    for i_v in range(out_size):
        for i_u in range(out_size):
            idx = tri_idx[i_v, i_u] - 1
            if idx < 0:
                continue
            else:
                face_n[i_v, i_u, :] = tri_n[:, idx].T
    return face_n


def _triangle_norm(vertex, tri):
    pt1 = vertex[:, tri[0, :] - 1]
    pt2 = vertex[:, tri[1, :] - 1]
    pt3 = vertex[:, tri[2, :] - 1]
    tri_norm = np.cross(pt1 - pt2, pt1 - pt3, axis=0)
    tri_norm = normalize(tri_norm.T)
    return tri_norm.T


def _z_buffer(s2d, vertex, tri, height, width):
    tri_dis = np.zeros((height, width, 1))
    tri_idx = np.zeros((height, width, 1), dtype=np.int32)
    n = np.shape(tri)[1]

    point1 = s2d[:, tri[0, :] - 1]
    point2 = s2d[:, tri[1, :] - 1]
    point3 = s2d[:, tri[2, :] - 1]

    cent3d = (vertex[:, tri[0, :] - 1] + vertex[:, tri[1, :] - 1] + vertex[:, tri[2, :] - 1]) / 3
    r = cent3d[0, :] ** 2 + cent3d[1, :] ** 2 + cent3d[2, :] ** 2

    z_buffer_c(n, height, width, point1, point2, point3, r, tri_dis, tri_idx)
    return tri_idx


def normalize(v):
    norm = np.linalg.norm(v, axis=1) + np.finfo(np.float32).eps
    return v / np.tile(norm[:, np.newaxis], (1, 3))
