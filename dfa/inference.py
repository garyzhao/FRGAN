from __future__ import division
from __future__ import absolute_import

import numpy as np
from math import sqrt
from .params import (std_size, param_mean, param_std, u, w_shp, w_exp, u_base, w_shp_base, w_exp_base)


def parse_param(param, whitening=True):
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    return p, offset, alpha_shp, alpha_exp


def reconstruct_vertex(p, offset, alpha_shp, alpha_exp, dense=True):

    if dense:
        vertex = (u + np.dot(w_shp, alpha_shp) + np.dot(w_exp, alpha_exp)).reshape(3, -1, order='F')
        p_vertex = np.dot(p, vertex) + offset

        # transform to image coordinate space
        p_vertex[1, :] = std_size + 1 - p_vertex[1, :]
    else:
        """For 68 pts"""
        vertex = (u_base + np.dot(w_shp_base, alpha_shp) + np.dot(w_exp_base, alpha_exp)).reshape(3, -1, order='F')
        p_vertex = np.dot(p, vertex) + offset

        # transform to image coordinate space
        p_vertex[1, :] = std_size + 1 - p_vertex[1, :]

    return p_vertex, vertex


def predict_vertices(p, offset, alpha_shp, alpha_exp, roi_box, out_box, out_size, dense):
    p_vertex, vertex = reconstruct_vertex(p, offset, alpha_shp, alpha_exp, dense=dense)
    sx, sy, ex, ey = roi_box

    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    p_vertex[0, :] = p_vertex[0, :] * scale_x + sx
    p_vertex[1, :] = p_vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    p_vertex[2, :] *= s

    sx, sy, ex, ey = out_box
    scale_x = out_size / (ex - sx)
    scale_y = out_size / (ey - sy)
    p_vertex[0, :] = (p_vertex[0, :] - sx) * scale_x
    p_vertex[1, :] = (p_vertex[1, :] - sy) * scale_y
    s = (scale_x + scale_y) / 2
    p_vertex[2, :] *= s

    return p_vertex, vertex


def predict_dense(p, offset, alpha_shp, alpha_exp, roi_box, out_box, out_size):
    return predict_vertices(p, offset, alpha_shp, alpha_exp, roi_box, out_box, out_size, dense=True)


def predict_68pts(p, offset, alpha_shp, alpha_exp, roi_box, out_box, out_size):
    return predict_vertices(p, offset, alpha_shp, alpha_exp, roi_box, out_box, out_size, dense=False)


def parse_roi_box_from_landmark(pts):
    """calc roi box from landmark"""
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box
