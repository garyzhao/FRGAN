from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import os.path as osp
import scipy.io as sio


def make_abs_path(f, d):
    return osp.join(osp.dirname(osp.realpath(f)), d)


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def load(fp):
    suffix = get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def dump_vertex(vertex, wfp):
    sio.savemat(wfp, {'vertex': vertex})
    print('Dump to {}'.format(wfp))


def dump_to_ply(vertex, tri, wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    n_vertex = vertex.shape[1]
    n_face = tri.shape[1]
    header = header.format(n_vertex, n_face)

    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1 - 1, idx2 - 1, idx3 - 1))
    print('Dump tp {}'.format(wfp))
