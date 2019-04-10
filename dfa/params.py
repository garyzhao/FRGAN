from __future__ import division
from __future__ import absolute_import

import os.path as osp
import scipy.io as sio
from .utils import (load, make_abs_path)

_base_dir = make_abs_path(__file__, '../configs')

_key_pts = load(osp.join(_base_dir, 'keypoints_sim.npy'))
w_shp = load(osp.join(_base_dir, 'w_shp_sim.npy'))
w_exp = load(osp.join(_base_dir, 'w_exp_sim.npy'))  # simplified version
_param_meta = load(osp.join(_base_dir, 'param_whitening.pkl'))

# param_mean and param_std are used for re-whitening
param_mean = _param_meta.get('param_mean')
param_std = _param_meta.get('param_std')
u_shp = load(osp.join(_base_dir, 'u_shp.npy'))
u_exp = load(osp.join(_base_dir, 'u_exp.npy'))
u = u_shp + u_exp
# w = np.concatenate((w_shp, w_exp), axis=1)
# w_base = w[keypoints]
# w_norm = np.linalg.norm(w, axis=0)
# w_base_norm = np.linalg.norm(w_base, axis=0)

# for inference
# dim = w_shp.shape[0] // 3
u_base = u[_key_pts].reshape(-1, 1)
w_shp_base = w_shp[_key_pts]
w_exp_base = w_exp[_key_pts]
std_size = 120

# for the refined mesh
_param_mesh = sio.loadmat(osp.join(_base_dir, 'param_mesh.mat'))
w_shp = _param_mesh['w'][:, 0:40]
w_exp = _param_mesh['w_exp'][:, 0:10]
u = _param_mesh['mu_shape'] + _param_mesh['mu_exp']
tri = _param_mesh['tri']
