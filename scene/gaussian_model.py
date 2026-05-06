import os, time
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from torch   import nn
from plyfile import PlyData, PlyElement
from simple_knn._C        import distCUDA2
from utils.sh_utils       import SH2RGB
from utils.system_utils   import mkdir_p
from utils.general_utils  import inverse_sigmoid, get_expon_lr_func
from utils.graphics_utils import BasicPointCloud
from models.models        import *

def load_term(plydata, name):
    term_names = [p.name for p in plydata.elements[0].properties if p.name.startswith(name)]
    term_names = sorted(term_names, key = lambda x: int(x.split('_')[-1]))
    term       = np.zeros((plydata.elements[0][term_names[0]].shape[0], len(term_names)))
    for idx, attr_name in enumerate(term_names):
        term[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return term
    
class GaussianModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        self.normal_activation          = torch.nn.functional.normalize
        self.rgb_activation             = torch.sigmoid
        self.net_activation             = torch.nn.functional.relu

        self._xyz        = torch.empty(0)
        self._normal     = torch.empty(0)
        self._scaling    = torch.empty(0)
        self._gamma_m    = torch.empty(0)
        self._feature    = torch.empty(0)
        self.scale_range = False
        self.dim_feature = 8
        
        self.rendernet = rendernet(8, 16, self.dim_feature, 64, device='cuda', view_pe=3, fea_pe = 0, btn_freq=[0.3, 10.0],)

        self.optimizer = None
        self.spatial_lr_scale = 0
        self.opt = opt

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return self.normal_activation(self._normal)

    @property
    def get_scaling(self):
        if self.scale_range:
            scale = torch.sigmoid(self._scaling) * self.scale_max_val[0]
        else:
            scale = self.scaling_activation(self._scaling)
        return scale

    @property
    def get_gamma_m(self):
        return self._gamma_m

    @property
    def get_feature(self):
        return self._feature
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = torch.tensor(np.asarray(pcd.points)).float().cuda()
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        normal = F.normalize(points - points.mean(dim=0, keepdim=True), dim=-1)

        dist2  = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.ones_like(dist2) * torch.sqrt(dist2).mean() * 2)[...,None]
        print("Number of points at initialisation : ", points.shape[0])

        self._xyz     = nn.Parameter(points.requires_grad_(True))
        self._normal  = nn.Parameter(normal.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        # init so softplus(_gamma_m) ≈ 2.0
        self._gamma_m = nn.Parameter(torch.full((points.shape[0], 1), 1.25, device="cuda", dtype=torch.float32).requires_grad_(True))
        self._feature = nn.Parameter(torch.randn((points.shape[0], self.dim_feature), device="cuda"))

    def create_from_3dgs(self, checkpoint_ply : str, spatial_lr_scale : float):
        # load point and normal
        self.spatial_lr_scale = spatial_lr_scale
        plydata = PlyData.read(checkpoint_ply)
        points  = np.stack((np.asarray(plydata.elements[0]["x"]),
                            np.asarray(plydata.elements[0]["y"]),
                            np.asarray(plydata.elements[0]["z"])),  axis=1)
        points  = torch.tensor(points).float().cuda()
        
        normal  = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal  = torch.tensor(normal).float().cuda()   

        diffuse = SH2RGB(load_term(plydata, 'f_dc_'))
        opacity = torch.sigmoid(torch.tensor(np.asarray(plydata.elements[0]["opacity"])).float()) 
        print("Loaded number of 3DGS points: ", points.shape[0])

        if 'scan' in checkpoint_ply:
            opacity_thred = 0.5
            points  = points[opacity  > opacity_thred]
            normal  = normal[opacity  > opacity_thred]
            diffuse = diffuse[opacity > opacity_thred]
            print("number of 3DGS points with opacity > ", str(opacity_thred) ," : ", points.shape[0])
            
        axis_max = points.max(dim=0)[0]
        axis_min = points.min(dim=0)[0]
        self.axis_center = nn.Parameter(((axis_max + axis_min) / 2).detach())
        
        # voxel_down_sample
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        pcd.normals = o3d.utility.Vector3dVector(normal.detach().cpu().numpy())
        pcd.colors  = o3d.utility.Vector3dVector(np.clip(diffuse, 0, 1))
        downpcd = pcd.voxel_down_sample(voxel_size=(self.opt.meshscale / (self.opt.grid_resolution - 1) / 2))
        points  = torch.tensor(np.asarray(downpcd.points)).float().cuda()     
        normal  = torch.tensor(np.asarray(downpcd.normals)).float().cuda()     
        diffuse = torch.tensor(np.asarray(downpcd.colors)).float().cuda()
        print("number of 3DGS points in grids ", points.shape[0])

        # voxel_down_sample
        max_scale  = (self.opt.meshscale / (self.opt.grid_resolution - 1) * 1.5)
        init_scale = (self.opt.meshscale / (self.opt.grid_resolution - 1) * 1.0)

        self.scale_max_val = torch.ones([points.shape[0],1]).cuda() * max_scale
        self.scale_range = True
        if self.scale_range:
            scales = inverse_sigmoid(init_scale / self.scale_max_val)
        else:
            scales  = torch.log(torch.clamp_min(torch.from_numpy(nearest_dist).float().cuda(), 0.003))[:,None]

        self._xyz     = nn.Parameter(points.requires_grad_(True))
        self._normal  = nn.Parameter(normal.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._gamma_m = nn.Parameter(torch.full((points.shape[0], 1), 1.25, device="cuda", dtype=torch.float32).requires_grad_(True))
        self._feature = nn.Parameter(torch.randn((points.shape[0], self.dim_feature), device="cuda"))
        print("Number of points at initialisation : ", points.shape[0])


    def create_from_ply(self, checkpoint_ply : str, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale

        plydata = PlyData.read(checkpoint_ply)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        normal  = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
        normal  = torch.tensor(normal).float().cuda()   

        self._xyz        = nn.Parameter(torch.tensor(xyz     , dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal     = nn.Parameter(torch.tensor(normal  , dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling    = nn.Parameter(torch.tensor(load_term(plydata, 'scale_')   , dtype=torch.float, device="cuda").requires_grad_(True))
        gamma_m_names    = [p.name for p in plydata.elements[0].properties if p.name.startswith("gamma_m_")]
        if len(gamma_m_names) >= 1:
            self._gamma_m = nn.Parameter(torch.tensor(load_term(plydata, 'gamma_m_'), dtype=torch.float, device="cuda").requires_grad_(True))
        else:
            self._gamma_m = nn.Parameter(torch.full((xyz.shape[0], 1), 1.25, device="cuda", dtype=torch.float32).requires_grad_(True))
        self._feature    = nn.Parameter(torch.tensor(load_term(plydata, 'feature_') , dtype=torch.float, device="cuda").requires_grad_(True))

        scale_max_val_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scalemaxval_")]
        if len(scale_max_val_names) >= 1:
            scale_max_val      = load_term(plydata, 'scalemaxval_')
            self.scale_range   = True
            self.scale_max_val = torch.tensor(scale_max_val, dtype=torch.float, device="cuda")

        self.axis_center = nn.Parameter(torch.zeros(3).cuda().detach())
        ckpt_path  = os.path.dirname(checkpoint_ply) + '/ckpt.th'
        model_dict = torch.load(ckpt_path)['state_dict']
        for key in list(model_dict.keys()):
            if key.startswith('_'):
                model_dict.pop(key)
        self.load_state_dict(model_dict, strict=False)
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._gamma_m.shape[1]):
            l.append('gamma_m_{}'.format(i))
        for i in range(self._feature.shape[1]):
            l.append('feature_{}'.format(i))
        if self.scale_range:
            for i in range(self.scale_max_val.shape[1]):
                l.append('scalemaxval_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz     = self._xyz.detach().cpu().numpy()
        normal  = self._normal.detach().cpu().numpy()
        scale   = self._scaling.detach().cpu().numpy()
        gamma_m = self._gamma_m.detach().cpu().numpy()
        feature = self._feature.detach().cpu().numpy()
        if self.scale_range:
            scale_max_val = np.ones([xyz.shape[0], 1]) * self.scale_max_val[0].item()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements   = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.scale_range:
            attributes  = np.concatenate((xyz, normal, scale, gamma_m, feature, scale_max_val), axis=1)
        else:
            attributes  = np.concatenate((xyz, normal, scale, gamma_m, feature), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        ckpt = {'state_dict': self.state_dict()}
        torch.save(ckpt, os.path.dirname(path) + '/ckpt.th')
    
    def training_setup(self, training_args):
        l = [
            {'params': [self._xyz],                 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal],              'lr': training_args.normal_lr,     "name": "normal"},
            {'params': [self._scaling],             'lr': training_args.scaling_lr,    "name": "scaling"},
            {'params': [self._gamma_m],             'lr': training_args.scaling_lr,   "name": "gamma_m"},
            {'params': [self._feature],             'lr': training_args.feature_lr,    "name": "feature"},
            {'params': self.rendernet.parameters(), 'lr': training_args.rendernet_lr,  "name": "rendernet"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, betas=(0.9,0.99))
        self.xyz_scheduler_args = get_expon_lr_func(lr_init       = training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final      = training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult = training_args.position_lr_delay_mult,
                                                    max_steps     = training_args.position_lr_max_steps)

        self.lr_factor = 0.1**(1/training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            else:
                param_group['lr'] = param_group['lr'] * self.lr_factor
            return lr
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ['rendernet']:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _densify_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ['rendernet']:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densify_points(self, tensors_dict):
        optimizable_tensors = self._densify_optimizer(tensors_dict)
        self._xyz     = optimizable_tensors["xyz"]
        self._normal  = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._gamma_m = optimizable_tensors["gamma_m"]
        self._feature = optimizable_tensors["feature"]

    def prune_points(self, mask):
        valid_points_mask   = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz     = optimizable_tensors["xyz"]
        self._normal  = optimizable_tensors["normal"]
        self._scaling = optimizable_tensors["scaling"]
        self._gamma_m = optimizable_tensors["gamma_m"]
        self._feature = optimizable_tensors["feature"]