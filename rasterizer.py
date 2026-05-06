import os, time
import trimesh
import torch
import numpy  as np
import open3d as o3d
import torch.nn.functional as F
import nvdiffrast.torch    as dr
import diffsplatting

from PIL import Image
from os import makedirs
from diso import DiffMC
from utils.general_utils import inverse_sigmoid

os.environ["PYOPENGL_PLATFORM"] = "egl"

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def auto_normals(meshdict):
    i0 = meshdict['mesh_faces'][:, 0].to(torch.long)
    i1 = meshdict['mesh_faces'][:, 1].to(torch.long)
    i2 = meshdict['mesh_faces'][:, 2].to(torch.long)

    v0 = meshdict['mesh_verts'][i0, :]
    v1 = meshdict['mesh_verts'][i1, :]
    v2 = meshdict['mesh_verts'][i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(meshdict['mesh_verts'])
    v_nrm.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_nrm = safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))
    
    meshdict['mesh_v_nrm'] = v_nrm
    meshdict['mesh_t_nrm_idx'] = meshdict['mesh_faces']
    meshdict['mesh_face_nrm'] = face_normals
    return meshdict


class splatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, normals, scales, gamma_m, features, resolution, laplacian_lambda, sp_alpha, sp_seed):
        args = (means3D, normals, scales, gamma_m, features, resolution, float(laplacian_lambda), float(sp_alpha), int(sp_seed))
        # num_computed, num_phases, out_sums, out_sdfs, out_feat, out_sdfs_base, geoBuffer, binBuffer, voxBuffer, spaBuffer = diffsplatting.splatterforward(*args)
        num_computed, num_phases, out_sums, out_sdfs, out_feat, out_sdfs_base, out_lap_aux, geoBuffer, binBuffer, voxBuffer, spaBuffer = diffsplatting.splatterforward(*args)

        ctx.resolution = resolution
        ctx.num_computed = num_computed
        ctx.num_phases = num_phases
        ctx.sp_alpha = float(sp_alpha)
        ctx.sp_seed = int(sp_seed)

        #for backprop
        ctx.laplacian_lambda = float(laplacian_lambda)
        # Backward implements dF only (quotient field); merged grid is F + λ∇²F in out_sdfs.
        ctx.save_for_backward(
            means3D, normals, scales, gamma_m, features, geoBuffer, binBuffer, voxBuffer, spaBuffer,
            out_sums, out_sdfs_base, out_feat, out_lap_aux)

        return out_sums, out_sdfs, out_feat

    @staticmethod
    def backward(ctx, dL_dout_sums, dL_dout_sdfs, dL_dout_feat):
        resolution = ctx.resolution
        num_computed = ctx.num_computed
        num_phases = ctx.num_phases
        sp_alpha = ctx.sp_alpha
        sp_seed = ctx.sp_seed

        # newly added
        laplacian_lambda = ctx.laplacian_lambda
        # means3D, normals, scales, gamma_m, features, geoBuffer, binBuffer, voxBuffer, spaBuffer, out_sums, out_sdfs_base, out_feat = ctx.saved_tensors

        means3D, normals, scales, gamma_m, features, geoBuffer, binBuffer, voxBuffer, spaBuffer, out_sums, out_sdfs_base, out_feat, out_lap_aux = ctx.saved_tensors

        args = (num_computed, num_phases, resolution,
                means3D, normals, scales, gamma_m, features,
                geoBuffer, binBuffer, voxBuffer, spaBuffer,
                out_sums, out_sdfs_base, out_feat, out_lap_aux,
                dL_dout_sums, dL_dout_sdfs, dL_dout_feat,
                float(laplacian_lambda), float(sp_alpha), int(sp_seed))

        dL_dmeans3D, dL_dnormals, dL_dscales, dL_dgamma_m, dL_dfeatures = diffsplatting.splatterbackward(*args)
        grads = (dL_dmeans3D, dL_dnormals, dL_dscales, dL_dgamma_m, dL_dfeatures, None, None, None, None)

        return grads

class imlsplatting():
    def __init__(self, opt):
        self.opt        = opt
        self.rasterizer = dr.RasterizeCudaContext()
        self.mc_cuda    = DiffMC(dtype=torch.float32).cuda()
        # kernel_type = str(opt.get('kernel_type', 'gaussian')).lower()
        # self.kernel_type = 1 if kernel_type == 'gamma' else 0
        # self.gamma_m = float(opt.get('gamma_m', 10.0))
        # self.gamma_kappa = float(opt.get('gamma_kappa', 1.0))
        # self.gamma_eps_scale = float(opt.get('gamma_eps_scale', 1.0))
        
    def normalize(self, means3D, scales, axis_center):
        means3D = means3D - axis_center
        means3D = means3D / self.opt['meshscale'] # -0.5 ~ 0.5
        means3D = means3D + 0.5 # 0 ~ 1
        scales  = scales  / self.opt['meshscale']
        return means3D, scales
    
    def denormlize(self, mesh_verts, axis_center):
        mesh_verts = (mesh_verts - 0.5) * self.opt['meshscale'] + axis_center
        return mesh_verts

    def proj2clip(self, meshdict, camera):
        verts = meshdict['mesh_verts']
        num_verts, _ = verts.shape
        projmatrix = camera.full_proj_transform
        ones = torch.ones(num_verts, 1, dtype=verts.dtype, device=verts.device)
        verts_hom  = torch.cat([verts, ones], dim=1)
        verts_clip = torch.matmul(verts_hom, projmatrix.unsqueeze(0))
        meshdict['verts_clip'] = verts_clip
        return meshdict
    
    def marchingcube(self, meshdict, isovalue=0):
        out_sums = meshdict['out_sums']
        out_sdfs = meshdict['out_sdfs']
        out_feat = meshdict['out_feat']

        grid_resolution = out_sums.shape[0]
        out_sdfs = out_sdfs + (out_sums == 0) * (1 / (grid_resolution - 1))

        verts, feats, tris = self.mc_cuda(out_sdfs, out_feat, isovalue=isovalue, normalize=True)
        verts = verts[:,[2,1,0]]
        tris  = tris[:,[2,1,0]]

        meshdict['mesh_verts'] = verts.contiguous()
        meshdict['mesh_feats'] = feats.contiguous()
        meshdict['mesh_faces'] = tris.contiguous()

        return meshdict

    def rasterize(self, meshdict, camera, bg_color, refer):
        h, w = camera.original_image.shape[1:]
        if refer:
            h*=self.opt['SSAA']
            w*=self.opt['SSAA']

        rast , _ = dr.rasterize(self.rasterizer, meshdict['verts_clip'], meshdict['mesh_faces'], resolution=[h, w])
        attri, _ = dr.interpolate(meshdict['mesh_attri'], rast, meshdict['mesh_faces'])
        meshdict['attri'] = attri
        meshdict['rast']  = rast
        return meshdict
    
    def shading(self, pc, meshdict, camera, bg_color, refer):
        feature = meshdict['attri'][...,: pc.dim_feature]
        gb_pos  = meshdict['attri'][...,  pc.dim_feature : pc.dim_feature + 3]
        normals = F.normalize(meshdict['attri'][...,pc.dim_feature + 3 : pc.dim_feature + 6], dim=-1)
        viewdir = F.normalize(gb_pos - camera.camera_center[None,None,:], dim=-1)

        map_dict = pc.rendernet(viewdir, normals, feature, (meshdict['rast'][..., -1:] > 0).float(), gb_pos, camera.camera_center)
        rgbs     = map_dict['rgb_map']

        background = torch.ones_like(rgbs) * bg_color[0]
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
        I = background

        alpha = (meshdict['rast'][..., -1:] > 0).float()
        I = torch.lerp(I, torch.cat((rgbs, torch.ones_like(alpha)), dim=-1), alpha)
        I = dr.antialias(I.contiguous(), meshdict['rast'], meshdict['verts_clip'], meshdict['mesh_faces'])
        
        if refer:
            ori_h, ori_w = I.shape[1] // self.opt['SSAA'], I.shape[2] // self.opt['SSAA']
            I       = F.interpolate(I.permute(0, 3, 1, 2),   size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            gb_pos  = F.interpolate(gb_pos.permute(0, 3, 1, 2),  size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            normals = F.interpolate(normals.permute(0, 3, 1, 2), size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            alpha   = F.interpolate(alpha.permute(0, 3, 1, 2),   size=(ori_h, ori_w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        meshdict['image']         = I[0]
        meshdict['image_normals'] = (((normals + 1) * 0.5) * alpha)[0]
        meshdict['image_gb_pos']  = (((gb_pos  / self.opt['meshscale']) + 0.5) * alpha)[0]
        
        return meshdict

    def resample(self, pc, meshdict, cameras, bg_color):
        verts    = meshdict['mesh_verts'].detach().cpu().numpy()
        faces    = meshdict['mesh_faces'].detach().cpu().numpy()
        vnormals = meshdict['mesh_v_nrm']

        sign_verts        = torch.zeros(vnormals.shape[0]).to('cuda')
        visiable_faces    = torch.zeros(faces.shape[0]).to('cuda')
        train_background  = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
        
        for camera in cameras:
            meshdict = self.proj2clip(meshdict, camera)
            meshdict = self.rasterize(meshdict, camera, bg_color, False)
            
            # rasterized tris
            tris_idxs = meshdict['rast'][0,:,:,3].reshape(-1)
            tris_idxs = (tris_idxs[tris_idxs > 0] - 1).to(torch.int64)
            tris_idxs = torch.unique(tris_idxs)
            vert_idxs = torch.unique(meshdict['mesh_faces'][tris_idxs, :].reshape(-1)).to(torch.int64)
            visiable_faces[tris_idxs] = 1

            # vert normal flip
            vert_mask = torch.zeros(vnormals.shape[0]).to('cuda')
            vert_mask[vert_idxs] = 1
            vert_mask = vert_mask.to(torch.bool)
            viewdir = F.normalize(camera.camera_center.unsqueeze(0) - meshdict['mesh_verts'], dim=-1)
            sign_vert = (torch.sum(vnormals * viewdir, dim=-1) > 0) * 2 - 1
            sign_verts[vert_mask] += sign_vert[vert_mask]

        # point normal flip and prune non-visiable points
        vnormals = (vnormals * ((sign_verts >= 0)*2 - 1).unsqueeze(-1))
        
        remain_faces = faces[visiable_faces.bool().detach().cpu().numpy()]
        num_resample = remain_faces.shape[0]

        face_verts = verts[remain_faces]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        face_normals = vnormals[remain_faces]
        n0, n1, n2 = face_normals[:, 0], face_normals[:, 1], face_normals[:, 2]

        face_feats = meshdict['mesh_feats'][remain_faces]
        f0, f1, f2 = face_feats[:, 0], face_feats[:, 1], face_feats[:, 2]

        points_center = (v0 + v1 + v2) / 3

        points_normal = (n0 + n1 + n2) / 3
        points_normal = np.asarray(F.normalize(points_normal, dim=-1).detach().cpu())

        points_feature = (f0 + f1 + f2) / 3
        points_feature = np.asarray(points_feature.detach().cpu())

        num_resample      = points_center.shape[0]
        resample_points   = torch.from_numpy(points_center).float().cuda()
        resample_normals  = torch.from_numpy(points_normal).float().cuda()
        resample_feature  = torch.from_numpy(points_feature).float().cuda()

        max_scale  = (self.opt['meshscale'] / (self.opt['grid_resolution'] - 1) * 1.5)
        init_scale = (self.opt['meshscale'] / (self.opt['grid_resolution'] - 1) * 1.0)
        min_scale  = 0.00001

        pc.scale_min_val = torch.ones([num_resample,1]).cuda() * min_scale
        pc.scale_max_val = torch.ones([num_resample,1]).cuda() * max_scale
        pc.scale_range = True
        resample_scaling = inverse_sigmoid(init_scale / pc.scale_max_val)
        resample_gamma_m = torch.full((num_resample, 1), 1.25, device="cuda", dtype=torch.float32)

        pc.prune_points(torch.ones(pc.get_xyz.shape[0]).bool().cuda())
        pc.densify_points({"xyz": resample_points, "normal": resample_normals, "scaling": resample_scaling, "gamma_m": resample_gamma_m, "feature": resample_feature})

        print('visiable mesh point number  : ', pc.get_xyz.shape[0])
    
    def visiable_mesh(self, pc, meshdict, cameras, bg_color):
        verts    = meshdict['mesh_verts'].detach().cpu().numpy()
        faces    = meshdict['mesh_faces'].detach().cpu().numpy()
        vnormals = meshdict['mesh_v_nrm']
        visiable_faces = torch.zeros(faces.shape[0]).to('cuda')

        for camera in cameras:
            meshdict = self.proj2clip(meshdict, camera)
            meshdict = self.rasterize(meshdict, camera, bg_color, False)

            # rasterized tris
            tris_idxs = meshdict['rast'][0,:,:,3].reshape(-1)
            tris_idxs = (tris_idxs[tris_idxs > 0] - 1).to(torch.int64)
            tris_idxs = torch.unique(tris_idxs)
            vert_idxs = torch.unique(meshdict['mesh_faces'][tris_idxs, :].reshape(-1)).to(torch.int64)
            visiable_faces[tris_idxs] = 1

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices  = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        invisiable_faces = (1 - visiable_faces).detach().cpu().numpy().astype(np.bool)
        mesh.remove_triangles_by_mask(invisiable_faces)
        mesh.remove_unreferenced_vertices()

        visiable_mesh = trimesh.Trimesh(vertices = np.asarray(mesh.vertices), faces = np.asarray(mesh.triangles))     
        meshdict['visiable_mesh'] = visiable_mesh
        return meshdict

    def record(self, pc, meshdict, camera , bg_color, record, refer=False):
        with torch.no_grad():
            save_path = os.path.join(self.opt['model_path'], os.path.dirname(record))
            save_name = record.split('/')[-1]
            makedirs(save_path, exist_ok=True)

            gt_image  = camera.original_image.cuda().permute(1,2,0)
            if camera.gt_alpha_mask is not None:
                gt_alpha_mask = camera.gt_alpha_mask.cuda().permute(1,2,0)
                gt_image = gt_image * gt_alpha_mask + bg_color * (1 - gt_alpha_mask)
            gt_image       = (gt_image.cpu().numpy() * 255).astype(np.uint8)
            render_image   = (torch.clamp(meshdict['image'][...,:3], 0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            image_normals  = (meshdict['image_normals'][...,:3].detach().cpu().numpy()   * 255).astype(np.uint8)
            image_gb_pos   = (meshdict['image_gb_pos'][...,:3].detach().cpu().numpy()    * 255).astype(np.uint8)
            all_image      = np.concatenate((gt_image, render_image, image_normals, image_gb_pos), axis=1)
            Image.fromarray(all_image).save(os.path.join(save_path, save_name+'_all.png'))

            if not refer:
                point_scales = pc.get_scaling
                scale_mean   = point_scales.mean().item()
                scale_max_thred = point_scales.max().item()
                point_scales = point_scales / scale_max_thred

                pcd = o3d.geometry.PointCloud()
                pcd.points  = o3d.utility.Vector3dVector(self.denormlize(meshdict['means3D'], pc.axis_center).detach().cpu().numpy())
                pcd.normals = o3d.utility.Vector3dVector(meshdict['normals'].detach().cpu().numpy())

                pcd.colors = o3d.utility.Vector3dVector(point_scales.repeat(1,3).detach().cpu().numpy())
                o3d.io.write_point_cloud(os.path.join(save_path, save_name + "_pc_scale_" + str(round(scale_max_thred, 3)) + '_' + str(round(scale_mean, 3)) + ".ply"), pcd)

                mesh_verts = meshdict['mesh_verts'].detach().cpu().numpy()
                mesh_faces = meshdict['mesh_faces'].detach().cpu().numpy()

                mesh_normal = (F.normalize(meshdict['mesh_v_nrm'], dim=-1) + 1) * 0.5
                mesh_normal = (mesh_normal * 255).detach().cpu().numpy()
        
                mesh = trimesh.Trimesh(vertices = mesh_verts, faces = mesh_faces, vertex_colors= mesh_normal)          
                mesh.export(os.path.join(save_path, save_name + '_mesh_nums.ply'))

    def __call__(self, pc, camera, bg_color, record=None, cameras=None, resample=False, refer=False, get_visiable=False, to_grid=False):
        means3D  = pc.get_xyz
        normals  = pc.get_normal
        scales   = pc.get_scaling
        gamma_m  = pc.get_gamma_m
        features = pc.get_feature
        means3D, scales = self.normalize(means3D, scales, pc.axis_center)
        meshdict = {'means3D': means3D, 'normals': normals, 'scales': scales, 'features': features}

        lap_lam = float(self.opt.get('laplacian_lambda', 0.1)) if self.opt.get('use_laplacian', True) else 0.0
        sp_alpha = float(self.opt.get('sp_alpha', 0.0)) if self.opt.get('stochastic_preconditioning', False) else 0.0
        sp_seed = int(self.opt.get('sp_seed', 0))
        out_sums, out_sdfs, out_feat = splatter.apply(
            means3D, normals, scales, gamma_m, features, self.opt['grid_resolution'], lap_lam, sp_alpha, sp_seed)
        meshdict['out_sums']  = out_sums[0]

        sdf = out_sdfs[0]

        grid_resolution = sdf.shape[0]
        sdf = sdf + (meshdict['out_sums'] == 0) * (1 / (grid_resolution - 1))

        # meshdict['out_sdfs']  = out_sdfs[0]
        meshdict['out_sdfs'] = sdf
        meshdict['out_feat']  = out_feat.permute(1,2,3,0)

        meshdict = self.marchingcube(meshdict, isovalue=0)
        meshdict = auto_normals(meshdict)

        meshdict['mesh_verts'] = self.denormlize(meshdict['mesh_verts'], pc.axis_center)
        meshdict['mesh_attri'] = torch.cat([meshdict['mesh_feats'], meshdict['mesh_verts'], meshdict['mesh_v_nrm']], dim=-1).contiguous()

        meshdict = self.proj2clip(meshdict, camera)
        meshdict = self.rasterize(meshdict, camera, bg_color, refer)
        meshdict = self.shading(pc, meshdict, camera, bg_color, refer)

        if resample:
            self.resample(pc, meshdict, cameras, bg_color)

        if record is not None:
            self.record(pc, meshdict, camera , bg_color, record, refer)

        if get_visiable:
            meshdict = self.visiable_mesh(pc, meshdict, cameras, bg_color)

        return meshdict

