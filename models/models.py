import torch
import numpy as np
import torch.nn.functional as F

def positional_encoding(positions, freqs):
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts
        
class rendernet(torch.nn.Module):
    def __init__(self, num_theta = 8, num_phi=16, dim_feature=192, featureC=256, device='cpu', 
                 view_pe=-1, fea_pe=-1, btn_freq=[]):
        super(rendernet, self).__init__()

        self.ch_cd = 3
        self.ch_s = 3
        self.ch_normal = 3
        self.ch_bottleneck = 128

        self.ch_normal_dot_viewdir = 1
        self.view_pe = view_pe
        self.fea_pe = fea_pe
        
        if len(btn_freq) >= 2:
            self.btn_freq = torch.linspace(np.log2(btn_freq[0]), np.log2(btn_freq[1]), 
                                            self.ch_bottleneck, device=device)
            self.btn_freq = torch.exp2(self.btn_freq)
        else:
            self.btn_freq = None

        in_dim = dim_feature
        if self.fea_pe > 0:
           in_dim += dim_feature * self.fea_pe * 2
           
        spatial_layers   = [torch.nn.Linear(in_dim, featureC), torch.nn.GELU(),
                           torch.nn.Linear(featureC, featureC), torch.nn.GELU(),
                           torch.nn.Linear(featureC, self.ch_cd + self.ch_s + self.ch_bottleneck)]
        self.spatial_mlp = torch.nn.Sequential(*spatial_layers).to(device)

        in_dim = self.ch_bottleneck + self.ch_normal_dot_viewdir
        if self.view_pe > -1:
            in_dim += 6
        if self.view_pe > 0:
            in_dim += 6 * self.view_pe * 2
            
        directional_layers  = [torch.nn.Linear(in_dim, featureC), torch.nn.GELU(),
                               torch.nn.Linear(featureC, featureC), torch.nn.GELU(),
                               torch.nn.Linear(featureC, 3)]
        self.directional_mlp = torch.nn.Sequential(*directional_layers).to(device)
        
        print(self.directional_mlp)

    def spatial_mlp_forward(self, x):
        out = self.spatial_mlp(x)
        sections = [self.ch_cd, self.ch_s, self.ch_bottleneck]
        diffuse_color, tint, bottleneck = torch.split(out, sections, dim=-1)
        return diffuse_color, tint, bottleneck

    def directional_mlp_forward(self, x):
        out = self.directional_mlp(x)
        return out

    def reflect(self, viewdir, normal):
        out = 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir
        return out

    def forward(self, viewdir_all, normal_all, feature_all, alpha, gb_pos_all=None, camera_center=None):
        B, H, W = viewdir_all.size()[:3]

        mask = alpha.reshape(-1) > 0
        viewdir = viewdir_all.reshape(B*H*W, -1)[mask, :]
        normal  = normal_all.reshape(B*H*W,  -1)[mask, :]
        feature = feature_all.reshape(B*H*W, -1)[mask, :]

        spa_mlp_input = [feature]
        if self.fea_pe > 0:
            spa_mlp_input += [positional_encoding(feature, self.fea_pe)]
        spa_mlp_input = torch.cat(spa_mlp_input, dim=-1)
        diffuse_color, tint, bottleneck = self.spatial_mlp_forward(spa_mlp_input)

        refdir = self.reflect(-viewdir, normal)
        
        if self.btn_freq is not None:
            bottleneck = bottleneck + torch.sin(bottleneck * self.btn_freq[None])

        normal_dot_viewdir = ((-viewdir) * normal).sum(dim=-1, keepdim=True)
        dir_mlp_input = [bottleneck, normal_dot_viewdir]
        
        if self.view_pe > -1:
            dir_mlp_input += [viewdir]
        if self.view_pe > 0:
            dir_mlp_input += [positional_encoding(viewdir, self.view_pe)]
        if self.view_pe > -1:
            dir_mlp_input += [refdir]
        if self.view_pe > 0:
            dir_mlp_input += [positional_encoding(refdir, self.view_pe)]

        dir_mlp_input = torch.cat(dir_mlp_input, dim=-1)
        specular_color = self.directional_mlp_forward(dir_mlp_input)

        raw_rgb = diffuse_color + tint * specular_color
        rgb = torch.sigmoid(raw_rgb)
        
        rgb_map = torch.zeros(B*H*W, 3, device=rgb.device)
        rgb_map[mask, :] += rgb
        rgb_map = rgb_map.reshape(B, H, W, 3)
        
        return {'rgb_map': rgb_map}