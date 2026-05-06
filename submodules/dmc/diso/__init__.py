import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from . import _C

class DiffMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            mc = _C.CUMCFloat()
        elif dtype == torch.float64:
            mc = _C.CUMCDouble()

        class DMCFunction(Function):
            @staticmethod
            def forward(ctx, sdfsgrid, featgrid, isovalue, dim_feats):
                verts, feats, tris = mc.forward(sdfsgrid, featgrid, isovalue, dim_feats)

                ctx.isovalue  = isovalue
                ctx.dim_feats = dim_feats
                ctx.save_for_backward(sdfsgrid, featgrid)
                return verts, feats, tris

            @staticmethod
            def backward(ctx, adj_verts, adj_feats, adj_faces):
                sdfsgrid, featgrid = ctx.saved_tensors
                adj_sdfsgrid = torch.zeros_like(sdfsgrid).contiguous()
                adj_featgrid = torch.zeros_like(featgrid).contiguous()
                mc.backward(
                    sdfsgrid, featgrid, ctx.isovalue, ctx.dim_feats, 
                    adj_verts.contiguous(), adj_feats.contiguous(), 
                    adj_sdfsgrid, adj_featgrid
                )
                return adj_sdfsgrid, adj_featgrid, None, None

        self.func = DMCFunction

    def forward(self, sdfsgrid, featgrid, isovalue=0.0, normalize=True):
        dim_feats = featgrid.shape[-1]
        if sdfsgrid.min() >= isovalue or sdfsgrid.max() <= isovalue:
            return  torch.zeros((0, 3), dtype=self.dtype, device=sdfsgrid.device), \
                    torch.zeros((0, dim_feats), dtype=self.dtype , device=sdfsgrid.device), torch.zeros((0, 3), dtype=torch.int32, device=sdfsgrid.device)
        dimX, dimY, dimZ = sdfsgrid.shape
        sdfsgrid = F.pad(sdfsgrid, (1, 1, 1, 1, 1, 1),       "constant", isovalue+1)
        featgrid = F.pad(featgrid, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)

        verts, feats, tris = self.func.apply(sdfsgrid, featgrid, isovalue, dim_feats)
        verts = verts - 1
        if normalize:
            verts = verts / (torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1)
        return verts, feats, tris

