import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    ''' Network class containing occupanvy and appearance field
    
    Args:
        cfg (dict): network configs
    '''

    def __init__(self, cfg, **kwargs):
        super().__init__()
        out_dim = 4
        dim = 3
        self.num_layers = cfg['num_layers']
        hidden_size = cfg['hidden_dim']
        self.octaves_pe = cfg['octaves_pe']
        self.octaves_pe_views = cfg['octaves_pe_views']
        self.skips = cfg['skips']
        self.rescale = cfg['rescale']
        self.feat_size = cfg['feat_size']
        geometric_init = cfg['geometric_init'] 

        bias = 0.6

        # init pe
        dim_embed = dim*self.octaves_pe*2 + dim
        dim_embed_view = dim + dim*self.octaves_pe_views*2 + dim + dim + self.feat_size 
        self.transform_points = PositionalEncoding(L=self.octaves_pe)
        self.transform_points_view = PositionalEncoding(L=self.octaves_pe_views)

        ### geo network
        dims_geo = [dim_embed]+ [ hidden_size if i in self.skips else hidden_size for i in range(0, self.num_layers)] + [self.feat_size+1] 
        self.num_layers = len(dims_geo)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]

            lin = nn.Linear(dims_geo[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            
            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

        ## appearance network
        dims_view = [dim_embed_view]+ [ hidden_size for i in range(0, 4)] + [3]

        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def infer_occ(self, p):
        pe = self.transform_points(p/self.rescale)
        x = pe
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, pe], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)     
        return x
    
    def infer_app(self, points, normals, view_dirs, feature_vectors):
        rendering_input = torch.cat([points, view_dirs, normals.squeeze(-2), feature_vectors], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina" + str(l))
            x = lina(x)
            if l < self.num_layers_app - 2:
                x = self.relu(x)
        x = self.tanh(x) * 0.5 + 0.5
        return x

    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_occ(p)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)

    def forward(self, p, ray_d=None, only_occupancy=False, return_logits=False,return_addocc=False, noise=False, **kwargs):
        x = self.infer_occ(p)
        if only_occupancy:
            return self.sigmoid(x[...,:1] * -10.0)
        elif ray_d is not None:
            
            input_views = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            input_views = self.transform_points_view(input_views)
            normals =  self.gradient(p)
            #normals = n / (torch.norm(n, dim=-1, keepdim=True)+1e-6)
            rgb = self.infer_app(p, normals, input_views, x[...,1:])
            if return_addocc:
                if noise:
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
                else: 
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
            else:
                return rgb
        elif return_logits:
            return -1*x[...,:1]


class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)