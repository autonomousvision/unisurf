import torch
import torch.optim as optim
from torch import autograd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import trimesh
from utils import libmcubes
from .common import make_3d_grid
from utils.libmise import MISE
import time
from skimage.morphology import binary_dilation, disk


#TODO Output masking yes or no
class Extractor3D(object):
    '''  Mesh extractor class for Occupancies

    The class contains functions for exctracting the meshes from a occupancy field

    Args:
        model (nn.Module): trained model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        refine_max_faces (int): max number of faces which are used as batch
            size for refinement process (we added this functionality in this
            work)
    '''

    def __init__(self, model, points_batch_size=100000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1,
                 refine_max_faces=10000):
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = None
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.padding = padding
        self.refine_max_faces = refine_max_faces

    def generate_mesh(self, data=None, return_stats=True, mask_loader=None):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        # inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        # c = self.model.encode_inputs(inputs)
        mesh = self.generate_from_latent(None, stats_dict=stats_dict,
                                         data=None, mask_loader=mask_loader, **kwargs)

        return mesh, stats_dict


    def generate_from_latent(self, c=None, stats_dict={}, data=None,
                             mask_loader=None, **kwargs):
        ''' Generates mesh from latent.

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 2 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,)*3, (0.5,)*3, (nx,)*3
            )
            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(
                self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()

            while points.shape[0] != 0:
                # Query points
                pointsf = torch.FloatTensor(points).to(self.device)
                # Normalize to bounding box
                pointsf = pointsf / mesh_extractor.resolution
                pointsf = box_size * (pointsf - 0.5)
                # Evaluate model and update
                values = self.eval_points(
                    pointsf, c, **kwargs).cpu().numpy()
                
                        
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()


            value_grid = mesh_extractor.to_dense()
            if mask_loader is not None:
                pointsf2 = box_size * make_3d_grid((-0.5,)*3, (0.5,)*3, (value_grid.shape[0],)*3)
                it = 0
                # for data in mask_loader:
                occ = filter_points(pointsf2, mask_loader) > 0.5
                value_grid[~occ.reshape(value_grid.shape)] = -30.0
                print("Masking Iteration: %03d" %it)
                it += 1
        # Extract mesh
        stats_dict['time (eval points)'] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def eval_points(self, p, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                occ_hat = self.model(pi, None, return_logits=True, **kwargs).squeeze(-1)

            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

        occ_hat = torch.cat(occ_hats, dim=0)

        return occ_hat

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 2 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # Undo padding
        vertices -= 1
        # Normalize to bounding box
        vertices /= np.array([n_x-1, n_y-1, n_z-1])
        vertices = box_size * (vertices - 0.5)


        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0
        else:
            normals = None
        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh


    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decoder(vi, None, only_occupancy=True).squeeze(-1)
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces)

        # detach c; otherwise graph needs to be retained
        # caused by new Pytorch version?
        # c = c.detach()

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-5)

        # Dataset
        ds_faces = TensorDataset(faces)
        dataloader = DataLoader(ds_faces, batch_size=self.refine_max_faces,
                                shuffle=True)

        # We updated the refinement algorithm to subsample faces; this is
        # usefull when using a high extraction resolution / when working on
        # small GPUs
        it_r = 0
        while it_r < self.refinement_step:
            for f_it in dataloader:
                f_it = f_it[0].to(self.device)
                optimizer.zero_grad()

                # Loss
                face_vertex = v[f_it]
                eps = np.random.dirichlet((0.5, 0.5, 0.5), size=f_it.shape[0])
                eps = torch.FloatTensor(eps).to(self.device)
                face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

                face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
                face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
                face_normal = torch.cross(face_v1, face_v2)
                face_normal = face_normal / \
                    (face_normal.norm(dim=1, keepdim=True) + 1e-10)

                face_value = torch.cat([
                    # torch.sigmoid(self.model.decode(p_split, c).logits)
                    self.model.decoder(p_split, None, only_occupancy=True).squeeze(-1)
                    for p_split in torch.split(
                        face_point.unsqueeze(0), 20000, dim=1)], dim=1)

                normal_target = -autograd.grad(
                    [face_value.sum()], [face_point], create_graph=True)[0]

                normal_target = \
                    normal_target / \
                    (normal_target.norm(dim=1, keepdim=True) + 1e-10)
                loss_target = (face_value - threshold).pow(2).mean()
                loss_normal = \
                    (face_normal - normal_target).pow(2).sum(dim=1).mean()

                loss = loss_target + 0.01 * loss_normal

                # Update
                loss.backward()
                optimizer.step()

                # Update it_r
                it_r += 1

                if it_r >= self.refinement_step:
                    break

        mesh.vertices = v.data.cpu().numpy()
        return mesh


def filter_points(p, mask_loader):

        # p = torch.from_numpy(p)
    p = p.cpu()
    n_p = p.shape[0]
    inside_mask = np.ones((n_p,), dtype=np.bool)
    inside_img = np.zeros((n_p,), dtype=np.bool)
    # for i in trange(n_images):
        # get data
    for data in mask_loader:
        datai = data
        maski_in = datai.get('img.mask')[0]


        # Apply binary dilation to account for errors in the mask
        maski = torch.from_numpy(binary_dilation(maski_in, disk(12))).float()

        #h, w = maski.shape
        h, w = maski.shape
        w_mat = datai.get('img.world_mat')
        c_mat = datai.get('img.camera_mat')
        s_mat = datai.get('img.scale_mat')

        # project points into image
        phom = torch.cat([p, torch.ones(n_p, 1)], dim=-1).transpose(1, 0)
        proj = c_mat @ w_mat @ s_mat @ phom
        proj = proj[0]
        proj = (proj[:2] / proj[-2].unsqueeze(0)).transpose(1, 0)

        # check which points are inside image; by our definition,
        # the coordinates have to be in [-1, 1]
        mask_p_inside = ((proj[:, 0] >= -1) &
            (proj[:, 1] >= -1) &
            (proj[:, 0] <= 1) &
            (proj[:, 1] <= 1)
        )
        inside_img |= mask_p_inside.cpu().numpy()

        # get image coordinates
        proj[:, 0] = (proj[:, 0] + 1) * (w - 1) / 2.
        proj[:, 1] = (proj[:, 1] + 1) * (h - 1) / 2.
        proj = proj.long()

        # fill occupancy values
        proj = proj[mask_p_inside]
        occ = torch.ones(n_p)
        occ[mask_p_inside] = maski[proj[:, 1], proj[:, 0]]
        inside_mask &= (occ.cpu().numpy() >= 0.5)

    occ_out = np.zeros((n_p,))
    occ_out[inside_img & inside_mask] = 1.
    return occ_out
