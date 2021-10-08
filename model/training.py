import os
import torch
from collections import defaultdict
from model.common import (
    get_tensor_values, sample_patch_points, arange_pixels
)
from tqdm import tqdm
import logging
from model.losses import Loss
import numpy as np
logger_py = logging.getLogger(__name__)
from PIL import Image


class Trainer(object):
    ''' Trainer object for the UNISURF.

    Args:
        model (nn.Module): model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): config file
        device (device): pytorch device
    '''

    def __init__(self, model, optimizer, cfg, device=None, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.n_training_points = cfg['n_training_points']
        self.n_eval_points = cfg['n_training_points']
        self.overwrite_visualization = True

        self.rendering_technique = cfg['type']

        self.loss = Loss(
            cfg['lambda_l1_rgb'], 
            cfg['lambda_normals'],
            cfg['lambda_occ_prob']
        )

    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)
        
        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.model.train()
        self.optimizer.zero_grad()

        loss_dict = self.compute_loss(data, it=it)
        loss = loss_dict['loss']
        loss.backward()
        self.optimizer.step()
        return loss_dict

    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict = {}
        #with torch.no_grad():
        try:
            eval_dict = self.compute_loss(
                data, eval_mode=True)
        except Exception as e:
            print(e)

        for (k, v) in eval_dict.items():
            eval_dict[k] = v.item()

        return eval_dict
    
    def render_visdata(self, data, resolution, it, out_render_path):
        (img, mask, world_mat, camera_mat, scale_mat, img_idx) = \
            self.process_data_dict(data)
        h, w = resolution
        
        p_loc, pixels = arange_pixels(resolution=(h, w))

        pixels = pixels.to(self.device)

        with torch.no_grad():
            mask_pred = torch.ones(pixels.shape[0], pixels.shape[1]).bool()

            rgb_pred = \
                [self.model(
                    pixels_i, camera_mat, world_mat, scale_mat, 'unisurf', 
                    add_noise=False, eval_=True, it=it)['rgb']
                    for ii, pixels_i in enumerate(torch.split(pixels, 1024, dim=1))]
           
            rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
            p_loc1 = p_loc[mask_pred]
            img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)

            if mask_pred.sum() > 0:
                rgb_hat = rgb_pred[mask_pred].detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '%04d_unisurf.png' % img_idx)
            )

        with torch.no_grad():
            mask_pred = torch.ones(pixels.shape[0], pixels.shape[1]).bool()

            rgb_pred = \
                [self.model(
                    pixels_i, camera_mat, world_mat, scale_mat, 'phong_renderer', 
                    add_noise=False, eval_=True, it=it)['rgb']
                    for ii, pixels_i in enumerate(torch.split(pixels, 1024, dim=1))]
           
            rgb_pred = torch.cat(rgb_pred, dim=1).cpu()
            p_loc1 = p_loc[mask_pred]
            img_out = (255 * np.zeros((h, w, 3))).astype(np.uint8)

            if mask_pred.sum() > 0:
                rgb_hat = rgb_pred[mask_pred].detach().cpu().numpy()
                rgb_hat = (rgb_hat * 255).astype(np.uint8)
                img_out[p_loc1[:, 1], p_loc1[:, 0]] = rgb_hat
        
            img1 = Image.fromarray(
                (img_out).astype(np.uint8)
            ).convert("RGB").save(
                os.path.join(out_render_path, '%04d_phong.png' % img_idx)
            )
            
        return img_out.astype(np.uint8)

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        device = self.device

        # Get "ordinary" data
       
        img = data.get('img').to(device)
        img_idx = data.get('img.idx')
        batch_size, _, h, w = img.shape
        mask_img = data.get('img.mask', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)

        return (img, mask_img, world_mat, camera_mat, scale_mat, img_idx)

    def compute_loss(self, data, eval_mode=False, it=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        n_points = self.n_eval_points if eval_mode else self.n_training_points
        (img, mask_img, world_mat, camera_mat, scale_mat, img_idx) = self.process_data_dict(data)

        # Shortcuts
        device = self.device
        batch_size, _, h, w = img.shape

        # Assertions
        assert(((h, w) == mask_img.shape[2:4]) and
               (n_points > 0))

        # Sample pixels
        if n_points >= h*w:
            p = arange_pixels((h, w), batch_size)[1].to(device)
            mask_gt = mask_img.bool().reshape(-1)
            pix = None
        else:
            p, pix = sample_patch_points(batch_size, n_points,
                                    patch_size=1.,
                                    image_resolution=(h, w),
                                    continuous=False,
                                    )
            p = p.to(device) 
            pix = pix.to(device) 
            mask_gt = get_tensor_values(mask_img, pix.clone()).bool().reshape(-1)

        out_dict = self.model(
            p, camera_mat, world_mat, scale_mat, 
            self.rendering_technique, it=it, mask=mask_gt, 
            eval_=eval_mode
        )
        
        rgb_gt = get_tensor_values(img, pix.clone())
        loss_dict = self.loss(out_dict['rgb'], rgb_gt, out_dict['normal'])
        return loss_dict
