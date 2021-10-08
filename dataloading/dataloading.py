import os
import glob
import random
import logging
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import yaml
from torchvision import transforms
from multiprocessing import Manager

logger = logging.getLogger(__name__)

## Adapted from DVR

def get_dataloader(cfg, mode='train', spilt_model_for_images=True, 
                   shuffle=True, with_mask=False):
    ''' Return dataloader instance

    Instansiate dataset class and dataloader and 
    return dataloader
    
    Args:
        cfg (dict): imported config for dataloading
        mode (str): whether dta laoding is used for train or test
        spilt_model_for_images (bool): as name
        shuffle (bool): as name
        with_mask (bool): as name
    
    '''
    
    dataset_folder = cfg['dataloading']['path']
    categories = cfg['dataloading']['classes']
    cache_fields = cfg['dataloading']['cache_fields']
    n_views = cfg['dataloading']['n_views']
    batch_size = cfg['dataloading']['batchsize']
    n_workers = cfg['dataloading']['n_workers']
    return_idx = False

    split = mode

    ## get fields
    fields = get_data_fields(cfg, mode, with_mask=with_mask)
    if return_idx:
        fields['idx'] = data.IndexField()

    ## get dataset
    manager = Manager()
    shared_dict = manager.dict()

    dataset = Shapes3dDataset(
        dataset_folder, fields, split=split,
        categories=categories,
        shared_dict=shared_dict,
        n_views=n_views, cache_fields=cache_fields,
        split_model_for_images=spilt_model_for_images)

    ## dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=n_workers, 
        shuffle=shuffle, collate_fn=collate_remove_none,
    )

    return dataloader


def get_data_fields(cfg, mode='train', with_mask=False):
    ''' Returns the data fields.

    Args:
        cfg (dict): imported yaml config
        mode (str): the mode which is used

    Return:
        field (dict): datafield
    '''
    resize_img_transform = ResizeImage(cfg['dataloading']['img_size'])
    all_images = mode == 'render'
    random_view = True if (
        mode == 'train' 
    ) else False
    n_views = cfg['dataloading']['n_views']
    fields = {}
    if mode in ('train', 'val', 'render', 'test'):
        img_field = ImagesField(
            'image', 
            mask_folder_name='mask',
            transform=resize_img_transform,
            extension='png',
            mask_extension='png',
            with_camera=True,
            with_mask=with_mask,
            random_view=random_view,
            dataset_name='DTU',
            all_images=all_images,
            n_views=n_views,
            ignore_image_idx=cfg['dataloading']['ignore_image_idx'],
        )
        fields['img'] = img_field

    return fields


class ResizeImage(object):
    ''' Resize image transformation class.

    It resizes an image and transforms it to a PyTorch tensor.

    Args:
        img_size (int or tuple): resized image size
    '''
    def __init__(self, img_size):
        if img_size is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])

    def __call__(self, img):
        img = self.transform(img)
        return img



class Shapes3dDataset(data.Dataset):
    '''Dataset class for image data of one 3D shape

    Dataset class that includes caching
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None,
                 shared_dict={}, n_views=24, cache_fields=False,
                 split_model_for_images=False):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            shared_dict (dict): shared dictionary (used for field caching)
            n_views (int): number of views (only relevant when using field
                caching)
            cache_fields(bool): whether to cache fields; this option can be
                useful for small overfitting experiments
            split_model_for_images (bool): whether to split a model by its
                views
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cache_fields = cache_fields
        self.n_views = n_views
        self.cached_fields = shared_dict
        self.split_model_for_images = split_model_for_images

        if split_model_for_images:
            assert(n_views > 0)
            print('You are splitting the models by images. Make sure that you entered the correct number of views.')

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]
        categories.sort()

        # Read metadata file
        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            }

        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx

        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, str(split) + '.lst')

            if not os.path.exists(split_file):
                models_c = [f for f in os.listdir(
                    subpath) if os.path.isdir(os.path.join(subpath, f))]
            else:
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            models_c = list(filter(lambda x: len(x) > 0, models_c))
            
            if split_model_for_images:
                for m in models_c:
                    for i in range(n_views):
                        self.models += [
                            {'category': c, 'model': m,
                                'category_id': c_idx, 'image_id': i}
                        ]
            else:
                self.models += [
                    {'category': c, 'model': m, 'category_id': c_idx}
                    for m in models_c
                ]

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        model_path = os.path.join(self.dataset_folder, category, model)
        data = {}
        for field_name, field in self.fields.items():
            try:
                if self.cache_fields:
                    if self.split_model_for_images:
                        idx_img = self.models[idx]['image_id']
                    else:
                        idx_img = np.random.randint(0, self.n_views)
                    k = '%s_%s_%d' % (model_path, field_name, idx_img)

                    if k in self.cached_fields:
                        field_data = self.cached_fields[k]
                        #print(k)
                    else:
                        
                        field_data = field.load(model_path, idx, c_idx,
                                                input_idx_img=idx_img)
                        self.cached_fields[k] = field_data
                        #print('Not cached %s' %k)
                    
                else:
                    if self.split_model_for_images:
                        idx_img = self.models[idx]['image_id']
                        field_data = field.load(
                            model_path, idx, c_idx, idx_img)
                    else:
                        field_data = field.load(model_path, idx, c_idx)
                        
            except Exception:
                if self.no_except:
                    logger.warn(
                        'Error occurred when loading field %s of model %s (%s)'
                        % (field_name, model, category)
                    )
                    return None
                else:
                    raise


            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data

        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_model_dict(self, idx):
        return self.models[idx]


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


class ImagesField(object):
    ''' Data field for images, masks and cameras

    '''

    def __init__(self, folder_name, mask_folder_name='mask',
                 transform=None, extension='png', mask_extension='png', with_camera=False, 
                 with_mask=False, 
                 random_view=True, all_images=False, n_views=0,
                 ignore_image_idx=[], **kwargs):
        self.folder_name = folder_name
        self.mask_folder_name = mask_folder_name

        self.transform = transform

        self.extension = extension
        self.mask_extension = mask_extension

        self.random_view = random_view
        self.n_views = n_views

        self.with_camera = with_camera
        self.with_mask = with_mask

        self.all_images = all_images
        self.ignore_image_idx = ignore_image_idx


    def load(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the field.

        Args:
            model_path (str): path to model
            idx (int): model id
            category (int): category id
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        '''
        if self.all_images:
            n_files = self.get_number_files(model_path)
            data = {}
            for input_idx_img in range(n_files):
                datai = self.load_field(model_path, idx, category,
                                        input_idx_img)
                data['img%d' % input_idx_img] = datai
            data['n_images'] = n_files
            return data
        else:
            return self.load_field(model_path, idx, category, input_idx_img)

    def get_number_files(self, model_path, ignore_filtering=False):
        ''' Returns how many views are present for the model.

        Args:
            model_path (str): path to model
            ignore_filtering (bool): whether the image filtering should be
                ignored
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()

        if not ignore_filtering and len(self.ignore_image_idx) > 0:
            files = [files[idx] for idx in range(
                len(files)) if idx not in self.ignore_image_idx]

        if not ignore_filtering and self.n_views > 0:
            files = files[:self.n_views]
        return len(files)

    def return_idx_filename(self, model_path, folder_name, extension, idx):
        ''' Loads the "idx" filename from the folder.

        Args:
            model_path (str): path to model
            folder_name (str): name of the folder
            extension (str): string of the extension
            idx (int): ID of data point
        '''
        folder = os.path.join(model_path, folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % extension))
        files.sort()
        if len(self.ignore_image_idx) > 0:
            files = [files[idx] for idx in range(
                len(files)) if idx not in self.ignore_image_idx]

        if self.n_views > 0:
            files = files[:self.n_views]
        return files[idx]

    def load_image(self, model_path, idx, data={}):
        ''' Loads an image.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''

        filename = self.return_idx_filename(model_path, self.folder_name,
                                            self.extension, idx)
        image = Image.open(filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.with_mask:
            filename_m = self.return_idx_filename(
                model_path, self.mask_folder_name, self.mask_extension, idx)
            mask = np.array(Image.open(filename_m)).astype(np.bool)
            mask = mask.reshape(mask.shape[0], mask.shape[1], -1)
            mask = mask[:, :, 0]
            mask = mask.astype(np.float32)
            image = image * mask + (1 - mask) * np.ones_like(image)

        data[None] = image
        data['idx'] = idx
    
    def load_camera(self, model_path, idx, data={}):
        ''' Loads an image.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''

        if len(self.ignore_image_idx) > 0:
            n_files = self.get_number_files(model_path, ignore_filtering=True)
            idx_list = [i for i in range(
                n_files) if i not in self.ignore_image_idx]
            idx_list.sort()
            idx = idx_list[idx]

        camera_file = os.path.join(model_path, 'cameras.npz')
        camera_dict = np.load(camera_file)
        Rt = camera_dict['world_mat_%s' % idx].astype(np.float32)
        K = camera_dict['camera_mat_%s' % idx].astype(np.float32)
        S = camera_dict.get(
            'scale_mat_%s' % idx, np.eye(4)).astype(np.float32)
        
        data['world_mat'] = Rt
        data['camera_mat'] = K
        data['scale_mat'] = S

    def load_mask(self, model_path, idx, data={}):
        ''' Loads an object mask.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        filename = self.return_idx_filename(
            model_path, self.mask_folder_name, self.mask_extension, idx)
        mask = np.array(Image.open(filename)).astype(np.bool)
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[:, :, 0]
        data['mask'] = mask.astype(np.float32)


    def load_field(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        '''

        

        if input_idx_img is not None:
            idx_img = input_idx_img
        elif self.random_view:
            n_files = self.get_number_files(model_path)
            idx_img = random.randint(0, n_files - 1)
        else:
            idx_img = 0
        # Load the data
        data = {}
        self.load_image(model_path, idx_img, data)
        if self.with_camera:
            self.load_camera(model_path, idx_img, data)
            
        if self.with_mask:
            self.load_mask(model_path, idx_img, data)
        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        return complete