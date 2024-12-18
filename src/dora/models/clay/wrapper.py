import torch
import loguru
from .module import ClayMAEModule
import numpy as np
from einops import rearrange, reduce, repeat

logger = loguru.logger

class ClayWrapper:

    def __init__(self, metadata_path, checkpoint_path):


        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        self.metadata_path = metadata_path
        
        logger.info(f'using device {self.device}')

        logger.info('creating clay model instance')
        self.clay_model = ClayMAEModule(
             model_size= 'large',
             mask_ratio= 0.75,
             norm_pix_loss= False,
             patch_size= 8,
             shuffle = True,
             metadata_path = '/opt/claymodel-src/configs/metadata.yaml',
             teacher = 'vit_large_patch14_reg4_dinov2.lvd142m',
             dolls = [16, 32, 64, 128, 256, 768, 1024],
             doll_weights = [1, 1, 1, 1, 1, 1, 1],
             lr = 5e-06,
             wd = 0.05,
             b1 = 0.9,
             b2 = 0.95,
             embeddings_level = 'mean',).to(self.device)

        logger.info('loading clay model weights') 
        z = torch.load(checkpoint_path, weights_only=False, map_location=torch.device(self.device))
        self.clay_model.load_state_dict(z['state_dict'])
        
        # mean and stds for normalization of RGB channels
        self.means = np.array([53.33853489, 44.41999383, 35.96075039])
        self.stds  = np.array([50.44633167, 43.54469652, 44.63162242])

        logger.info('done')
        
    def batch_embeddings(self, batch):
        """
        batch: [batch_size, 12, img_size, img_size]
               12 for [red_winter, green_winter, blue_winter, red_spring, ... summer, fall]
        """
        if not batch.shape[1]==12:
            raise ValueError(f'expecting 12 channels (rgb winter spring summer fall), but found {batch.shape[1]}')
        image_size = batch.shape[-1]
        batch = np.r_[[img[3*i:3*(i+1)] for img in batch for i in range(4)]]
        
        batch_normalized = np.transpose((np.transpose(batch, [0,2,3,1]) - self.means) / self.stds, [0,3,1,2])

        x = {'pixels': torch.tensor(batch_normalized).type(torch.float),
             'time': torch.zeros([len(batch_normalized), 4]),
             'latlon': torch.zeros([len(batch_normalized), 4]),
             'gsd': torch.tensor(10.),
             'waves': torch.tensor([1552., 1355., 1105.])} # rgb freqs

        with torch.no_grad():
            embeddings_raw, *_ = self.clay_model.model.encoder(x)
        
        patch_size = self.clay_model.model.patch_size
        # compute patch and image embeddings
        patch_embeddings = rearrange(
            embeddings_raw[:, :-1, :], # :-1; last embedding is the cls_token
            "b (h w) d -> b h w d",
            w=image_size//patch_size//2,
            h=image_size//patch_size//2,
        )
        # image embeddings
        e = reduce(patch_embeddings, "b h w d -> b d", "mean")
        e = torch.stack([e[i*4:(i+1)*4] for i in range(len(e)//4)])
        return e

