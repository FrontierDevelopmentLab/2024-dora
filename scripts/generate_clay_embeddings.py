# ------------------------------------------------------------------
# CONFIG

FILES_SPEC      = '/opt/buckets/disasterbrain-data/chips/chipsets/*/s2rgb-2024/*.tif'
batch_size      = 4
CHECKPOINT_PATH = "/opt/claymodel-weights/clay-v1.5.ckpt"
METADATA_PATH   = "/opt/claymodel-weights/metadata.yaml"
N_SAMPLES       = 100
DEST_DIR        = '/opt/claymodel-embeddings'
# ------------------------------------------------------------------

import os
import pickle
import sys
import warnings
warnings.filterwarnings(action="ignore")
sys.path.append('../src')

from dora.models.clay import wrapper
from dora.data import norm

import numpy as np
from glob import glob
import rasterio
from loguru import logger
from joblib import Parallel, delayed
from progressbar import progressbar as pbar


# ------------------------------------------------------------------
# SETUP

# load clay model
cw = wrapper.ClayWrapper(metadata_path=METADATA_PATH, checkpoint_path=CHECKPOINT_PATH)

# sample files to get embeddings form
logger.info(f'looking for files under {FILES_SPEC}')
files = np.r_[glob(FILES_SPEC)]
logger.info(f'found {len(files)} files')


# ------------------------------------------------------------------
# SAMPLE FILES TO GET THE EMBEDDINGS FOR

_sampled_files = files if N_SAMPLES is None else np.random.permutation(files)[:N_SAMPLES]
logger.info(f'considering a sample of {len(_sampled_files)} files')

sampled_files = []
# do only the files for which there are no embeddings
for file in _sampled_files:
    chipset = file.split('/')[-3]
    chip = file.split('/')[-1].split('.')[0]

    chipset_folder = f'{DEST_DIR}/{chipset}'
    embeddings_file = f'{chipset_folder}/{chip}.pkl'    

    if not os.path.isfile(embeddings_file):
        sampled_files.append(file)
    
logger.info(f'getting {len(_sampled_files)} new embeddings')


# ------------------------------------------------------------------
# GET THE EMBEDDINGS
nsteps = len(sampled_files) // batch_size
if nsteps % batch_size != 0:
    nsteps += 1
    
for j in pbar(range(nsteps)):

    batch_files = sampled_files[j*batch_size:(j+1)*batch_size]

    # assemble batch 
    def loadimg(file):
        with rasterio.open(file) as f:
            x = f.read()
        return x
        
    batch = np.r_[Parallel(n_jobs=-1, verbose=0)(delayed(loadimg)(file) for file in batch_files)]
    
    # get embeddings
    e = cw.batch_embeddings(batch).numpy()

    # write the embeddings of this batch
    for ii, file in enumerate(batch_files):
        
        chipset = file.split('/')[-3]
        chip = file.split('/')[-1].split('.')[0]
    
        chipset_folder = f'{DEST_DIR}/{chipset}'
        embeddings_file = f'{chipset_folder}/{chip}.pkl'    
    
        if not os.path.isdir(chipset_folder):
            os.makedirs(chipset_folder, exist_ok=True)
    
    
        ei = {'winter': e[ii,0], 'spring': e[ii,1], 'summer': e[ii,2], 'fall': e[ii,3]}
    
        with open(embeddings_file, 'wb') as f:
            pickle.dump(ei, f)

    



