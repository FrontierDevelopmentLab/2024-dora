import numpy as np
from progressbar import progressbar as pbar
from joblib import Parallel, delayed
import rasterio
from loguru import logger
from glob import glob

class StatsRecorder:
    def __init__(self, data=None):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if data is not None:
            data = np.atleast_2d(data)
            self.mean = data.mean(axis=0)
            self.std  = data.std(axis=0)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            self.nobservations = 0

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        if self.nobservations == 0:
            self.__init__(data)
        else:
            data = np.atleast_2d(data)
            if data.shape[1] != self.ndimensions:
                raise ValueError("Data dims don't match prev observations.")

            newmean = data.mean(axis=0)
            newstd  = data.std(axis=0)

            m = self.nobservations * 1.0
            n = data.shape[0]

            tmp = self.mean

            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = np.sqrt(self.std)

            self.nobservations += n

def compute_normalization_constants(file_spec, n_samples=1024):
    """
    computes the mean and std per rgb channel on s2rgb data containing 12 channels (rgb for each season)

    file_spec: the glob spec of files. for instance '/opt/buckets/disasterbrain-data/chips/chipsets/*/s2rgb-2024/*.tif'
    """

    logger.info(f'looking for files under {file_spec}')
    files = np.r_[glob(file_spec)]    

    logger.info(f'found {len(files)} files')
    logger.info(f'sampling and reading {n_samples} files in parallel')
    batch_idxs = np.random.permutation(len(files))[:n_samples]
    batch_files = files[batch_idxs]

    def loadimg(file):
        with rasterio.open(file) as f:
            x = f.read()
            d = f.descriptions
        return x
        
    imgs = Parallel(n_jobs=-1, verbose=5)(delayed(loadimg)(file) for file in batch_files)

    s = StatsRecorder()

    logger.info('computing mean and std')
    for x in pbar(imgs):
            
        # get all pixels from all seasons for each channel
        rgb = np.r_[[x[i::3].reshape(-1) for i in range(3)]].T
        
        # remove pixels with zero value
        rgb = rgb[(rgb==0).sum(axis=1)==0]
        
        # remove pixels with nan
        rgb = rgb[np.isnan(rgb).sum(axis=1)==0]
        
        if len(rgb)>0:
            s.update(rgb)

    return {'mean': s.mean, 'std': s.std}
