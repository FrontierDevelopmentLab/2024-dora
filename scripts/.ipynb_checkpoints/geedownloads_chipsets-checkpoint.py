import geetiles
import os
import geopandas as gpd
import pandas as pd
from geetiles import cmds
import argparse
from loguru import logger
from progressbar import progressbar as pbar
from joblib import Parallel, delayed
import numpy as np
from geetiles import utils

#groups = "0,1,L0,L1,L2,L3,L4"
groups = None

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='commands', dest='cmd')

    download_parser = subparsers.add_parser('download', help='download chipsets')

    download_parser.add_argument('--chipsets_folder', required=True, type=str, help='folder containing one geetiles geojson per chipset')
    download_parser.add_argument('--dataset', required=True, type=str, help='gee dataset to download')
    download_parser.add_argument('--pixels_lonlat', default='[512,512]', type=str, help='a tuple, if set, the tile will have this exact size in pixels, regardless the physical size. For instance --pixels_lonlat [100,100]')
    download_parser.add_argument('--shuffle_order', default=False, action='store_true', help='a tuple, if set, the tile will have this exact size in pixels, regardless the physical size. For instance --pixels_lonlat [100,100]')
    download_parser.add_argument('--aoi', default=None, type=str, help="download only predefined aoi. available 'lux', 'eur'")

    summary_parser = subparsers.add_parser('summary', help='summary of chipsets download progress')
    summary_parser.add_argument('--chipsets_folder', required=True, type=str, help='folder containing one geetiles geojson per chipset')
    summary_parser.add_argument('--dataset', required=True, type=str, help='gee dataset to download')
    summary_parser.add_argument('--output_stats_file', default='chipsets_stats.csv', type=str, help='output file for stats')
    summary_parser.add_argument('--project', required=True, type=str, help='google cloud project')

    print ("-----------------------------------------------------------")
    print (f"Google Earth Engine chipsets download utility")
    print ("-----------------------------------------------------------")
    print ()
    args = parser.parse_args()

    if args.cmd=='summary':
        summary(args.chipsets_folder, args.dataset, args.output_stats_file)

    if args.cmd=='download':
        download(args.chipsets_folder, args.dataset, args.pixels_lonlat, args.shuffle_order, args.aoi)

def summary(chipsets_folder, dataset, output_stats_file):

    chipsets_files = [i for i in os.listdir(chipsets_folder) if '_partitions_' in i and i.endswith('.geojson')]
    logger.info(f"summarizing {len(chipsets_files)} chipsets")

    def chipset_file_stats(chipset_file):
        chipset_folder = f"{chipsets_folder}/{chipset_file.split('.')[0]}"
        z = gpd.read_file(f"{chipsets_folder}/{chipset_file}")
        z = z[z.group.isin(groups.split(","))]
        if os.path.isdir(chipset_folder):
            dataset_folder = f"{chipset_folder}/{dataset}"
            dataset_files = [i.split(".")[0] for i in os.listdir(dataset_folder)]
            missing_files = set(z.identifier.values).difference(dataset_files)
            stats = {'chips': len(z), 'missing': len(missing_files) }
        else:
            stats = {'chips': len(z), 'missing': len(z) }
        return stats
            
    r = Parallel(n_jobs=-1, verbose = 2)(delayed(chipset_file_stats)(chipset_file) for chipset_file in chipsets_files)
    r = pd.DataFrame(r, index=chipsets_files)
    print (r.sum())
    r.to_csv(output_stats_file, index=False)
    logger.info (f"stats written to {output_stats_file}")

    logger.info(f"completed chipsets {sum(r.missing==0)} / {len(r)}")
    partially_downloaded = (r.missing>0) & (r.missing<r.chips)
    logger.info(f"partially downloaded chipsets {sum(partially_downloaded)} / {len(r)}")
    print (r[partially_downloaded])


def download(chipsets_folder, dataset, pixels_lonlat, shuffle_order, aoi):

    chipsets_files = [f for f in os.listdir(chipsets_folder) if f.endswith(".geojson")]
    if aoi is not None:
        # use only the chipsets covering the requested aoi
        logger.info(f"getting chipsets for aoi {aoi}")
        utils.aoinames.load()
        aoi_geom = utils.aoinames.get_aoi(aoi)
    
        z = gpd.read_parquet(f"{chipsets_folder}/../chipsets-definitions.parquet")
        z = z[[gi.intersects(aoi_geom) for gi in z.geometry.values]]
        chipsets_files = [i for i in chipsets_files if sum([ci in i for ci in z.index])==1 ]
        logger.info(f"found {len(chipsets_files)} chipsets")

    if shuffle_order:
        chipsets_files = np.random.permutation(chipsets_files)

    for chipset_file in chipsets_files:

        logger.info(f"processing chipset {chipset_file}")
        dataset_folder = f"{chipsets_folder}/{chipset_file.split('.')[0]}/{dataset}"
        z = gpd.read_file(f"{chipsets_folder}/{chipset_file}")
        if os.path.isdir(dataset_folder):
            ntiles_in_geojson = len(z)
            ntiles_in_folder  = len(os.listdir(dataset_folder))
            if ntiles_in_geojson != ntiles_in_folder:
                needs_run = True
            else:
                needs_run = False
            logger.info(f"missing {ntiles_in_geojson-ntiles_in_folder} chips out of {ntiles_in_folder}")
        else:
            needs_run = True
            
        if needs_run:
            logger.info(f"downloading chipset")
            cmds.download(
                tiles_file     = f"{chipsets_folder}/{chipset_file}",
                dataset_def    = dataset,
                pixels_lonlat  = pixels_lonlat,
                skip_confirm   = True,
                skip_if_exists = True,
                ee_auth_mode   = None,
                n_processes    = 20, 
                meters_per_pixel = None, 
                shuffle          = True,
                max_downloads   = None,
                groups          = groups,
                aoi             = aoi
            )
        else:
            logger.info("chipset already fully downloaded")

        
if __name__ == '__main__':
    main()
