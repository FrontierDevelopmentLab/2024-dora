CHIPSETS_FOLDER=/opt/buckets/disasterbrain-data/chips

DATASET=s2rgb-2024
PIXELS_LONLAT='--pixels_lonlat [512,512]'

#DATASET=s1count-2017
#PIXELS_LONLAT='--pixels_lonlat [512,512]'

#DATASET=s1grdobs-202201-asc
#AOI='--aoi lux'

#DATASET=s2_13bands-2021
#AOI='--aoi lux'

SHUFFLE=--shuffle_order
PROJECT=esl-sar-fm
python geedownload_chipsets.py download --chipsets_folder $CHIPSETS_FOLDER --dataset $DATASET $AOI $PIXELS_LONLAT $SHUFFLE
