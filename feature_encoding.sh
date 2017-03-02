#!/bin/bash

#SBATCH -n 4
#SBATCH --mem 64G
#SBATCH -t 12:00:00
#SBATCH --out feature_encoding.out

# first variable is path to centroids CSV file
# second is encoding type: vlad or fisher
# third: path to vlfeat directory
# fourth: path to directory containing all keys
# fifth: CSV file to write out feature encodings
matlab-threaded -r "feature_encoding('/users/ipan/scratch/ml_code/all_terarecon_processed/km-centroids/mbkm_cen_64_1.csv', 'vlad', '/users/ipan/MATLAB/vlfeat/', '/users/ipan/scratch/ml_code/all_terarecon_processed/keys/', '/users/ipan/scratch/ml_code/all_terarecon_processed/vlad_64_encoding.csv')"
