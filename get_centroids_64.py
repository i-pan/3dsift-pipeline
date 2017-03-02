import ml_stuff
import pandas as pd 
import subprocess

# these already exist
labels_file = "/users/ipan/scratch/ml_code/ProTECTIII_Labels_v2.csv" # path to file containing labels 
# note that this script is written as if labels_file has 2 columns:
## id: must match corresponding key file names i.e., 100.key corresponds to id = 100
## pos_OR: label [1/0] for that case
keys_dir = "/users/ipan/scratch/ml_code/neg_tilt/keys/" # directory where keys are located

# these will be written out
train_test_split_out = "/users/ipan/scratch/ml_code/neg_tilt/train_test_split.csv"
pkl_file = "/users/ipan/scratch/ml_code/neg_tilt/km-pickles/mbkm_64_1.pkl"
centroids_csv_file = "/users/ipan/scratch/ml_code/neg_tilt/km-centroids/mbkm_64_1.csv"

labels = pd.read_csv(labels_file)
labels = labels[(labels.exclude == False)]

all_keys = subprocess.check_output("ls " + keys_dir, shell=True)
all_keys = all_keys.split("\n")
all_keys = [i if i != '' else None for i in all_keys]
all_keys = filter(None, all_keys)
all_keys_df = pd.DataFrame({'file': all_keys,
                            'name': [int(i[0:len(i)-4]) for i in all_keys]})

labels = labels.merge(all_keys_df,left_on"id",right_on="name")

train_test_df = ml_stuff.assign_train_test(labels.id, labels.pos_OR, train_test_split_out) 

train_df = train_test_df[(train_test_df.set == "train")]
train_keys = all_keys_df.merge(train_df,left_on="name",right_on="id")

ml_stuff.k_means_cluster(keys_dir, train_keys.file, pkl_file, centroids_csv_file, 64)
