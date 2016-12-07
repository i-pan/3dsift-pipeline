import ml_stuff
import pandas as pd 
import subprocess

labels_file = "/users/ipan/scratch/ml_code/ProTECTIII_Labels_v2.csv")
train_test_split_out = "/users/ipan/scratch/ml_code/neg_tilt/train_test_split.csv"
keys_dir = "/users/ipan/scratch/ml_code/neg_tilt/keys/"
pkl_file = "/users/ipan/scratch/ml_code/neg_tilt/km-pickles/mbkm_64_1.pkl"
centroids_csv_file = "/users/ipan/scratch/ml_code/neg_tilt/km-centroids/mbkm_64_1.csv"

labels = pd.read_csv(labels_file)
labels = labels[(labels.exclude == False)]

train_test_df = ml_stuff.assign_train_test(labels.id, labels.pos_OR, train_test_split_out) 

all_keys = subprocess.check_output("ls " + keys_dir, shell=True)
all_keys = all_keys.split("\n")
all_keys = [i if i != '' else None for i in all_keys]
all_keys = filter(None, all_keys)

all_keys_df = pd.DataFrame({'file': all_keys,
                            'name': [int(i[0:len(i)-4]) for i in all_keys]})

train_df = train_test_df[(train_test_df.set == "train")]
train_keys = all_keys_df.merge(train_df,left_on="name",right_on="id")

ml_stuff.k_means_cluster(keys_dir, train_keys.file, pkl_file, centroids_csv_file, 64)
