import ml_stuff
import pandas as pd 

# sklearn stuff
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC 

lr_lib = {'clf': LogisticRegression,
          'param_dict': {'C': [1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4], 
                         'penalty': ['l1', 'l2'],
                         'class_weight': [None, 'balanced'],
                         'n_jobs': [-1]
                        }
         }

nb_lib = {'clf': GaussianNB,
          'param_dict': {}
         }

ld_lib = {'clf': LinearDiscriminantAnalysis,
          'param_dict': {}
         }

sv_lib = {'clf': SVC, 
          'param_dict': {'kernel': ['linear', 'rbf'], 
                         'class_weight': [None, 'balanced'],
                         'C': [1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4], 
                         'gamma': ['auto', 1e-6, 1e-4, 1e-2, 1, 10]
                        }
         }

clf_library = [lr_lib, nb_lib, ld_lib, sv_lib]

features = pd.read_csv("/users/ipan/scratch/ml_code/neg_tilt/vlad_64_encoding.csv", header=None)
# I know that the last column is the id, so I'm going to add column names
col_names = ['x'+str(i) for i in range(1,features.shape[1])]
col_names.append('id')
features.columns = col_names
labels = pd.read_csv("/users/ipan/scratch/ml_code/neg_tilt/train_test_split.csv")
data = features.merge(labels, on="id")
train_data = data[(data.set == "train")]
X = train_data[['x'+str(i) for i in range(1,features.shape[1])]]
y = train_data.label
eval_dct, pkl_dct = ml_stuff.machine_learner(X,y, clf_library, n_folds=5, verbose=True)

eval_df = ml_stuff.dict_to_dataframe(eval_dct, pkl_dct).sort(columns=['roc_auc_mean', 'acc_mean'], ascending=False)
eval_df.to_csv("eval_df_vlad_64.csv")
