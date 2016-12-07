import pandas as pd 
import numpy as np 
import os 
import datetime 
from itertools import izip_longest, product

# sklearn stuff
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics 
from sklearn.externals import joblib 
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def assign_train_test(ids, labels, out_path, test_size=0.2): 
    """
    Assigns each row to training or test set
    Only for binary (1/0) labels
    Will return stratified split 
    -- ids, labels should be lists/numpy arrays and correspond to each other 
    -- i.e., ids[0] correponds to labels[0], etc.
    """
    X = np.array(ids) 
    y = np.array(labels) 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=0)
    for train_index, test_index in sss.split(X,y):
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index]
    train_df = pd.DataFrame({'id': X_train, 
                             'label': y_train, 
                             'set': np.repeat("train", len(X_train))})
    test_df = pd.DataFrame({'id': X_test, 
                            'label': y_test,
                            'set': np.repeat("test", len(X_test))})
    train_test_df = train_df.append(test_df) 
    train_test_df.to_csv(out_path, index=False)
    return (train_test_df)

def k_means_cluster(parent_path, key_list, pkl_path, centroid_path, n_clusters, 
                    mb=True, init_size=1e6, batch_size=1e5, n_init=10):
    """
    parent_path = path to directory containing .key files 
    key_list = list of .key files WITH EXTENSION
    pkl_path = path to pickle file for kmeans model 
    """ 

    total_start = datetime.datetime.now()
    if mb:
        print("Starting MINI-BATCH K-MEANS with "+str(n_clusters)+" clusters ...")
        km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                             init_size=init_size, n_init=n_init)
    else:
        print("Starting K-MEANS with "+str(n_clusters)+" clusters ...")
        km = KMeans(n_clusters=n_clusters, n_init=n_init)

    read_start = datetime.datetime.now() 
    print("Reading in IMAGE FEATURES ...")
    key_mat = pd.DataFrame()
    for each_key in key_list:
        temp_key = pd.read_table(parent_path+"/"+each_key,header=None,
                                 skiprows=6).ix[:,17:80]
        key_mat = key_mat.append(temp_key)
    key_mat = np.array(key_mat) 
    read_runtime = datetime.datetime.now()-read_start 
    print("Reading took " + str(read_runtime))
    print("COMBINED IMAGE FEATURES are of dimensions: "+str(key_mat.shape))

    fit_start = datetime.datetime.now()
    print("Fitting model ...")
    km.fit(key_mat[:,0:64])
    fit_runtime = datetime.datetime.now()-fit_start
    print("Fitting took " + str(fit_runtime))

    print("Pickling model ...")
    if os.path.exists(os.path.dirname(pkl_path)) is False:
        os.system("mkdir " + os.path.dirname(pkl_path))
    joblib.dump(km,pkl_path,compress=3)
    if os.path.exists(os.path.dirname(centroid_path)) is False: 
        os.system("mkdir " + os.path.dirname(centroid_path)) 
    print("Writing centroids ...")
    pd.DataFrame(km.cluster_centers_).to_csv(centroid_path, header=False,
                                             index=False)
    total_runtime = datetime.datetime.now()-total_start
    print("DONE in " + str(total_runtime) + " // model saved in " + pkl_path)

def generate_models(clf_library):
    """
    This function returns a list of classifiers with all combinations of
    hyperparameters specified in the dictionary of hyperparameter lists.
    usage example:
        lr_dict = {
                      'clf': LogisticRegression,
                      'param_dict': {
                           'C': [0.001, 0.1, 1, 10],
                           'penalty': ['l1', 'l2']
                           }
                  }
        sgd_dict = {
                       'clf': SGDClassifier,
                       'param_dict': {
                       'alpha': [0.0001, 0.001, 0.01, 0.1],
                       'penalty': ['l1', 'l2']
                       }
                   }
        clf_library = [lr_dict, sgd_dict]
        generate_models(clf_library)
    """
    clf_list = []
    for i in clf_library:
        param_dict = i['param_dict']
        dict_list = [dict(izip_longest(param_dict, v)) for v in product(*param_dict.values())]
        clf_list = clf_list+[i['clf'](**param_set) for param_set in dict_list]
    return clf_list

def machine_learner(X,y, clf_library, n_folds, pkl='pickles', verbose=False,
                    thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]): 
    """
    This function trains models using K-fold cross-validation. 
    Outputs models + metrics.
    """

    # get start time for runtime calculation
    total_start_time = datetime.datetime.now()

    # make sure clf_library is iterable (a list)
    if type(clf_library) is not list: clf_library = [clf_library]
    clf_library = generate_models(clf_library)

    # make folder for pickle files if it doesn't exist
    if pkl[-1] != '/': pkl += '/'
    if not os.path.exists(pkl): os.makedirs(pkl)

    eval_dct = {} 
    pkl_dct = {} 

    for i,clf in enumerate(clf_library): 
        if verbose: 
            print('Running '+str(clf)+'\n...')
        start_time = datetime.datetime.now()
        eval_dct[str(clf)] = kfold_cv(X,y, clf, n_folds, pkl, thresholds, i=i)
        clf_name = str(clf)[:str(clf).index('(')]
        pkl_file_path = pkl+clf_name+str(i)+'.pkl'
        clf.fit(X,y)
        joblib.dump(clf, pkl_file_path,
                    compress=3) # dump pickle file 
        pkl_dct[str(clf)] = pkl_file_path 
        if verbose: print('Finished in: '+str(datetime.datetime.now()-
                                              start_time)+'\n')

    print('machine_learner: finished running models')
    print('machine_learner: pickle files available in ' + pkl)
    total_rtime = str(datetime.datetime.now()-total_start_time)
    print('machine_learner: total runtime was '+total_rtime)

    return eval_dct, pkl_dct 

def kfold_cv(X,y, clf, n_folds, pkl, thresholds, i):
    """
    Do K-fold CV 
    """ 
    X = np.array(X).astype(float) 
    y = np.array(y).astype(float) 

    summary_df = pd.DataFrame() 

    roc_auc_list = [] 
    acc_df = pd.DataFrame() 
    spf_df = pd.DataFrame() 
    sns_df = pd.DataFrame() 
    prc_df = pd.DataFrame() 
    ## 0.18.1 ##
    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)
    for train_index, test_index in skf.split(X,y): 
        X_train, X_test = X[train_index], X[test_index] 
        y_train, y_test = y[train_index], y[test_index] 
        clf.fit(X_train, y_train) 
        clf_name = str(clf)[:str(clf).index('(')]

        if clf_name == 'SVC':
            scores = clf.decision_function(X_test)
            thresholds = np.linspace(min(scores),max(scores),11)[1:-1]
            thresholds = np.append(thresholds, 0)
        else:
            scores = clf.predict_proba(X_test)[:,1] 

        roc_auc_list.append(metrics.roc_auc_score(y_test, scores))

        temp_acc_list = []
        temp_spf_list = []
        temp_sns_list = []
        temp_prc_list = [] 

        for th in thresholds: 
            y_pred = [1 if sc >= th else 0 for sc in scores]
            cm = metrics.confusion_matrix(y_test,y_pred)
            temp_acc_list.append(metrics.accuracy_score(y_test,y_pred))
            temp_spf_list.append(float(cm[0][0])/(cm[0][0]+cm[0][1]))
            temp_sns_list.append(metrics.recall_score(y_test,y_pred))
            temp_prc_list.append(metrics.precision_score(y_test,y_pred))

        fold_count = 1
        this_fold = 'fold_'+str(fold_count)
        acc_df[this_fold] = temp_acc_list 
        spf_df[this_fold] = temp_spf_list
        sns_df[this_fold] = temp_sns_list
        prc_df[this_fold] = temp_prc_list
        fold_count += 1

    range_thres = range(0,len(thresholds))
    summary_df['roc_auc_mean'] = np.repeat(np.mean(roc_auc_list),len(thresholds))
    summary_df['roc_auc_std'] = np.repeat(np.std(roc_auc_list),len(thresholds))
    summary_df['threshold'] = thresholds 
    summary_df['acc_mean'] = [np.mean(acc_df.ix[i,:]) for i in range_thres]
    summary_df['acc_std'] = [np.std(acc_df.ix[i,:]) for i in range_thres]
    summary_df['spf_mean'] = [np.mean(spf_df.ix[i,:]) for i in range_thres]
    summary_df['spf_std'] = [np.std(spf_df.ix[i,:]) for i in range_thres]
    summary_df['sns_mean'] = [np.mean(sns_df.ix[i,:]) for i in range_thres]
    summary_df['sns_std'] = [np.std(sns_df.ix[i,:]) for i in range_thres]
    summary_df['prc_mean'] = [np.mean(prc_df.ix[i,:]) for i in range_thres]
    summary_df['prc_std'] = [np.std(prc_df.ix[i,:]) for i in range_thres]
    
    return summary_df
        
def dict_to_dataframe(eval_dct, pkl_dct):
    """
    This function takes the output of models.machine_learner(), which are 
    a dictionary of dataframes for each classifier and a dictionary 
    of pickle file locations for each classifier and concatenates everything 
    into a single dataframe that can be used for analysis. 

    eval_dct - dict, each value is a dataframe of evaluation metrics for 
               each classifier ran 
    pkl_dct - dict, each value is a string of where the pickle file is located 
    """
    df = pd.DataFrame(columns=eval_dct[eval_dct.keys()[0]].columns.values)
    for key in eval_dct.keys():
        eval_dct[key].index = np.repeat(key, eval_dct[key].shape[0])
        df = df.append(eval_dct[key])
    pkl_df = pd.DataFrame({'index': pkl_dct.keys(),
                           'pickle_file': pkl_dct.values()}).set_index('index')
    return df.join(pkl_df)
         

