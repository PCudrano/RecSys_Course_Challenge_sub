
# coding: utf-8

# ### Load data and build matrices


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import scipy.sparse as sps
from scipy.stats import iqr


import src.utils.build_icm as build_icm
from src.utils.data_splitter import train_test_holdout, train_test_user_holdout, train_test_row_holdout


# In[4]:

import sys
sys.path.append("src/libs/RecSys_Course_2018/SequenceAware/sars_tutorial_master") # go to parent dir
sys.path.append("src/libs/RecSys_Course_2018") # go to parent dir


# In[5]:


from SequenceAware.sars_tutorial_master.util.data_utils import create_seq_db_filter_top_k, sequences_to_spfm_format
from SequenceAware.sars_tutorial_master.util.split import last_session_out_split
from SequenceAware.sars_tutorial_master.util.metrics import precision, recall, mrr
from SequenceAware.sars_tutorial_master.util import evaluation
from SequenceAware.sars_tutorial_master.recommenders.FPMCRecommender import FPMCRecommender


# In[6]:


import datetime


# #### Global vars

# In[7]:


JUPYTER = False


# #### Load data

# In[8]:


if JUPYTER:
    # Jupyter
    tracks_csv_file = "../../../data/tracks.csv"
    interactions_csv_file = "../../../data/train.csv"
    playlist_id_csv_file = "../../../data/target_playlists.csv"
    sequential_csv_file = "../../../data/train_sequential.csv"
else:
    # PyCharm
    tracks_csv_file = "data/tracks.csv"
    interactions_csv_file = "data/train.csv"
    playlist_id_csv_file = "data/target_playlists.csv"
    sequential_csv_file = "data/train_sequential.csv"

tracks_df = pd.read_csv(tracks_csv_file)
interactions_df = pd.read_csv(interactions_csv_file)
playlist_id_df = pd.read_csv(playlist_id_csv_file)
train_sequential_df = pd.read_csv(sequential_csv_file)

userList = interactions_df["playlist_id"]
itemList = interactions_df["track_id"]
ratingList = np.ones(interactions_df.shape[0])
targetsList = playlist_id_df["playlist_id"]
targetsListOrdered = targetsList[:5000].tolist()
targetsListCasual = targetsList[5000:].tolist()

userList_unique = pd.unique(userList)
itemList_unique = tracks_df["track_id"]
numUsers = len(userList_unique)
numItems = len(itemList_unique)
numberInteractions = interactions_df.size


# In[9]:


URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all_csr = URM_all.tocsr()


# In[10]:


itemPopularity = (URM_all>0).sum(axis=0)
itemPopularity = np.array(itemPopularity).squeeze()
itemPopularity_unsorted = itemPopularity
itemPopularity = np.sort(itemPopularity)


# #### Prepare ICM and URM with splits

# In[11]:



# Build ICM
ICM_all = build_icm.build_icm(tracks_df, split_duration_lenght=800, feature_weights={'albums': 1, 'artists': 0.5, 'durations': 0.1})

IDF_ENABLED = True

if IDF_ENABLED:
    num_tot_items = ICM_all.shape[0]
    # let's count how many items have a certain feature
    items_per_feature = (ICM_all > 0).sum(axis=0)
    IDF = np.array(np.log(num_tot_items / items_per_feature))[0]
    ICM_idf = ICM_all.copy()
    # compute the number of non-zeros in each col
    # NOTE: this works only if X is instance of sparse.csc_matrix
    col_nnz = np.diff(sps.csc_matrix(ICM_idf).indptr)
    # then normalize the values in each col
    ICM_idf.data *= np.repeat(IDF, col_nnz)
    ICM_all = ICM_idf  # use IDF features

# #### Build URM

URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all_csr = URM_all.tocsr()

URM_IDF_ENABLED = False

if URM_IDF_ENABLED:
    num_tot_items = URM_all.shape[0]
    # let's count how many items have a certain feature
    items_per_feature = (URM_all > 0).sum(axis=0)
    IDF = np.array(np.log(num_tot_items / items_per_feature))[0]
    URM_idf = URM_all.copy()
    # compute the number of non-zeros in each col
    # NOTE: this works only if X is instance of sparse.csc_matrix
    col_nnz = np.diff(sps.csc_matrix(URM_idf).indptr)
    # then normalize the values in each col
    URM_idf.data *= np.repeat(IDF, col_nnz)
    URM_all = URM_idf  # use IDF features

# #### Train/test split: ratings and user holdout

seed = 0
# ratings holdout
# URM_train, URM_test_pred = train_test_holdout(URM_all, train_perc=0.8, seed=seed)
# URM_valid=URM_test_pred
# URM_test_known = None

# user holdout
# URM_train, URM_test_known, URM_test_pred = train_test_user_holdout(URM_all, user_perc=0.8, train_perc=0.8, seed=seed)

# row holdout
# URM_train, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8, seed=seed, targetsListOrdered=targetsListOrdered, nnz_threshold=10)
# URM_test_known = None

# row holdout - validation
# URM_train_val, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8,
#                                                       seed=seed, targetsListOrdered=targetsListOrdered,
#                                                       nnz_threshold=10)
# URM_train, URM_valid = train_test_holdout(URM_train_val, train_perc=0.7, seed=seed)
# URM_test_known = None
# URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df,
#                                                        train_perc=0.6,
#                                                        seed=seed, targetsListOrdered=targetsListOrdered,
#                                                        nnz_threshold=2)
# URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, userList_unique, train_sequential_df,
#                                                   train_perc=0.5,
#                                                  seed=seed, targetsListOrdered=targetsListOrdered,
#                                                  nnz_threshold=1)

# URM_train = URM_train
#URM_validation = URM_valid
# URM_test = URM_test_pred


# In[12]:


import pickle

with open('dump/dump_URM_train_rowholdout0802th2', 'rb') as dump_file:
    URM_train = pickle.load(dump_file)
with open('dump/dump_URM_test_rowholdout0802th2', 'rb') as dump_file:
    URM_test = pickle.load(dump_file)


# #### 3. Fitting the recommender
# 
# Here we fit the recommedation algorithm over the sessions in the training set.  
# This recommender is based on the following paper:
# 
# _Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010). Factorizing personalized Markov chains for next-basket recommendation. Proceedings of the 19th International Conference on World Wide Web - WWW ’10, 811_
# 
# In short, FPMC factorizes a personalized order-1 transition tensor using Tensor Factorization with pairwise loss function akin to BPR (Bayesian Pairwise Ranking).
# 
# <img src="images/fpmc.png" width="200px" />
# 
# TF allows to impute values for the missing transitions between items for each user. For this reason, FPMC can be used for generating _personalized_ recommendations in session-aware recommenders as well.
# 
# In this notebook, you will be able to change the number of latent factors and a few other learning hyper-parameters and see the impact on the recommendation quality.
# 
# The class `FPMCRecommender` has the following initialization hyper-parameters:
# * `n_factor`: (optional) the number of latent factors
# * `learn_rate`: (optional) the learning rate
# * `regular`: (optional) the L2 regularization coefficient
# * `n_epoch`: (optional) the number of training epochs
# * `n_neg`: (optional) the number of negative samples used in BPR learning
# 

# In[13]:


recommender = FPMCRecommender(URM_train, train_sequential_df, targetsListOrdered)


# In[14]:


recommender.fit(n_factor=16, n_epoch=2, learn_rate=0.01, regular=0.001, n_neg=10) 


# In[ ]:


est_rat = recommender.compute_item_score(targetsListOrdered)
print(est_rat)


# In[ ]:


recommendations = recommender.recommend(targetsListOrdered, cutoff=10)
print(recommendations[0:10])


# #### Saving and loading mechanism

# ##### Saving

# In[16]:


# recommender.saveModel("dump/", "test_save_FPMCRecommender")


# ##### Loading

# In[11]:


# recommender_load = FPMCRecommender(URM_train, train_sequential_df, targetsListOrdered)

