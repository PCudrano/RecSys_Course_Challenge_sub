
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import scipy.sparse as sps
from scipy.stats import iqr

import src.utils.build_icm as build_icm
from src.utils.data_splitter import train_test_holdout, train_test_user_holdout, train_test_row_holdout

import sys
sys.path.append("src/libs/RecSys_Course_2018/SequenceAware/sars_tutorial_master") # go to parent dir
sys.path.append("src/libs/RecSys_Course_2018") # go to parent dir

from SequenceAware.sars_tutorial_master.recommenders.MixedMarkovRecommender import MixedMarkovChainRecommender


# #### Global vars

# In[6]:


JUPYTER = False


# #### Load data

# In[7]:


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


# In[8]:


URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
URM_all_csr = URM_all.tocsr()


# In[9]:


itemPopularity = (URM_all>0).sum(axis=0)
itemPopularity = np.array(itemPopularity).squeeze()
itemPopularity_unsorted = itemPopularity
itemPopularity = np.sort(itemPopularity)


# #### Prepare ICM and URM with splits

# In[10]:

# On all URM

# import pickle
#
# with open('dump/dump_URM_train_rowholdout0802', 'rb') as dump_file:
#     URM_train = pickle.load(dump_file)
# with open('dump/dump_URM_test_rowholdout0802', 'rb') as dump_file:
#     URM_test = pickle.load(dump_file)

# URM_train = URM_all_csr

# usersNonOrdered = [i for i in userList_unique if i not in targetsListOrdered]
URM_train, URM_test = train_test_row_holdout(URM_all, targetsListOrdered, train_sequential_df,
                                                        train_perc=0.8,
                                                        seed=0, targetsListOrdered=targetsListOrdered,
                                                        nnz_threshold=2)
# #### 3. Fitting the recommender

# In[12]:


# You can try with max_order=2 or higher too, but it will take some time to complete though due to slow heristic computations
recommender = MixedMarkovChainRecommender(URM_train, train_sequential_df, targetsListOrdered,
                                          min_order=1, 
                                          max_order=1)

recommender.fit()

est_rat = recommender.compute_markov_score(targetsListOrdered)
print(est_rat)

recommendations = recommender.recommend(targetsListOrdered, cutoff=10) # tree is good, there must be sth wrong here
print(recommendations)

# #### Saving and loading mechanism

# ##### Saving

# In[ ]:


recommender.saveModel("dump/", "dump_urmth2_MixedMarkovChainRecommender_ord_1_1")


# ##### Loading

# In[16]:


recommender_load = MixedMarkovChainRecommender(URM_train, train_sequential_df, targetsListOrdered,
                                          min_order=1, 
                                          max_order=1)


# In[ ]:


recommender_load.loadModel("dump/", "dump_urmth2_MixedMarkovChainRecommender_ord_1_1")


# ##### Test loaded model

# In[ ]:


recommendations = recommender_load.recommend(targetsListOrdered, cutoff=10) # tree is good, there must be sth wrong here
recommendations

