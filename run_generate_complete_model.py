import numpy as np
import pandas as pd
# import matplotlib.pyplot as pyplot
# import src.utils.build_icm as build_icm
import scipy.sparse as sps
from src.utils.data_splitter import train_test_holdout, train_test_user_holdout, train_test_row_holdout
# from src.utils.evaluation import evaluate_algorithm, evaluate_algorithm_recommendations
# from src.utils.top_n_idx_sparse import top_n_idx_sparse
# from src.utils.Compute_Similarity_Python import Compute_Similarity_Python
# from src.libs.similarity import cosine
# import src.utils.similarity_wrapper as sim
# import time

import random

import traceback, os
import datetime

from src.utils.csv_printer import print_to_csv

import sys
sys.path.append('src/libs/RecSys_Course_2018')


from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.recommenders.HybridLinCombEstRatings import HybridLinCombEstRatings
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from MatrixFactorization.MatrixFactorization_BPR_Theano import MatrixFactorization_BPR_Theano

from Base.NonPersonalizedRecommender import TopPop, Random

#from KNN.UserKNNCFRecommender import UserKNNCFRecommender
#from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
#from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
#from GraphBased.P3alphaRecommender import P3alphaRecommender


from src.recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from src.recommenders.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from src.recommenders.P3AlphaRecommender import P3AlphaRecommender
from src.recommenders.UserCFKNNRecommender import UserCFKNNRecommender
from src.recommenders.UserCBFKNNRecommender import UserCBFKNNRecommender
from src.recommenders.ImplicitALSRecommender import ImplicitALSRecommender
import src.utils.build_icm as build_icm
import time


from FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

# from data.Movielens_10M.Movielens10MReader import Movielens10MReader


if __name__ == '__main__':


    # dataReader = Movielens10MReader()
    #
    # URM_train = dataReader.get_URM_train()
    # URM_validation = dataReader.get_URM_validation()
    # URM_test = dataReader.get_URM_test()

    # #### Global vars


    JUPYTER = True
    sys.path.append('src/data/')

    # #### Load data
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

    # #### Build URM

    URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
    URM_all_csr = URM_all.tocsr()

    # #### Train/test split: ratings and user holdout

    seed = 0
    # ratings holdout
    # URM_train, URM_test_pred = train_test_holdout(URM_all, train_perc=0.8, seed=seed)
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


    URM_train = URM_all.tocsr()

    ICM_all = build_icm.build_icm(tracks_df)

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


    # URM_validation = URM_valid
    # URM_test = URM_test_pred.tocsr()

    recommender_class = SLIM_BPR_Cython


    from Base.Evaluation.Evaluator import SequentialEvaluator

    # evaluator = SequentialEvaluator(URM_test, [10], minRatingsPerUser=1, exclude_seen=True)


    # output_root_path = "result_experiments/"

    # If directory does not exist, create
    # if not os.path.exists(output_root_path):
    #     os.makedirs(output_root_path)


    # logFile = open(output_root_path + "result_{algtype:}_{date:%Y%m%d%H%M%S}.txt".format(algtype=recommender_class, date=datetime.datetime.now()), "a")


    try:
        print("Algorithm: {}".format(recommender_class))
        if recommender_class is SLIM_BPR_Cython:
            recommender = recommender_class(URM_train, symmetric = False)
            recommender.fit(epochs=155, batch_size = 1000, lambda_i = 0.001, lambda_j = 1e-06, learning_rate = 0.001, topK = 300,
                sgd_mode='rmsprop', gamma=0.995, beta_1=0.9, beta_2=0.999,
                stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "MAP",
                evaluator_object = None, validation_every_n = 1000)
            recommender.saveModel("result_experiments/hyb_est_ratings_3/", file_name="SLIM_BPR_300_complete")

        if recommender_class is ImplicitALSRecommender:
            print("Init recsys")
            recommender = recommender_class(URM_train)
            print("Fitting recsys")
            recommender.fit(alpha=15, factors=495, regularization=0.04388, iterations=20)
            print("Hopefully done")




    except Exception as e:
        traceback.print_exc()
        # logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
        # logFile.flush()
