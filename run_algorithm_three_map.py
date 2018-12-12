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
import time

import random

import traceback, os
import datetime

import sys
sys.path.append('src/libs/RecSys_Course_2018')


from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from MatrixFactorization.MatrixFactorization_BPR_Theano import MatrixFactorization_BPR_Theano

from Base.NonPersonalizedRecommender import TopPop, Random

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender

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
    targetsListList = targetsList.tolist()
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
    #URM_train, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8, seed=seed, targetsListOrdered=targetsListOrdered, nnz_threshold=10)
    #URM_test_known = None

    # row holdout - validation
    # URM_train_val, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8,
    #                                                        seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                        nnz_threshold=1)
    # URM_train, URM_valid = train_test_holdout(URM_train_val, train_perc=0.7, seed=seed)
    # URM_test_known = None
    URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, targetsListList, train_sequential_df,
                                                            train_perc=0.6,
                                                            seed=seed, targetsListOrdered=targetsListOrdered,
                                                            nnz_threshold=2)
    URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, targetsListList, train_sequential_df,
                                                      train_perc=0.5,
                                                      seed=seed, targetsListOrdered=targetsListOrdered,
                                                      nnz_threshold=1)


    URM_train = URM_train.tocsr()
    URM_validation = URM_valid
    URM_test = URM_test_pred.tocsr()

    # def get_all_tuples_uip_generator(URM):
    #     URM = URM.tocsr()
    #     return ((u, URM.indices[ind]) for u in range(URM.shape[0]) for ind in range(URM.indptr[u], URM.indptr[u + 1]))
    #
    # def get_rand_tuples_uip_generator(URM, repeated_times=1):
    #     def gen(URM):
    #         URM = URM.tocsr()
    #         URM.eliminate_zeros()
    #         ind = random.randint(0, len(URM.data))
    #         u = bis.bisect_right(URM.indptr, ind) - 1
    #         ip = URM.indices[ind]
    #         return (u, ip)
    #
    #     for i in range(repeated_times * URM.nnz):
    #         yield gen(URM)
    #
    # train_data_u_ip = list(get_all_tuples_uip_generator(URM_train))
    # # valid_data_u_ip = list(get_all_tuples_uip_generator(URM_valid))
    # test_data_u_ip = list(get_all_tuples_uip_generator(URM_test))

    recommender_class = SLIM_BPR_Cython # SLIMElasticNetRecommender

    from Base.Evaluation.Evaluator import SequentialEvaluator, CompleteEvaluator, FastEvaluator

    #evaluator = SequentialEvaluator(URM_test, [10], minRatingsPerUser=1, exclude_seen=True)
    #evaluator = CompleteEvaluator(URM_test, [10], minRatingsPerUser=1, exclude_seen=True)
    # evaluator = FastEvaluator(URM_test, [10], minRatingsPerUser=1, exclude_seen=True)

    users_excluded_targets = [u for u in userList_unique if u not in targetsListList]
    users_excluded_targetsOrdered = [u for u in userList_unique if u not in targetsListOrdered]
    users_excluded_targetsCasual = [u for u in userList_unique if u not in targetsListCasual]
    evaluator = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=users_excluded_targets)
    evaluator_sequential = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True,
                                        ignore_users=users_excluded_targetsOrdered)
    evaluator_random = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True,
                                         ignore_users=users_excluded_targetsCasual)

    evaluator_validation_earlystopping = FastEvaluator(URM_validation, cutoff_list=[10], minRatingsPerUser=1,
                                                       exclude_seen=True, ignore_users=users_excluded_targets)


    output_root_path = "result_experiments/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    logFile = open(output_root_path + "result_{algtype:}_{date:%Y%m%d%H%M%S}.txt".format(algtype=recommender_class, date=datetime.datetime.now()), "a")


    try:
        print("Algorithm: {}".format(recommender_class))
        t = time.time()
        recommender = recommender_class(URM_train)

        # l1_coeff = 0.001
        # l2_coeff = 0.001
        # alpha = l1_coeff + l2_coeff
        # l1_ratio = l1_coeff / (l1_coeff + l2_coeff)
        #
        # print("alpha: {}, l1_ratio: {}".format(alpha, l1_ratio))

        # recommender.fit(l1_ratio=l1_ratio, positive_only=False, topK = 200, alpha=alpha, max_iter=100, selection="random", tol=1e-2)
        # recommender.fit(alpha=0.13816102768095773, beta=0.006448891019063874, normalize_similarity=True, topK=100)
        recommender.fit(epochs=1000, logFile=logFile,
            batch_size = 1000, lambda_i = 1e-12, lambda_j = 1e-12, learning_rate = 0.1, topK = 500,
            sgd_mode='adam',
            stop_on_validation = True, lower_validatons_allowed = 2, validation_metric = "MAP",
            evaluator_object = evaluator_validation_earlystopping, validation_every_n = 10)

        results_run, results_run_string = evaluator.evaluateRecommender(recommender)

        elapsed_time = time.time() - t
        print("Elapsed time (recommend_mat_fast) [mm:ss.fff]: {:02d}:{:06.3f}".format(int(elapsed_time / 60),
                                                                                      elapsed_time - 60 * int(
                                                                                          elapsed_time / 60)))

        print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
        logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
        logFile.flush()

        results_run_sequential, results_run_string_sequential = evaluator_sequential.evaluateRecommender(recommender)

        elapsed_time = time.time() - t
        print("Elapsed time (recommend_mat_fast) [mm:ss.fff]: {:02d}:{:06.3f}".format(int(elapsed_time / 60),
                                                                                      elapsed_time - 60 * int(
                                                                                          elapsed_time / 60)))

        print("Algorithm: {}, results SEQUENTIAL: \n{}".format(recommender.__class__, results_run_string_sequential))
        logFile.write("Algorithm: {}, results SEQUENTIAL: \n{}\n".format(recommender.__class__, results_run_string_sequential))
        logFile.flush()

        results_run_random, results_run_string_random = evaluator_random.evaluateRecommender(recommender)

        elapsed_time = time.time() - t
        print("Elapsed time (recommend_mat_fast) [mm:ss.fff]: {:02d}:{:06.3f}".format(int(elapsed_time / 60),
                                                                                      elapsed_time - 60 * int(
                                                                                          elapsed_time / 60)))

        print("Algorithm: {}, results RANDOM: \n{}".format(recommender.__class__, results_run_string_random))
        logFile.write("Algorithm: {}, results RANDOM: \n{}\n".format(recommender.__class__, results_run_string_random))
        logFile.flush()

    except Exception as e:
        traceback.print_exc()
        logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
        logFile.flush()
