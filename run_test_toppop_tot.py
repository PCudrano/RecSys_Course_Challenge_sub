
import sys
# sys.path.append('src/libs/RecSys_Course_2018')
sys.path.append('/home/stefano/git/recsys/recsys_challenge/src/libs/RecSys_Course_2018')
sys.path.append('/home/stefano/git/recsys/recsys_challenge')

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

# import sys
# sys.path.append('src/libs/RecSys_Course_2018')


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

from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from MatrixFactorization.PureSVD import PureSVDRecommender
from src.recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from src.recommenders.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from src.recommenders.P3AlphaRecommender import P3AlphaRecommender
from src.recommenders.UserCFKNNRecommender import UserCFKNNRecommender
from src.recommenders.UserCBFKNNRecommender import UserCBFKNNRecommender
from src.recommenders.ImplicitALSRecommender import ImplicitALSRecommender
from src.recommenders.TopPopSimple import TopPopSimple
import src.utils.build_icm as build_icm
import time

from src.utils.evaluation import evaluate_algorithm, evaluate_algorithm_targets

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
    targetsListList = targetsList.tolist()

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
    #URM_train, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8, seed=seed, targetsListOrdered=targetsListOrdered, nnz_threshold=1)
    URM_test_known = None

    # row holdout - validation
    # URM_train_val, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8,
    #                                                       seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                       nnz_threshold=10)
    # URM_train, URM_valid = train_test_holdout(URM_train_val, train_perc=0.7, seed=seed)
    # URM_test_known = None

    ##############################################àà
    # URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df,
    #                                                         train_perc=0.6,
    #                                                         seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                         nnz_threshold=2)
    # URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, userList_unique, train_sequential_df,
    #                                                   train_perc=0.5,
    #                                                   seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                   nnz_threshold=1)
    # URM_train = URM_train
    # URM_validation = URM_valid
    # URM_test = URM_test_pred
    #####################################################
    # usersNonOrdered = [i for i in userList_unique if i not in targetsListOrdered]
    # URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, targetsListOrdered, train_sequential_df,
    #                                                         train_perc=0.6,
    #                                                         seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                         nnz_threshold=2)
    # URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, targetsListOrdered, train_sequential_df,
    #                                                   train_perc=0.5,
    #                                                   seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                   nnz_threshold=1)
    ######################################################
    URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df,
                                                            train_perc=0.8,
                                                            seed=seed, targetsListOrdered=targetsListOrdered,
                                                            nnz_threshold=2)
    #
    # URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df,
    #                                                         train_perc=0.8,
    #                                                         seed=seed, targetsListOrdered=[],
    #                                                         nnz_threshold=2)
    URM_test_known = None

    URM_train = URM_train
    URM_validation = URM_valid_test_pred
    # URM_validation = URM_valid
    # URM_test = URM_test_pred

    #URM_train = URM_all.tocsr()

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

    recommender_class = HybridLinCombEstRatings


    from Base.Evaluation.Evaluator import SequentialEvaluator

    # evaluator = SequentialEvaluator(URM_test, [10], minRatingsPerUser=1, exclude_seen=True)


    # output_root_path = "result_experiments/"

    # If directory does not exist, create
    # if not os.path.exists(output_root_path):
    #     os.makedirs(output_root_path)


    # logFile = open(output_root_path + "result_{algtype:}_{date:%Y%m%d%H%M%S}.txt".format(algtype=recommender_class, date=datetime.datetime.now()), "a")


    try:
        print("Algorithm: {}".format(recommender_class))

        print("Starting initing the single recsys")

        N_cbf = 2
        N_cf = 4
        N_p3a = 0
        N_ucf = 2
        N_ucbf = 0
        N_rp3b = 1
        N_slim = 1
        N_elnet = 2
        N_als = 2
        N_top = 1
        N_hyb_item_sim = 0
        N_pure_svd = 0
        N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_elnet + N_als + N_hyb_item_sim + N_pure_svd + N_top
        recsys = []
        for i in range(N_cbf):
            recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
        for i in range(N_cf):
            recsys.append(ItemCFKNNRecommender(URM_train))
        for i in range(N_p3a):
            recsys.append(P3AlphaRecommender(URM_train))
        for i in range(N_ucf):
            recsys.append(UserCFKNNRecommender(URM_train))
        for i in range(N_ucbf):
            recsys.append(UserCBFKNNRecommender(URM_train, ICM_all))
        for i in range(N_rp3b):
            recsys.append(RP3betaRecommender(URM_train))
        for i in range(N_slim):
            recsys.append(SLIM_BPR_Cython(URM_train))
        for i in range(N_elnet):
            recsys.append(SLIMElasticNetRecommender(URM_train))
        for i in range(N_als):
            recsys.append(ImplicitALSRecommender(URM_train))
        for i in range(N_pure_svd):
            recsys.append(PureSVDRecommender(URM_train))
        recsys.append(TopPopSimple(URM_train))

        recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
        recsys_params2 = list((zip(np.linspace(5, 200, N_cf).tolist(), [12] * N_cf)))
        recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
        recsys_params4 = list((zip(np.linspace(10, 180, N_ucf).tolist(), [2] * N_ucf)))
        recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
        recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))

        print("Starting fitting single recsys")
        t = time.time()
        for i in range(N_cbf):
            # print("Training system {:d}...".format(i))
            topK = recsys_params[i][0]
            shrink = recsys_params[i][1]
            recsys[i].fit(topK=topK, shrink=shrink, type="tanimoto")
        for i in range(N_cf):
            # print("Training system {:d}...".format(i+N_cbf))
            topK = recsys_params2[i][0]
            shrink = recsys_params2[i][1]
            recsys[i + N_cbf].fit(topK=topK, shrink=shrink, type="cosine", alpha=0.15)
        for i in range(N_p3a):
            # print("Training system {:d}...".format(i+N_cbf))
            topK = recsys_params3[i][0]
            shrink = recsys_params3[i][1]
            recsys[i + N_cbf + N_cf].fit(topK=topK, shrink=shrink, alpha=0.31)
        for i in range(N_ucf):
            # print("Training system {:d}...".format(i+N_cbf))
            topK = recsys_params4[i][0]
            shrink = recsys_params4[i][1]
            recsys[i + N_cbf + N_cf + N_p3a].fit(topK=topK, shrink=shrink, type="jaccard")
        for i in range(N_ucbf):
            # print("Training system {:d}...".format(i+N_cbf))b
            topK = recsys_params5[i][0]
            shrink = recsys_params5[i][1]
            recsys[i + N_cbf + N_cf + N_p3a + N_ucf].fit(topK=topK, shrink=shrink, type="tanimoto")
        for i in range(N_rp3b):
            # print("Training system {:d}...".format(i+N_cbf))b
            topK = int(recsys_params6[i][0])
            shrink = recsys_params6[i][1]
            recsys[i + N_cbf + N_cf + N_p3a + N_ucf + N_ucbf].fit(topK=topK, alpha=0.5927789387679869,
                                                                  beta=0.009260542392306892)

        # # toppop
        # recsys[-1].fit()
        #
        # # load slim bpr
        # slims_dir = "result_experiments/hyb_est_ratings_6/"
        # # recsys[-3].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_100")
        # recsys[-3].loadModel(slims_dir, "SLIM_BPR_rw_300")
        # print("Load complete of slim bpr")
        # el_t = time.time() - t
        # print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))
        #
        # # print("Starting fitting als")
        # # recsys[-1].fit(alpha=15, factors=495, regularization=0.04388, iterations=20)
        # # print("Ended fitting als")
        #
        # print("Starting recommending the est_ratings")
        # t2 = time.time()
        # recsys_est_ratings = []
        # for i in range(0, N_hyb - 2):
        #     if i >= N_cbf + N_cf + N_p3a + N_ucf + N_ucbf:
        #         recsys_est_ratings.append(recsys[i].compute_item_score(userList_unique, 160))
        #     else:
        #         recsys_est_ratings.append(recsys[i].estimate_ratings(userList_unique, 160))
        # el_t = time.time() - t2
        # print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))
        #
        # print("Recommending als")
        # t2 = time.time()
        # #recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))
        # recsys_est_ratings.append(recsys[-2].loadEstRatings(slims_dir,"ALS_rw_est_rat")[0])
        # el_t = time.time() - t2
        # print("ALS done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))
        #
        # recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))
        #
        # print("Starting hopefully the tuning")

        # toppop
        recsys[-1].fit()

        # load slim bpr
        slims_dir = "result_experiments/hyb_est_ratings_6/"
        # recsys[-3].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_100")
        recsys[-6].loadModel(slims_dir, "SLIM_BPR_rw_300")
        print("Load complete of slim bpr")
        el_t = time.time() - t
        print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        # load slim elnet
        # slims_dir = "result_experiments/tuning_skopt_20190114231643/"
        # recsys[-3].loadModel(slims_dir, "SLIMLinearRegressionRecommender_best_model") # this is seq tho! overfitted
        slims_dir = "result_experiments/elastic_net_2/"
        recsys[-5].loadModel(slims_dir, "SLIMLinearRegressionRecommender_best_model")
        print("Load complete of slim elasticnet")
        el_t = time.time() - t
        print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        # load slim elnet
        # slims_dir = "result_experiments/tuning_skopt_20190114231643/"
        # recsys[-3].loadModel(slims_dir, "SLIMLinearRegressionRecommender_best_model") # this is seq tho! overfitted
        slims_dir = "result_experiments/elastic_net_3/"
        recsys[-4].loadModel(slims_dir, "SLIMElasticNetRecommender_best_model")
        print("Load complete of slim elasticnet")
        el_t = time.time() - t
        print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        # print("Starting fitting als")
        # recsys[-1].fit(alpha=12, factors=600, regularization=0.05388, iterations=80)
        # print("Ended fitting als")

        # print("Starting fitting PureSVD")
        # recsys[-1].fit(num_factors=165)
        # print("PureSVD fitted")

        # print("Starting recommending svd")
        # svd_est = recsys[-1].compute_score_SVD(userList_unique, 160)
        # print("Ended recommending svd")

        print("Starting recommending the est_ratings")
        t2 = time.time()
        recsys_est_ratings = []
        for i in range(0, N_hyb - 3):
            if i >= N_cbf + N_cf + N_p3a + N_ucf + N_ucbf:
                recsys_est_ratings.append(recsys[i].compute_item_score(userList_unique, 160))
            else:
                recsys_est_ratings.append(recsys[i].estimate_ratings(userList_unique, 160))
        el_t = time.time() - t2
        print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        print("Recommending als")
        t2 = time.time()
        # recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))
        slims_dir = "result_experiments/hyb_est_ratings_6/"
        recsys_est_ratings.append(recsys[-3].loadEstRatings(slims_dir, "ALS_rw_est_rat")[0])
        recsys_est_ratings.append(recsys[-2].loadEstRatings(slims_dir, "ALS_rw_est_rat_2")[0])
        el_t = time.time() - t2
        print("ALS done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        # toppop
        recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))

        print("Building the alphas")

        # a = {'alphas0': 12.010528861567822, 'alphas1': 30.389030169991045, 'alphas2': 79.13675426820069,
        #  'alphas3': 0.8825253038265624, 'alphas4': 0.0, 'alphas5': 0.43305412202467247, 'alphas6': 110.0,
        #  'alphas7': 0.3996542429842397, 'alphas8': 44.36734517467593, 'alphas9': 44.7868387839422,
        #  'alphas10': 2.5054716603458216, 'alphas11': 134.08766498406362, 'alphas12': 130.0,
        #  'alphas13': 146.27814168552362}
        a = {'alphas0': 12.010528861567822, 'alphas1': 30.389030169991045, 'alphas2': 79.13675426820069,
         'alphas3': 0.8825253038265624, 'alphas4': 0.0, 'alphas5': 0.43305412202467247, 'alphas6': 110.0,
         'alphas7': 0.3996542429842397, 'alphas8': 44.36734517467593, 'alphas9': 44.7868387839422,
         'alphas10': 2.5054716603458216, 'alphas11': 134.08766498406362, 'alphas12': 130.0,
         'alphas13': 146.27814168552362, "alphas14": -0.3}


        print("Init recsys")
        recommender = recommender_class(URM_train, recsys_est_ratings)
        print("Fitting recsys")
        recommender.fit(**a)
        print("Hopefully done")

        #users_excluded_targets = [u for u in userList_unique if u not in targetsListList]
        result_dict = evaluate_algorithm_targets(URM_validation, recommender, targets=userList_unique, at=10,
                                                  ours=False)
        # result_dict = evaluate_algorithm_targets(URM_test, recommender, targets=userList_unique, at=10, ours=False)
        # result_dict = evaluate_algorithm_targets(URM_validation, recommender, targets=targetsListList, at=10, ours=False)
        # result_dict = evaluate_algorithm_targets(URM_test, recommender, targets=targetsListList, at=10, ours=False)
        #result_dict = evaluate_algorithm_targets(URM_validation, recommender, targets=targetsListOrdered, at=10, ours=False)
        #result_dict = evaluate_algorithm_targets(URM_test, recommender, targets=targetsListOrdered, at=10, ours=False)


    except Exception as e:
        traceback.print_exc()
        # logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
        # logFile.flush()
