#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

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

import traceback, os
import datetime

import sys
sys.path.append('src/libs/RecSys_Course_2018')

from src.recommenders.HybridLinCombItemSimilarities import HybridLinCombItemSimilarities
from src.recommenders.HybridLinCombEstRatings import HybridLinCombEstRatings
from Base.NonPersonalizedRecommender import TopPop, Random
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from MatrixFactorization.MatrixFactorization_BPR_Theano import MatrixFactorization_BPR_Theano

from ParameterTuning.BayesianSearch import BayesianSearch

import traceback, pickle
from Utils.PoolWithSubprocess import PoolWithSubprocess

from ParameterTuning.AbstractClassSearch import DictionaryKeys



def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, n_cases, output_root_path, metric_to_optimize):

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]


    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases = n_cases,
                                             output_root_path = output_root_path_similarity,
                                             metric=metric_to_optimize)





def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, ICM_train, n_cases, output_root_path, metric_to_optimize):

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    hyperparamethers_range_dictionary["shrink"] = [0, 10, 50, 100, 200, 300, 500, 1000]
    hyperparamethers_range_dictionary["similarity"] = [similarity_type]
    hyperparamethers_range_dictionary["normalize"] = [True, False]

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = range(0, 2)
        hyperparamethers_range_dictionary["tversky_beta"] = range(0, 2)
        hyperparamethers_range_dictionary["normalize"] = [True]

    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = ["none", "BM25", "TF-IDF"]



    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_train, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    output_root_path_similarity = output_root_path + "_" + similarity_type

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             n_cases = n_cases,
                                             output_root_path = output_root_path_similarity,
                                             metric=metric_to_optimize)





def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, n_cases = 30,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             output_root_path ="result_experiments/", parallelizeKNN = False):


    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)





   ##########################################################################################################

    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNCBFRecommender_on_similarity_type,
                                                   parameterSearch = parameterSearch,
                                                   URM_train = URM_train,
                                                   ICM_train = ICM_object,
                                                   n_cases = n_cases,
                                                   output_root_path = this_output_root_path,
                                                   metric_to_optimize = metric_to_optimize)



    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)








def runParameterSearch_Collaborative(recommender_class, URM_train, metric_to_optimize = "PRECISION",
                                     evaluator_validation= None, evaluator_test=None, evaluator_validation_earlystopping = None,
                                     output_root_path ="result_experiments/", parallelizeKNN = False, n_cases = 30):


    from ParameterTuning.AbstractClassSearch import DictionaryKeys


    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    try:


        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME

        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)




        if recommender_class in [TopPop, Random]:

            recommender = recommender_class(URM_train)

            recommender.fit()

            output_file = open(output_root_path_rec_name + "_BayesianSearch.txt", "a")
            result_dict, result_baseline = evaluator_validation.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_validation. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_root_path_rec_name + "_best_result_validation", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            result_dict, result_baseline = evaluator_test.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_test. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_root_path_rec_name + "_best_result_test", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)


            output_file.close()

            return



        ##########################################################################################################

        if recommender_class is UserKNNCFRecommender:

            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                           parameterSearch = parameterSearch,
                                                           URM_train = URM_train,
                                                           n_cases = n_cases,
                                                           output_root_path = output_root_path_rec_name,
                                                           metric_to_optimize = metric_to_optimize)


            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


            return



        ##########################################################################################################

        if recommender_class is ItemKNNCFRecommender:

            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                           parameterSearch = parameterSearch,
                                                           URM_train = URM_train,
                                                           n_cases = n_cases,
                                                           output_root_path = output_root_path_rec_name,
                                                           metric_to_optimize = metric_to_optimize)


            if parallelizeKNN:
                pool = PoolWithSubprocess(processes=int(2), maxtasksperchild=1)
                resultList = pool.map(run_KNNCFRecommender_on_similarity_type_partial, similarity_type_list)

            else:

                for similarity_type in similarity_type_list:
                    run_KNNCFRecommender_on_similarity_type_partial(similarity_type)


            return



        ##########################################################################################################

        # if recommender_class is MultiThreadSLIM_RMSE:
        #
        #     hyperparamethers_range_dictionary = {}
        #     hyperparamethers_range_dictionary["topK"] = [50, 100]
        #     hyperparamethers_range_dictionary["l1_penalty"] = [1e-2, 1e-3, 1e-4]
        #     hyperparamethers_range_dictionary["l2_penalty"] = [1e-2, 1e-3, 1e-4]
        #
        #
        #     recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
        #                              DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
        #                              DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
        #                              DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
        #                              DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
        #
        #


       ##########################################################################################################

        if recommender_class is P3alphaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            hyperparamethers_range_dictionary["beta"] = range(0, 2)
            hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
            #hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
            hyperparamethers_range_dictionary["num_factors"] = [10, 20, 30, 50, 70, 90, 110, 350, 500]
            hyperparamethers_range_dictionary["reg"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1000,
                                                                       "validation_every_n":5, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":20, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam", "sgd"]
            hyperparamethers_range_dictionary["epochs"] = [20, 30, 50, 90, 100, 200, 300, 400]
            hyperparamethers_range_dictionary["num_factors"] = [20, 50, 90, 110, 300, 400, 500]
            # hyperparamethers_range_dictionary["batch_size"] = [1000]
            hyperparamethers_range_dictionary["positive_reg"] = [0.0, 0.1, 1e-3]
            hyperparamethers_range_dictionary["negative_reg"] = [0.0, 0.1, 1e-3]
            hyperparamethers_range_dictionary["user_reg"] = [0.0, 0.1, 1e-3]
            hyperparamethers_range_dictionary["learning_rate"] = [0.5, 0.1, 0.01]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold':0.5},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1, "validation_every_n":1000, "stop_on_validation":False,
                                                                       "evaluator_object":evaluator_validation,
                                                                       "lower_validatons_allowed":1000, "validation_metric":metric_to_optimize},
                                     # DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1000, "epochs":300, "validation_every_n":20, "stop_on_validation":True,
                                     #                                   "evaluator_object":evaluator_validation_earlystopping,
                                     #                                   "lower_validatons_allowed":20, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = list(range(0, 500, 5))

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Theano:

            hyperparamethers_range_dictionary = {}

            hyperparamethers_range_dictionary["learning_rate"] = [0.5, 0.1, 0.01]
            hyperparamethers_range_dictionary["epochs"] = [20, 30, 50, 90, 100, 200, 300, 400]
            hyperparamethers_range_dictionary["num_factors"] = [20, 50, 90, 110, 300, 400, 500]
            hyperparamethers_range_dictionary["positive_reg"] = [0.0, 0.1, 1e-3]
            hyperparamethers_range_dictionary["negative_reg"] = [0.0, 0.1, 1e-3]
            hyperparamethers_range_dictionary["user_reg"] = [0.0, 0.1, 1e-3]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1000},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["epochs"] = [5, 20, 30, 50, 90, 100, 200, 300, 400]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam", "sgd"]
            hyperparamethers_range_dictionary["lambda_i"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["lambda_j"] = [0.0, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [0.5, 1e-1, 1e-2, 1e-4]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':True, 'symmetric':True, 'positive_threshold':0.5},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1, "validation_every_n": 1000,
                                                                       "stop_on_validation": False,
                                                                       "evaluator_object": evaluator_validation,
                                                                       "lower_validatons_allowed": 1000,
                                                                       "validation_metric": metric_to_optimize},
                                     # DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs":300, "batch_size":1000,
                                     #                                   "validation_every_n":30, "stop_on_validation":True,
                                     #                                   "evaluator_object":evaluator_validation_earlystopping,
                                     #                                   "lower_validatons_allowed":30, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["l1_ratio"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            hyperparamethers_range_dictionary["alpha"] = [0.0, 1e-3, 1e-6, 1e-9]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"tol":1e-3, "selection": "random"},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        #########################################################################################################


        if recommender_class is HybridLinCombItemSimilarities:

            print("Starting importing everything")

            from src.recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
            from src.recommenders.ItemCBFKNNRecommender import ItemCBFKNNRecommender
            from src.recommenders.P3AlphaRecommender import P3AlphaRecommender
            import src.utils.build_icm as build_icm
            import time

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

            print("Starting initing the single recsys")

            N_cbf = 3
            N_cf = 24
            N_p3a = 3
            N_hyb = N_cbf + N_cf + N_p3a
            recsys = []
            for i in range(N_cbf):
                recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
            for i in range(N_cf):
                recsys.append(ItemCFKNNRecommender(URM_train))
            for i in range(N_p3a):
                recsys.append(P3AlphaRecommender(URM_train))

            recsys_params = list(zip(np.linspace(10, 120, N_cbf).tolist(), [4] * N_cbf))
            recsys_params2 = list((zip(np.linspace(5, 400, N_cf).tolist(), [12] * N_cf)))
            recsys_params3 = list((zip(np.linspace(90, 110, N_p3a).tolist(), [1] * N_p3a)))

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
                recsys[i + N_cbf].fit(topK=topK, shrink=shrink, type="cosine", alpha=0.3)
            for i in range(N_p3a):
                # print("Training system {:d}...".format(i+N_cbf))
                topK = recsys_params3[i][0]
                shrink = recsys_params3[i][1]
                recsys[i + N_cbf + N_cf].fit(topK=topK, shrink=shrink, alpha=0.31)
            el_t = time.time() - t
            print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

            print("Starting hopefully the tuning")
            hyperparamethers_range_dictionary = {}
            #hyperparamethers_range_dictionary["alphas0"] = range(0, 20)
            for i in range(0, N_hyb):
                text = "alphas" + str(i)
                hyperparamethers_range_dictionary[text] = range(0, 10)

            #hyperparamethers_range_dictionary["alphas1"] = range(0, 20)
            #hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            #hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, recsys],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################


        if recommender_class is HybridLinCombEstRatings:

            print("Starting importing everything")

            from src.recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
            from src.recommenders.ItemCBFKNNRecommender import ItemCBFKNNRecommender
            from src.recommenders.P3AlphaRecommender import P3AlphaRecommender
            from src.recommenders.UserCFKNNRecommender import UserCFKNNRecommender
            from src.recommenders.UserCBFKNNRecommender import UserCBFKNNRecommender
            import src.utils.build_icm as build_icm
            import time

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

            print("Starting initing the single recsys")

            N_cbf = 3
            N_cf = 15
            N_p3a = 2
            N_ucf = 8
            N_ucbf = 4
            N_rp3b = 3
            N_slim = 1
            N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim
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
            recsys.append(SLIM_BPR_Cython(URM_train))

            recsys_params = list(zip(np.linspace(10, 120, N_cbf).tolist(), [4] * N_cbf))
            recsys_params2 = list((zip(np.linspace(5, 600, N_cf).tolist(), [12] * N_cf)))
            recsys_params3 = list((zip(np.linspace(90, 110, N_p3a).tolist(), [1] * N_p3a)))
            recsys_params4 = list((zip(np.linspace(10, 400, N_ucf).tolist(), [2] * N_ucf)))
            recsys_params5 = list((zip(np.linspace(50, 200, N_ucbf).tolist(), [5] * N_ucbf)))
            recsys_params6 = list((zip(np.linspace(80, 120, N_rp3b).tolist(), [0] * N_rp3b)))

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
                recsys[i + N_cbf].fit(topK=topK, shrink=shrink, type="cosine", alpha=0.3)
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
                recsys[i + N_cbf + N_cf + N_p3a + N_ucf + N_ucbf].fit(topK=topK, alpha=0.5927789387679869, beta=0.009260542392306892)

            # load slim bpr
            recsys[-1].loadModel("result_experiments/tuning_20181206151851_good/", "SLIM_BPR_Recommender_best_model")
            print("Load complete of slim bpr")
            el_t = time.time() - t
            print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))


            print("Starting recommending the est_ratings")
            t2 = time.time()
            recsys_est_ratings = []
            for i in range(0, N_hyb):
                if i >= N_cbf + N_cf + N_p3a + N_ucf + N_ucbf:
                    recsys_est_ratings.append(recsys[i].compute_item_score(userList_unique, 160))
                else:
                    recsys_est_ratings.append(recsys[i].estimate_ratings(userList_unique, 160))
            el_t = time.time() - t2
            print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))


            print("Starting hopefully the tuning")
            hyperparamethers_range_dictionary = {}
            #hyperparamethers_range_dictionary["alphas0"] = range(0, 20)
            for i in range(0, N_hyb):
                text = "alphas" + str(i)
                hyperparamethers_range_dictionary[text] = range(0, 20)

            #hyperparamethers_range_dictionary["alphas1"] = range(0, 20)
            #hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            #hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, recsys_est_ratings],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 n_cases = n_cases,
                                                 output_root_path = output_root_path_rec_name,
                                                 metric = metric_to_optimize)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_root_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()

import os, multiprocessing
from functools import partial



#from data.Movielens_10M.Movielens10MReader import Movielens10MReader



def read_data_split_and_search(parallel=False):
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """
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
    URM_train_val, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8,
                                                          seed=seed, targetsListOrdered=targetsListOrdered,
                                                          nnz_threshold=1)
    URM_train, URM_valid = train_test_holdout(URM_train_val, train_perc=0.7, seed=seed)
    URM_test_known = None


    URM_train = URM_train
    URM_validation = URM_valid
    URM_test = URM_test_pred

    # URM_train.data = 15 * URM_train.data


    output_root_path = "result_experiments/tuning_{date:%Y%m%d%H%M%S}/".format(date=datetime.datetime.now())
    # output_root_path = "result_experiments/tuning_main_fix3/"

    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    collaborative_algorithm_list = [
        # Random,
        # TopPop,
        # P3alphaRecommender,
        # RP3betaRecommender,
        # ItemKNNCFRecommender,
        # UserKNNCFRecommender,
        # MatrixFactorization_BPR_Cython,
        # MatrixFactorization_FunkSVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # SLIMElasticNetRecommender,
        #MatrixFactorization_BPR_Theano
        #HybridLinCombItemSimilarities
        HybridLinCombEstRatings
    ]


    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import FastEvaluator

    # FIXME maybe minRatingsPerUser in valid is too much? too few users?
    evaluator_validation_earlystopping = FastEvaluator(URM_validation, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True)
    evaluator_test = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True)


    evaluator_validation = EvaluatorWrapper(evaluator_validation_earlystopping)
    evaluator_test = EvaluatorWrapper(evaluator_test)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train=URM_train,
                                                       metric_to_optimize="MAP",
                                                       evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                       evaluator_validation=evaluator_validation,
                                                       evaluator_test=evaluator_test,
                                                       output_root_path=output_root_path,
                                                       parallelizeKNN=(not parallel),
                                                       n_cases=100
                                                       )

    if parallel:

        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    else:

        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


if __name__ == '__main__':

    read_data_split_and_search(parallel=False)
