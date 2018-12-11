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

from Base.NonPersonalizedRecommender import TopPop, Random
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from SLIM_BPR.Cython.UserSLIM_BPR_Cython import UserSLIM_BPR_Cython
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



def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, init_points, n_cases,
                                            output_root_path, metric_to_optimize, **kwargs):

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
                                             init_points=init_points,
                                             n_cases = n_cases,
                                             output_root_path = output_root_path_similarity,
                                             metric=metric_to_optimize,
                                             **kwargs)





def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch, URM_train, ICM_train, init_points,
                                             n_cases, output_root_path, metric_to_optimize, **kwargs):

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
                                             init_points=init_points,
                                             output_root_path = output_root_path_similarity,
                                             metric=metric_to_optimize,
                                             **kwargs)





def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name, init_points=5, n_cases = 30,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             output_root_path ="result_experiments/", parallelizeKNN = False, parameterSearch=None,
                             **kwargs):


    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)





   ##########################################################################################################

    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    if parameterSearch is None:
        parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNCBFRecommender_on_similarity_type,
                                                   parameterSearch = parameterSearch,
                                                   URM_train = URM_train,
                                                   ICM_train = ICM_object,
                                                   init_points=init_points,
                                                   n_cases = n_cases,
                                                   output_root_path = this_output_root_path,
                                                   metric_to_optimize = metric_to_optimize,
                                                   **kwargs)



    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)








def runParameterSearch_Collaborative(recommender_class, URM_train, metric_to_optimize = "PRECISION",
                                     evaluator_validation= None, evaluator_test=None, evaluator_validation_earlystopping = None,
                                     output_root_path ="result_experiments/", parallelizeKNN = True, init_points=5, n_cases = 30,
                                     parameterSearch=None, **kwargs):


    from ParameterTuning.AbstractClassSearch import DictionaryKeys


    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


    try:


        output_root_path_rec_name = output_root_path + recommender_class.RECOMMENDER_NAME

        if parameterSearch is None:
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
                                                           init_points=init_points,
                                                           n_cases = n_cases,
                                                           output_root_path = output_root_path_rec_name,
                                                           metric_to_optimize = metric_to_optimize,
                                                           **kwargs)


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
                                                           init_points=init_points,
                                                           n_cases = n_cases,
                                                           output_root_path = output_root_path_rec_name,
                                                           metric_to_optimize = metric_to_optimize,
                                                           **kwargs)


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
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam", "sgd", "rmsprop"]
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

            # hyperparamethers_range_dictionary = {}
            # hyperparamethers_range_dictionary["topK"] = [50, 100, 200, 300, 400, 500, 600, 700, 800]
            # # hyperparamethers_range_dictionary["epochs"] = [5, 20, 30, 50, 90, 100, 200, 300, 400, 600, 1000]
            # hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]  # ["adagrad", "adam", "sgd", "rmsprop"]
            # hyperparamethers_range_dictionary["lambda_i"] = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
            #                                                  1e-9]  # [1e-1, 1e-3, 1e-6, 1e-9]
            # hyperparamethers_range_dictionary["lambda_j"] = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
            #                                                  1e-9]  # [1e-1, 1e-3, 1e-6, 1e-9]
            # hyperparamethers_range_dictionary["learning_rate"] = [1e-1, 1e-2, 1e-3]  # [0.5, 1e-1, 1e-2]
            #
            # recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
            #                          DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': False,
            #                                                                    'symmetric': False,
            #                                                                    'positive_threshold': 0.5},
            #                          DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
            #                          # DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1000, "validation_every_n": 1000,
            #                          #                                   "stop_on_validation": False,
            #                          #                                   "evaluator_object": evaluator_validation,
            #                          #                                   "lower_validatons_allowed": 1000,
            #                          #                                   "validation_metric": metric_to_optimize},
            #                          DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 2000, "batch_size": 1000,
            #                                                            "validation_every_n": 10, # 10,
            #                                                            "stop_on_validation": True,
            #                                                            "evaluator_object": evaluator_validation_earlystopping,
            #                                                            "lower_validatons_allowed": 2, # 3,
            #                                                            "validation_metric": metric_to_optimize},
            #                          DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}
            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [50, 100, 200, 300, 400, 500, 600, 700, 800]
            # hyperparamethers_range_dictionary["epochs"] = [5, 20, 30, 50, 90, 100, 200, 300, 400, 600, 1000]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam", "sgd", "rmsprop"]
            hyperparamethers_range_dictionary["lambda_i"] = [1e-1, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["lambda_j"] = [1e-1, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [0.5, 1e-1, 1e-2]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': False,
                                                                               'symmetric': False,
                                                                               'positive_threshold': 0.5},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     # DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1000, "validation_every_n": 1000,
                                     #                                   "stop_on_validation": False,
                                     #                                   "evaluator_object": evaluator_validation,
                                     #                                   "lower_validatons_allowed": 1000,
                                     #                                   "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 2000, "batch_size": 1000,
                                                                       "validation_every_n": 10,  # 10,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 5,  # 3,
                                                                       "validation_metric": metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        #########################################################################################################

        if recommender_class is UserSLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800]
            hyperparamethers_range_dictionary["epochs"] = [5, 20, 30, 50, 90, 100, 200, 300, 400, 600, 1000]
            hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam", "sgd", "rmsprop"]
            hyperparamethers_range_dictionary["lambda_i"] = [1e-1, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["lambda_j"] = [1e-1, 1e-3, 1e-6, 1e-9]
            hyperparamethers_range_dictionary["learning_rate"] = [0.5, 1e-1, 1e-2, 1e-4]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights': True,
                                                                               'symmetric': False,
                                                                               'positive_threshold': 0.5},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"batch_size": 1000, "validation_every_n": 1000,
                                                                       "stop_on_validation": False,
                                                                       "evaluator_object": evaluator_validation,
                                                                       "lower_validatons_allowed": 1000,
                                                                       "validation_metric": metric_to_optimize},
                                     # DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 2000, "batch_size": 1000,
                                     #                                   "validation_every_n": 10,  # 10,
                                     #                                   "stop_on_validation": True,
                                     #                                   "evaluator_object": evaluator_validation_earlystopping,
                                     #                                   "lower_validatons_allowed": 5,  # 3,
                                     #                                   "validation_metric": metric_to_optimize},
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

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 init_points=init_points,
                                                 n_cases = n_cases,
                                                 output_root_path = output_root_path_rec_name,
                                                 metric = metric_to_optimize,
                                                 **kwargs)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_root_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()


import os, multiprocessing
from functools import partial


#from data.Movielens_10M.Movielens10MReader import Movielens10MReader

if __name__ == '__main__':
    parallel = False
    # """
    # This function provides a simple example on how to tune parameters of a given algorithm
    #
    # The BayesianSearch object will save:
    #     - A .txt file with all the cases explored and the recommendation quality
    #     - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
    #     - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
    #     - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
    #     - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    # """


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
    URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, targetsListList, train_sequential_df,
                                                            train_perc=0.6,
                                                            seed=seed, targetsListOrdered=targetsListOrdered,
                                                            nnz_threshold=2)
    URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, targetsListList, train_sequential_df,
                                                      train_perc=0.5,
                                                      seed=seed, targetsListOrdered=targetsListOrdered,
                                                      nnz_threshold=1)


    URM_train = URM_train
    URM_validation = URM_valid
    URM_test = URM_test_pred

    # URM_train.data = 15 * URM_train.data


    output_root_path = "result_experiments/tuning_{date:%Y%m%d%H%M%S}/".format(date=datetime.datetime.now())

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
        UserSLIM_BPR_Cython
        # SLIMElasticNetRecommender,
        # MatrixFactorization_BPR_Theano
    ]


    from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import SequentialEvaluator, CompleteEvaluator, FastEvaluator

    # FIXME maybe minRatingsPerUser in valid is too much? too few users?

    users_excluded_targets = [u for u in userList_unique if u not in targetsListList]
    evaluator_validation_earlystopping = FastEvaluator(URM_validation, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=users_excluded_targets)
    evaluator_test = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=users_excluded_targets)


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
                                                       init_points=5,
                                                       n_cases=15,
                                                       loggerPath=output_root_path,
                                                       loadLogsPath=None,
                                                       kappa=2,
                                                       #acq='ei',  # acq='ucb', #
                                                       #xi=0.005, # xi=0.0 #,
                                                       )

    # If used with only one class of recommenders, at the end of the optimization parameterSearch can be accessed from
    # Python console to extend the search, etc, by directly accessing parameterSearch.bayesian_optimizer (careful with hyperparams)
    # or by using the wrapper function parameterSearch.callMaximizer(init_points=5, n_iter=30, kappa=2, acq = 'ucb', xi = 0.0)
    parameterSearch = None

    if parallel:

        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    else:

        for recommender_class in collaborative_algorithm_list:
            try:
                output_root_path_recsys = output_root_path + "{}/".format(
                    recommender_class.RECOMMENDER_NAME if recommender_class.RECOMMENDER_NAME is not None else recommender_class)
                parameterSearch = BayesianSearch(recommender_class, evaluator_validation=evaluator_validation,
                                                 evaluator_test=evaluator_test)
                runParameterSearch_Collaborative_partial(recommender_class, parameterSearch=parameterSearch,
                                                         output_root_path=output_root_path_recsys,
                                                         loggerPath=output_root_path_recsys
                                                         )
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()

