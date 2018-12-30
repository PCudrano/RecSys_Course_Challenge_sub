#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17
@author: Maurizio Ferrari Dacrema
"""

import sys
# sys.path.append('src/libs/RecSys_Course_2018')
sys.path.append('/home/stefano/git/recsys/recsys_challenge/src/libs/RecSys_Course_2018')
sys.path.append('/home/stefano/git/recsys/recsys_challenge')

import numpy as np
import pandas as pd
# import matplotlib.pyplot as pyplot
import src.utils.build_icm as build_icm
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

# import sys
# sys.path.append('src/libs/RecSys_Course_2018')


from src.recommenders.HybridLinCombItemSimilarities import HybridLinCombItemSimilarities
from src.recommenders.HybridLinCombEstRatings import HybridLinCombEstRatings

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
from FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from LightFM.LightFMRecommender import LightFMRecommender

from src.recommenders.ImplicitALSRecommender import ImplicitALSRecommender
from src.recommenders.ImplicitBPRRecommender import ImplicitBPRRecommender


from ParameterTuning.AbstractClassSearch_new import DictionaryKeys
from ParameterTuning.BayesianSkoptSearch import BayesianSkoptSearch
from skopt.space import Real, Integer, Categorical


import traceback, pickle
from Utils.PoolWithSubprocess import PoolWithSubprocess


import os, multiprocessing
from functools import partial


def run_KNNCFRecommender_on_similarity_type(similarity_type, parameterSearch,
                                            URM_train,
                                            output_folder_path,
                                            output_file_name_root,
                                            metric_to_optimize,
                                            **kwargs):



    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
    hyperparamethers_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparamethers_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])


    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

    best_parameters = parameterSearch.search(recommenderDictionary,
                                             output_folder_path = output_folder_path,
                                             output_file_name_root = output_file_name_root + "_" + similarity_type,
                                             metric_to_optimize = metric_to_optimize,
                                             **kwargs)





def run_KNNCBFRecommender_on_similarity_type(similarity_type, parameterSearch,
                                            URM_train, ICM_train,
                                            output_folder_path,
                                            output_file_name_root,
                                            metric_to_optimize,
                                            **kwargs):

    hyperparamethers_range_dictionary = {}
    hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
    hyperparamethers_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparamethers_range_dictionary["similarity"] = Categorical([similarity_type])
    hyperparamethers_range_dictionary["normalize"] = Categorical([True, False])

    if similarity_type == "asymmetric":
        hyperparamethers_range_dictionary["asymmetric_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyperparamethers_range_dictionary["tversky_alpha"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["tversky_beta"] = Real(low = 0, high = 2, prior = 'uniform')
        hyperparamethers_range_dictionary["normalize"] = Categorical([True])

    if similarity_type in ["cosine", "asymmetric"]:
        hyperparamethers_range_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])



    recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [ICM_train, URM_train],
                             DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                             DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                             DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                             DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


    best_parameters = parameterSearch.search(recommenderDictionary,
                                             output_folder_path = output_folder_path,
                                             output_file_name_root = output_file_name_root + "_" + similarity_type,
                                             metric_to_optimize = metric_to_optimize,
                                             **kwargs)





def runParameterSearch_Content(recommender_class, URM_train, ICM_object, ICM_name,
                             evaluator_validation= None, evaluator_test=None, metric_to_optimize = "PRECISION",
                             output_root_path ="result_experiments/", parallelizeKNN = False, parameterSearch=None,
                             **kwargs):


    # If directory does not exist, create
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)


   ##########################################################################################################

    this_output_root_path = output_root_path + recommender_class.RECOMMENDER_NAME + "_{}".format(ICM_name)

    if parameterSearch is None:
        parameterSearch = BayesianSkoptSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)


    similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

    run_KNNCBFRecommender_on_similarity_type_partial = partial(run_KNNCBFRecommender_on_similarity_type,
                                                   parameterSearch = parameterSearch,
                                                   URM_train = URM_train,
                                                   ICM_train = ICM_object,
                                                   output_root_path = this_output_root_path,
                                                   metric_to_optimize = metric_to_optimize,
                                                   **kwargs)



    if parallelizeKNN:
        pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(run_KNNCBFRecommender_on_similarity_type_partial, similarity_type_list)

    else:

        for similarity_type in similarity_type_list:
            run_KNNCBFRecommender_on_similarity_type_partial(similarity_type)


def runParameterSearch_Collaborative(recommender_class, URM_train, ICM_all=None, metric_to_optimize = "PRECISION",
                                     evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                                     output_folder_path ="result_experiments/", parallelizeKNN = True,
                                     parameterSearch=None, **kwargs):


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    try:

        output_file_name_root = recommender_class.RECOMMENDER_NAME

        if parameterSearch is None:
            parameterSearch = BayesianSkoptSearch(recommender_class, evaluator_validation=evaluator_validation, evaluator_test=evaluator_test)




        if recommender_class in [TopPop, Random]:

            recommender = recommender_class(URM_train)

            recommender.fit()

            output_file = open(output_folder_path + output_file_name_root + "_BayesianSearch.txt", "a")
            result_dict, result_baseline = evaluator_validation.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_validation. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_folder_path + output_file_name_root + "_best_result_validation", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            result_dict, result_baseline = evaluator_test.evaluateRecommender(recommender)
            output_file.write("ParameterSearch: Best result evaluated on URM_test. Results: {}".format(result_baseline))

            pickle.dump(result_dict.copy(),
                        open(output_folder_path + output_file_name_root + "_best_result_test", "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)


            output_file.close()

            return



        ##########################################################################################################

        if recommender_class is UserKNNCFRecommender:

            similarity_type_list = ['cosine', 'jaccard', "asymmetric", "dice", "tversky"]

            run_KNNCFRecommender_on_similarity_type_partial = partial(run_KNNCFRecommender_on_similarity_type,
                                                           parameterSearch = parameterSearch,
                                                           URM_train = URM_train,
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
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
                                                           output_folder_path = output_folder_path,
                                                           output_file_name_root = output_file_name_root,
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

        if recommender_class is P3alphaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is RP3betaRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800)
            hyperparamethers_range_dictionary["alpha"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["beta"] = Real(low = 0, high = 2, prior = 'uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True, False])

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is MatrixFactorization_FunkSVD_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"]) # Categorical(["adagrad", "adam", "sgd", "rmsprop"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150) # Integer(20,200)
            hyperparamethers_range_dictionary["reg"] = Real(low = 1e-3, high = 1, prior = 'log-uniform') # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 1000, "batch_size": 1000,
                                                                       "validation_every_n":2, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":3, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is MatrixFactorization_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150)
            hyperparamethers_range_dictionary["batch_size"] = Categorical([1])
            hyperparamethers_range_dictionary["positive_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["negative_reg"] = Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'positive_threshold':0},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"validation_every_n":5, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":20, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is MatrixFactorization_AsySVD_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"]) # Categorical(["adagrad", "adam", "sgd", "rmsprop"])
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["num_factors"] = Integer(1, 150) # Integer(20,200)
            hyperparamethers_range_dictionary["reg"] = Real(low = 1e-3, high = 1, prior = 'log-uniform') # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low = 1e-5, high = 1e-2, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 1000, "batch_size": 1000,
                                                                       "validation_every_n":2, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":3, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is PureSVDRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["num_factors"] = Integer(1,1000) # Integer(1, 250) # Integer(10, 1000)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        #########################################################################################################

        if recommender_class is SLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800) # Integer(20, 1000)
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"]) # Categorical(["adagrad", "adam", "sgd", "rmsprop"])
            hyperparamethers_range_dictionary["lambda_i"] = Real(low = 1e-3, high = 1, prior = 'log-uniform') # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["lambda_j"] = Real(low = 1e-3, high = 1, prior = 'log-uniform') # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low=1e-3, high=1e-2, prior='log-uniform')  # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':False, 'positive_threshold':0.5},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 1000, "batch_size": 1000,
                                                                       "validation_every_n":2, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":3, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        #########################################################################################################

        if recommender_class is UserSLIM_BPR_Cython:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800) # Integer(20, 1000)
            #hyperparamethers_range_dictionary["epochs"] = Integer(1, 150)
            hyperparamethers_range_dictionary["sgd_mode"] = Categorical(["adagrad", "adam"]) # Categorical(["adagrad", "adam", "sgd", "rmsprop"])
            hyperparamethers_range_dictionary["lambda_i"] = Real(low = 1e-3, high = 1, prior = 'log-uniform') # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["lambda_j"] = Real(low = 1e-3, high = 1, prior = 'log-uniform') # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low=1e-3, high=1e-2, prior='log-uniform')  # Real(low = 1e-12, high = 1e-3, prior = 'log-uniform')

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'train_with_sparse_weights':False, 'symmetric':False, 'positive_threshold':0.5},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"epochs": 1000, "batch_size": 1000,
                                                                       "validation_every_n":2, "stop_on_validation":True,
                                                                       "evaluator_object":evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed":3, "validation_metric":metric_to_optimize},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################

        if recommender_class is SLIMElasticNetRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(5, 800) # Integer(20, 800)
            hyperparamethers_range_dictionary["l1_ratio"] = Real(low = 1e-5, high = 1.0, prior = 'log-uniform')
            # hyperparamethers_range_dictionary["alpha"] = Real(low = 1e-9, high = 1.0, prior = 'log-uniform')# [1, 1e-1, 1e-3, 1e-6, 1e-9]
            # hyperparamethers_range_dictionary["max_iter"] = Integer(100, 2000) # [100, 250, 500, 1000, 2000]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}



        ##########################################################################################################
        from src.recommenders.ImplicitALSRecommender import ImplicitALSRecommender
        if recommender_class is ImplicitALSRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["alpha"] = Real(low = 1, high = 200, prior = 'uniform')
            hyperparamethers_range_dictionary["factors"] = Integer(10,1000)
            hyperparamethers_range_dictionary["regularization"] = Real(low = 1e-3, high = 1.0, prior = 'log-uniform')
            hyperparamethers_range_dictionary["iterations"] = Integer(5,100)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"final_model_sparse_weights": True},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"use_native": True,
                                                                       "use_cg": True,
                                                                       "num_threads": 2,
                                                                       "calculate_training_loss": True,
                                                                       "use_gpu": False},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        if recommender_class is ImplicitBPRRecommender:

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["factors"] = Integer(10, 1000)
            hyperparamethers_range_dictionary["regularization"] = Real(low=1e-3, high=1.0, prior='log-uniform')
            hyperparamethers_range_dictionary["learning_rate"] = Real(low=1e-4, high=1e-1, prior='log-uniform')
            hyperparamethers_range_dictionary["iterations"] = Integer(10, 100)

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"final_model_sparse_weights": True},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"verify_negative_samples": True,
                                                                       "num_threads": 1,
                                                                       "use_gpu": False},
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is CFW_D_Similarity_Linalg:

            # train CF
            cf_rec = ItemKNNCFRecommender(URM_train)
            cf_rec.fit(topK=50, shrink=10, similarity='cosine', normalize=True)

            hyperparamethers_range_dictionary = {}
            hyperparamethers_range_dictionary["topK"] = Integer(10, 1000)
            # hyperparamethers_range_dictionary["damp_coeff"] = Real(low=0.0, high=1.0, prior='uniform')
            # hyperparamethers_range_dictionary["add_zeros_quota"] = Real(low=0.0, high=1.0, prior='uniform')
            hyperparamethers_range_dictionary["normalize_similarity"] = Categorical([True,False])

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, ICM_all, cf_rec.W_sparse],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"show_max_performance": True,
                                                                       #"loss_tolerance": None, # 1e-6
                                                                       #"iteration_limit": None, # 50000
                                                                       },
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is LightFMRecommender:

            hyperparamethers_range_dictionary = {}
            # hyperparamethers_range_dictionary["epochs"] = Integer(10, 1000)
            hyperparamethers_range_dictionary["num_components"] = Integer(10,1000)
            hyperparamethers_range_dictionary["alpha"] = Real(low=1e-16, high=1.0, prior='log-uniform') # reg
            # hyperparamethers_range_dictionary["learning_rate"] = Real(low=5e-4, high=5e-2, prior='log-uniform')  # reg
            hyperparamethers_range_dictionary["learning_schedule"] = Categorical(["adadelta"])#"adagrad"

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train], # ICM_all
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {"add_identity_features": True},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: {"loss": "warp", # 'logistic', 'bpr', 'warp', 'warp-kos'
                                                                       "max_sampled": 100, # 10
                                                                       "learning_rate": 5e-3, # for adagrad learning schedule
                                                                       "rho": 0.99, "epsilon": 1e-08, # for adadelta learning schedule
                                                                       "epochs": 2000,
                                                                       "validation_every_n": 10,
                                                                       "stop_on_validation": True,
                                                                       "evaluator_object": evaluator_validation_earlystopping,
                                                                       "lower_validatons_allowed": 2,
                                                                       "validation_metric": metric_to_optimize
                                                                       },
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}


        ##########################################################################################################

        if recommender_class is HybridLinCombEstRatings:

            print("Starting importing everything")

            from src.recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
            from src.recommenders.ItemCBFKNNRecommender import ItemCBFKNNRecommender
            from src.recommenders.P3AlphaRecommender import P3AlphaRecommender
            from src.recommenders.UserCFKNNRecommender import UserCFKNNRecommender
            from src.recommenders.UserCBFKNNRecommender import UserCBFKNNRecommender
            from src.recommenders.ImplicitALSRecommender import ImplicitALSRecommender
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

            # ICM_all = build_icm.build_icm(tracks_df)
            #
            # IDF_ENABLED = True
            #
            # if IDF_ENABLED:
            #     num_tot_items = ICM_all.shape[0]
            #     # let's count how many items have a certain feature
            #     items_per_feature = (ICM_all > 0).sum(axis=0)
            #     IDF = np.array(np.log(num_tot_items / items_per_feature))[0]
            #     ICM_idf = ICM_all.copy()
            #     # compute the number of non-zeros in each col
            #     # NOTE: this works only if X is instance of sparse.csc_matrix
            #     col_nnz = np.diff(sps.csc_matrix(ICM_idf).indptr)
            #     # then normalize the values in each col
            #     ICM_idf.data *= np.repeat(IDF, col_nnz)
            #     ICM_all = ICM_idf  # use IDF features

            print("Starting initing the single recsys")

            # N_cbf = 6
            # N_cf = 40
            # N_p3a = 3
            # N_ucf = 20
            # N_ucbf = 8
            # N_rp3b = 3
            N_cbf = 2
            N_cf = 4
            N_p3a = 0
            N_ucf = 2
            N_ucbf = 1
            N_rp3b = 1
            N_slim = 1
            N_als = 1
            N_hyb_item_sim = 0
            N_pure_svd = 0
            N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_als + N_hyb_item_sim + N_pure_svd
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
            for i in range(N_als):
                recsys.append(ImplicitALSRecommender(URM_train))
            for i in range(N_pure_svd):
                recsys.append(PureSVDRecommender(URM_train))

            # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
            # recsys_params2 = list((zip(np.linspace(5, 800, N_cf).tolist(), [12] * N_cf)))
            # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
            # recsys_params4 = list((zip(np.linspace(170, 180, N_ucf).tolist(), [2] * N_ucf)))
            # recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
            # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))
            recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
            recsys_params2 = list((zip(np.linspace(5, 200, N_cf).tolist(), [12] * N_cf)))
            recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
            recsys_params4 = list((zip(np.linspace(10, 180, N_ucf).tolist(), [2] * N_ucf)))
            recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
            recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))
            # today
            # N_cbf = 2
            # N_cf = 4
            # N_p3a = 0
            # N_ucf = 2
            # N_ucbf = 1
            # N_rp3b = 1
            # N_slim = 1
            # N_als = 1
            # N_hyb_item_sim = 0
            # N_pure_svd = 0
            # N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_als + N_hyb_item_sim + N_pure_svd
            # recsys = []
            # for i in range(N_cbf):
            #     recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
            # for i in range(N_cf):
            #     recsys.append(ItemCFKNNRecommender(URM_train))
            # for i in range(N_p3a):
            #     recsys.append(P3AlphaRecommender(URM_train))
            # for i in range(N_ucf):
            #     recsys.append(UserCFKNNRecommender(URM_train))
            # for i in range(N_ucbf):
            #     recsys.append(UserCBFKNNRecommender(URM_train, ICM_all))
            # for i in range(N_rp3b):
            #     recsys.append(RP3betaRecommender(URM_train))
            # for i in range(N_slim):
            #     recsys.append(SLIM_BPR_Cython(URM_train))
            # for i in range(N_als):
            #     recsys.append(ImplicitALSRecommender(URM_train))
            # for i in range(N_pure_svd):
            #     recsys.append(PureSVDRecommender(URM_train))
            #
            # # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
            # # recsys_params2 = list((zip(np.linspace(5, 800, N_cf).tolist(), [12] * N_cf)))
            # # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
            # # recsys_params4 = list((zip(np.linspace(170, 180, N_ucf).tolist(), [2] * N_ucf)))
            # # recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
            # # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))
            # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
            # recsys_params2 = list((zip(np.linspace(5, 641, N_cf).tolist(), [12] * N_cf)))
            # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
            # recsys_params4 = list((zip(np.linspace(10, 180, N_ucf).tolist(), [2] * N_ucf)))
            # recsys_params5 = list((zip(np.linspace(10, 180, N_ucbf).tolist(), [5] * N_ucbf)))
            # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))

            # N_cbf = 2
            # N_cf = 6
            # N_p3a = 1
            # N_ucf = 1
            # N_ucbf = 1
            # N_rp3b = 1
            # N_slim = 1
            # N_als = 1
            # N_hyb_item_sim = 0
            # N_pure_svd = 0
            # N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_als + N_hyb_item_sim + N_pure_svd
            # recsys = []
            # for i in range(N_cbf):
            #     recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
            # for i in range(N_cf):
            #     recsys.append(ItemCFKNNRecommender(URM_train))
            # for i in range(N_p3a):
            #     recsys.append(P3AlphaRecommender(URM_train))
            # for i in range(N_ucf):
            #     recsys.append(UserCFKNNRecommender(URM_train))
            # for i in range(N_ucbf):
            #     recsys.append(UserCBFKNNRecommender(URM_train, ICM_all))
            # for i in range(N_rp3b):
            #     recsys.append(RP3betaRecommender(URM_train))
            # for i in range(N_slim):
            #     recsys.append(SLIM_BPR_Cython(URM_train))
            # for i in range(N_als):
            #     recsys.append(ImplicitALSRecommender(URM_train))
            # for i in range(N_pure_svd):
            #     recsys.append(PureSVDRecommender(URM_train))
            #
            # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
            # recsys_params2 = list((zip(np.linspace(5, 800, N_cf).tolist(), [12] * N_cf)))
            # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
            # recsys_params4 = list((zip(np.linspace(170, 180, N_ucf).tolist(), [2] * N_ucf)))
            # recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
            # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))

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
                recsys[i + N_cbf + N_cf + N_p3a + N_ucf + N_ucbf].fit(topK=topK, alpha=0.5927789387679869,
                                                                      beta=0.009260542392306892)

            # load slim bpr
            slims_dir = "result_experiments/hyb_est_ratings_4/"
            # recsys[-3].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_100")
            recsys[-2].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_300")
            print("Load complete of slim bpr")
            el_t = time.time() - t
            print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

            print("Starting fitting als")
            recsys[-1].fit(alpha=15, factors=495, regularization=0.04388, iterations=20)
            print("Ended fitting als")

            # print("Starting fitting PureSVD")
            # recsys[-1].fit(num_factors=165)
            # print("PureSVD fitted")

            # print("Starting recommending svd")
            # svd_est = recsys[-1].compute_score_SVD(userList_unique, 160)
            # print("Ended recommending svd")

            print("Starting recommending the est_ratings")
            t2 = time.time()
            recsys_est_ratings = []
            for i in range(0, N_hyb - 1):
                if i >= N_cbf + N_cf + N_p3a + N_ucf + N_ucbf:
                    recsys_est_ratings.append(recsys[i].compute_item_score(userList_unique, 160))
                else:
                    recsys_est_ratings.append(recsys[i].estimate_ratings(userList_unique, 160))
            el_t = time.time() - t2
            print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))
            print("Recommending als")
            recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))
            # print("Recommending hyb item sim")
            # recsys_est_ratings.append(svd_est)

            print("Starting hopefully the tuning")
            hyperparamethers_range_dictionary = {}
            # hyperparamethers_range_dictionary["alphas0"] = range(0, 20)
            for i in range(0, N_hyb):
                text = "alphas" + str(i)
                #hyperparamethers_range_dictionary[text] = Real(low = 0.0, high = 40.0, prior = 'uniform')
                hyperparamethers_range_dictionary[text] = Real(low=0.0, high=60.0)
            # text = "alphas" + str(N_hyb-1)
            # hyperparamethers_range_dictionary[text] = range(0, 2)

            # hyperparamethers_range_dictionary["alphas1"] = range(0, 20)
            # hyperparamethers_range_dictionary["alpha"] = range(0, 2)
            # hyperparamethers_range_dictionary["normalize_similarity"] = [True, False]

            recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [URM_train, recsys_est_ratings],
                                     DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {},
                                     DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                                     DictionaryKeys.FIT_KEYWORD_ARGS: dict(),
                                     DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

        ##########################################################################################################

        ## Final step, after the hyperparameter range has been defined for each type of algorithm
        best_parameters = parameterSearch.search(recommenderDictionary,
                                                 output_folder_path = output_folder_path,
                                                 output_file_name_root = output_file_name_root,
                                                 metric_to_optimize = metric_to_optimize,
                                                 **kwargs)




    except Exception as e:

        print("On recommender {} Exception {}".format(recommender_class, str(e)))
        traceback.print_exc()

        error_file = open(output_folder_path + "ErrorLog.txt", "a")
        error_file.write("On recommender {} Exception {}\n".format(recommender_class, str(e)))
        error_file.close()




if __name__ == '__main__':
    parallel = False
    # """
    # This function provides a simple example on how to tune parameters of a given algorithm
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

    # Build ICM
    #ICM_all = build_icm.build_icm(tracks_df, split_duration_lenght=800, feature_weights={'albums': 1, 'artists': 0.5, 'durations': 0.1})
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
    usersNonOrdered = [i for i in userList_unique if i not in targetsListOrdered]
    URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, targetsListOrdered, train_sequential_df,
                                                            train_perc=0.6,
                                                            seed=seed, targetsListOrdered=targetsListOrdered,
                                                            nnz_threshold=2)
    URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, targetsListOrdered, train_sequential_df,
                                                      train_perc=0.5,
                                                      seed=seed, targetsListOrdered=targetsListOrdered,
                                                      nnz_threshold=1)
    URM_test_known = None

    URM_train = URM_train
    URM_validation = URM_valid
    URM_test = URM_test_pred

    output_root_path = "result_experiments/tuning_skopt_{date:%Y%m%d%H%M%S}_seq/".format(date=datetime.datetime.now())

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
        # MatrixFactorization_AsySVD_Cython,
        # PureSVDRecommender,
        # SLIM_BPR_Cython,
        # UserSLIM_BPR_Cython
        # SLIMElasticNetRecommender,
        # MatrixFactorization_BPR_Theano,
        # ImplicitALSRecommender,
        # ImplicitBPRRecommender,
        # CFW_D_Similarity_Linalg,
        #LightFMRecommender
        HybridLinCombEstRatings
    ]


    #from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
    from Base.Evaluation.Evaluator import SequentialEvaluator, CompleteEvaluator, FastEvaluator

    # FIXME maybe minRatingsPerUser in valid is too much? too few users?
    # users_excluded_targets = [u for u in userList_unique if u not in targetsListList]
    # evaluator_validation_earlystopping = FastEvaluator(URM_validation, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=users_excluded_targets)
    # evaluator_test = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=users_excluded_targets)
    evaluator_validation = FastEvaluator(URM_validation, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=usersNonOrdered)
    evaluator_test = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True, ignore_users=usersNonOrdered)

    runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       ICM_all = ICM_all,
                                                       metric_to_optimize = "MAP",
                                                       #n_cases = 600,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_root_path,
                                                       optimizer="bayesian", # "forest", "gbrt", "bayesian"
                                                       # params
                                                       n_calls=800, # 70,
                                                       #n_random_starts= 20, #20,
                                                       n_points=10000,
                                                       n_jobs=1,
                                                       #noise='gaussian',  # only bayesian
                                                       noise=1e-10,  # only bayesian
                                                       acq_func='LCB', # 'gp_hedge' only for bayesian, use EI or LCB otherwise
                                                       acq_optimizer='auto', # only bayesian and gbrt
                                                       random_state=None,
                                                       verbose=True,
                                                       n_restarts_optimizer=10,  # only bayesian
                                                       xi=0.01,
                                                       kappa=3, # 1.96,
                                                       x0=None,
                                                       y0=None,
                                                       )

    parameterSearch = None

    if parallel:

        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
        resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    else:

        for recommender_class in collaborative_algorithm_list:
            try:
                output_root_path_recsys = output_root_path + "{}/".format(
                    recommender_class.RECOMMENDER_NAME if recommender_class.RECOMMENDER_NAME is not None else recommender_class)
                parameterSearch = BayesianSkoptSearch(recommender_class, evaluator_validation=evaluator_validation,
                                                 evaluator_test=evaluator_test)
                runParameterSearch_Collaborative_partial(recommender_class, parameterSearch=parameterSearch,
                                                         #output_root_path=output_root_path_recsys,
                                                         #loggerPath=output_root_path_recsys
                                                         )
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()
