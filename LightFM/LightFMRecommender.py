#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Base.Evaluation.Evaluator import SequentialEvaluator

import numpy as np
import scipy.sparse as sps
import math
import os, sys, time
import pickle

from src.utils.top_n_idx_sparse import top_n_idx_sparse, top_n_idx_sparse_submatrix
from src.libs.similarity.dot_product import dot_product

from lightfm import LightFM

class LightFMRecommender(Recommender, Incremental_Training_Early_Stopping):
    """ LightFMRecommender"""

    RECOMMENDER_NAME = "LightFMRecommender"

    def __init__(self, URM_train, ICM=None, add_identity_features=True, URM_validation = None, sparse_weights=True):
        super(LightFMRecommender, self).__init__()

        self.ICM = ICM.copy() if ICM is not None else None
        if ICM is not None and add_identity_features:
            self.ICM = sps.hstack([self.ICM, sps.identity(self.ICM.shape[0], format="csr")]) # include per-item features as per Notes in http://lyst.github.io/lightfm/docs/lightfm.html#lightfm

        # CSR is faster during evaluation
        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.sparse_weights = sparse_weights

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        self.compute_item_score = self.compute_score_MF


    def fit(self, epochs=10, loss='warp', num_components=100, learning_schedule='adagrad',
            learning_rate=0.05, rho=0.95, epsilon=1e-06, alpha=0.0, max_sampled=100, random_state=None, k=5,n=10,
            stop_on_validation=False, lower_validatons_allowed=5, validation_metric="MAP",
            evaluator_object=None, validation_every_n=1):

        self.num_components = num_components
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = LightFM(no_components=num_components,
                            loss=loss, # 'logistic', 'bpr', 'warp', 'warp-kos'
                            learning_schedule=learning_schedule, # 'adagrad', 'adadelta'
                            max_sampled=max_sampled,
                            user_alpha=alpha,
                            item_alpha=alpha,
                            random_state=random_state,
                            k=k, # for warp-kos loss
                            n=n, # for warp-kos loss
                            learning_rate=learning_rate, # for adagrad learning schedule
                            rho=rho, # for adadelta learning schedule
                            epsilon=epsilon, # for adadelta learning schedule
        )

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        if evaluator_object is None and stop_on_validation:
            evaluator_object = SequentialEvaluator(self.URM_validation, cutoff=[10])

        self._train_with_early_stopping(epochs, validation_every_n, stop_on_validation,
                                    validation_metric, lower_validatons_allowed, evaluator_object,
                                    algorithm_name = self.RECOMMENDER_NAME)

        self._set_best()

        sys.stdout.flush()


    def _run_epoch(self, num_epoch):

        self.model.fit_partial(interactions=self.URM_train,
                               item_features=self.ICM,
                               user_features=None,
                               epochs=1,
                               num_threads=1,
                               verbose=False,
                               sample_weight=None)


    def _initialize_incremental_model(self):
        pass
        # self.item_factors_incremental = self.model.get_item_representations(self.ICM)
        # self.user_factors_incremental = self.model.get_user_representations()
        # self.item_factors_best = self.item_factors_incremental.copy()
        # self.user_factors_best = self.user_factors_incremental.copy()

    def _update_incremental_model(self):
        # TODO no biases used
        item_biases, item_factors = self.model.get_item_representations(self.ICM)
        self.item_factors_incremental = item_factors
        user_biases, user_factors = self.model.get_user_representations()
        self.user_factors_incremental = user_factors
        # copy to use compute_score_MF during training
        self.item_factors = sps.csr_matrix(self.item_factors_incremental.copy(), shape=(self.URM_train.shape[1], self.num_components))
        self.user_factors = sps.csr_matrix(self.user_factors_incremental.copy(), shape=(self.URM_train.shape[0], self.num_components))

    def _update_best_model(self):
        self.item_factors_best = self.item_factors_incremental.copy()
        self.user_factors_best = self.user_factors_incremental.copy()

    def _set_best(self):
        self.item_factors = sps.csr_matrix(self.item_factors_best.copy(), shape=(self.URM_train.shape[1], self.num_components))
        self.user_factors = sps.csr_matrix(self.user_factors_best.copy(), shape=(self.URM_train.shape[0], self.num_components))

    def compute_score_MF(self, user_id_array, k=160):
        est_ratings = dot_product(self.user_factors[user_id_array], self.item_factors.T, k=k, num_threads=1)
        est_ratings = est_ratings.tocsr()
        return est_ratings


    def saveModel(self, folder_path, file_name=None):
        import pickle
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))
        data_dict = {
            "user_factors": self.user_factors,
            "item_factors": self.item_factors
        }
        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("{}: Saving complete")


    def loadModel(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        print("{}: Loading model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))
        data_dict = pickle.load(open(folder_path + file_name, "rb"))
        self.user_factors = data_dict["user_factors"]
        self.item_factors = data_dict["item_factors"]

        print("{}: Loading complete".format(self.RECOMMENDER_NAME))