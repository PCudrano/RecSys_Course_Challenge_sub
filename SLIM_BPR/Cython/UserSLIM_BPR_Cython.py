#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""


from Base.Recommender import Recommender
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
from Base.Recommender_utils import similarityMatrixTopK
from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


import subprocess
import os, sys, time

import numpy as np
from Base.Evaluation.Evaluator import SequentialEvaluator







class UserSLIM_BPR_Cython(SLIM_BPR_Cython):

    RECOMMENDER_NAME = "UserSLIM_BPR_Recommender"


    def __init__(self, URM_train, positive_threshold=0.5, URM_validation = None,
                 recompile_cython = False, final_model_sparse_weights = True, train_with_sparse_weights = True,
                 symmetric = True):

        super(SLIM_BPR_Cython, self).__init__()

        self.URM_train = URM_train.copy()
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.positive_threshold = positive_threshold

        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        if self.train_with_sparse_weights:
            self.sparse_weights = True

        self.URM_mask = self.URM_train.T.copy()

        self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
        self.URM_mask.eliminate_zeros()

        assert self.URM_mask.nnz > 0, "SLIM_BPR_Recommender: URM_train_positive is empty, positive threshold is too high"

        self.symmetric = symmetric

        if not self.train_with_sparse_weights:

            n_users = URM_train.shape[0]
            requiredGB = 8 * n_users ** 2 / 1e+06

            if symmetric:
                requiredGB /= 2

            print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} users is {:.2f} MB".format(
                n_users, requiredGB))

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

        self.compute_item_score = self.compute_score_user_based


    def get_S_incremental_and_set_W(self):

        super(UserSLIM_BPR_Cython, self).get_S_incremental_and_set_W()

        # restore W as its transpose
        if self.train_with_sparse_weights:
            self.W_sparse = self.W_sparse.T
        else:
            if self.sparse_weights:
                self.W_sparse = self.W_sparse.T
            else:
                self.W = np.transpose(self.W)
