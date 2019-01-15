#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""


import numpy as np
import scipy.sparse as sps
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge, SGDRegressor

from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import time, sys



class SLIMSGDLinearRegressionRecommender(SimilarityMatrixRecommender, Recommender):
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    # NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available

    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides

        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    RECOMMENDER_NAME = "SLIMSGDLinearRegressionRecommender"

    def __init__(self, URM_train):

        super(SLIMSGDLinearRegressionRecommender, self).__init__()

        self.URM_train = URM_train


    def fit(self, loss='squared_loss', penalty='elasticnet', l1_ratio=0.15, learning_rate='optimal', alpha = 0.0001,
            eta0=0.01, power_t=0.25, topK = 100, max_iter=1000, tol=1e-3):

        # assert l1_ratio>= 0 and l1_ratio<=1, "SLIMLinearRegression: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)
        # assert l1_ratio * alpha >= 1e-2, "SLIMLinearRegression: l1_ratio*alpha must be >= 1e-2, provided value was {}*{}={}"\
        #     .format(l1_ratio,alpha,l1_ratio*alpha)

        self.topK = topK
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.loss = loss
        self.penalty = penalty
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate

        self.model = SGDRegressor(  loss=self.loss, # ‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
                                    penalty=self.penalty, # regularization term: 'none', ‘l2’, ‘l1’, or ‘elasticnet’ (l2 not sparse)
                                    l1_ratio=self.l1_ratio, # Elastic Net mixing parameter
                                    alpha = self.alpha, # Constant that multiplies the regularization term
                                    fit_intercept=False,
                                    max_iter=self.max_iter, # max epochs
                                    tol=self.tol,
                                    shuffle = True, # Whether or not the training data should be shuffled after each epoch
                                    verbose = 0, # verbosity level
                                    epsilon = 0.1, # only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’
                                    random_state = None,
                                    learning_rate ='optimal', # 'constant', 'optimal', 'invscaling', 'adaptive'
                                    eta0 = eta0, # initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules
                                    power_t = power_t,
                                    early_stopping = True,
                                    validation_fraction = 0.1,
                                    n_iter_no_change = 5, # Number of iterations with no improvement to wait before early stopping and adaptive learning schedule
                                    warm_start = False,
                                    average = False, # bool or int: coeff are averaged after seeing <int> samples
                                    # n_iter=self.max_iter # Deprecated
                                )

        URM_train = check_matrix(self.URM_train, 'csc', dtype=np.float32)

        n_items = URM_train.shape[1]


        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0


        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in range(n_items):

            # get the target column
            y = URM_train[:, currentItem].toarray()

            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0



            # fit one ElasticNet model per column
            self.model.fit(URM_train, y.ravel())

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            self.model.sparsify()
            nonzero_model_coef_index = self.model.coef_.indices
            nonzero_model_coef_value = self.model.coef_.data

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]


            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1


            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup


            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                                  currentItem+1,
                                  100.0* float(currentItem+1)/n_items,
                                  (time.time()-start_time)/60,
                                  float(currentItem)/(time.time()-start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()


        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

