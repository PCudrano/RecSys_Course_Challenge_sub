#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""


import numpy as np
import scipy.sparse as sps
from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from sklearn.linear_model import ElasticNet, Ridge, Lasso

from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import time, sys



class SLIMLinearRegressionRecommender(SimilarityMatrixRecommender, Recommender):
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

    RECOMMENDER_NAME = "SLIMLinearRegressionRecommender"

    def __init__(self, URM_train):

        super(SLIMLinearRegressionRecommender, self).__init__()

        self.URM_train = URM_train


    def fit(self, reg_type="l2", solver='auto', topK = 100, alpha=1, positive_only=True, max_iter=None, selection="random", tol=None):

        # assert l1_ratio>= 0 and l1_ratio<=1, "SLIMLinearRegression: l1_ratio must be between 0 and 1, provided value was {}".format(l1_ratio)
        # assert l1_ratio * alpha >= 1e-2, "SLIMLinearRegression: l1_ratio*alpha must be >= 1e-2, provided value was {}*{}={}"\
        #     .format(l1_ratio,alpha,l1_ratio*alpha)
        assert reg_type in ["l1","l2"], "SLIMLinearRegression: reg_type must be either 'l1' or 'l2'"

        self.reg_type = reg_type
        self.topK = topK
        self.alpha = alpha
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol
        self.positive_only = positive_only

        # initialize the ElasticNet model
        if self.reg_type == "l2":
            self.model = Ridge( alpha=self.alpha,
                                fit_intercept=False,
                                normalize=False,
                                copy_X=False,
                                max_iter=self.max_iter, # def: None
                                solver=self.solver, #  {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
                                random_state = None,
                                tol=self.tol if self.tol else 1e-3,
                                ) # def 1e-3
        elif self.reg_type == "l1":
            self.model = Lasso(alpha=self.alpha,
                               positive=self.positive_only,
                               fit_intercept=False,
                               copy_X=False,
                               precompute=True, # For sparse input this option is always True to preserve sparsity.
                               selection=selection,
                               normalize=False,
                               max_iter=self.max_iter if self.max_iter else 1000, # def: 1000
                               tol=self.tol if self.tol else 1e-4,
                               warm_start=False)

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
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            if self.reg_type == "l1":
                nonzero_model_coef_index = self.model.sparse_coef_.indices
                nonzero_model_coef_value = self.model.sparse_coef_.data
            elif self.reg_type == "l2":
                model_coef = self.model.coef_
                model_coef_sparse = sps.csr_matrix(model_coef)
                model_coef_sparse.eliminate_zeros()
                nonzero_model_coef_index = model_coef_sparse.indices
                nonzero_model_coef_value = model_coef_sparse.data

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

