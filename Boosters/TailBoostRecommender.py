#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.SimilarityMatrixRecommender import SimilarityMatrixRecommender
import sys
sys.path.append('src/libs/RecSys_Course_2018')

import pickle
from src.utils.top_n_idx_sparse import top_n_idx_sparse
import numpy as np

sys.path.append("../spotify_recsys_challenge_master/") # go to parent dir
from utils.pre_processing import * # norms
from boosts.tail_boost import TailBoost


class TailBoostRecommender(Recommender):
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "TailBoostRecommender"

    def __init__(self, URM_train, test_interactions_df, eurm, similarity):
        super(TailBoostRecommender, self).__init__()
        self.URM_train = check_matrix(URM_train, 'csr')
        self.test_interactions_df = test_interactions_df
        self.eurm = eurm
        self.similarity = similarity

    def fit(self, targets, last_tracks, k, gamma, norm=norm_l2_row):
        self.tb = TailBoost(self.test_interactions_df, self.eurm, self.similarity, norm=norm)
        self.mine_est_ratings = self.tb.boost_eurm(targets, last_tracks, k, gamma)

    def recommend(self, user_id_array, cutoff=10, remove_seen_flag=True, remove_top_pop_flag = False,
                  remove_CustomItems_flag = False, newURM=None):
        URM = self.__get_right_URM(newURM)
        URM = URM[user_id_array]
        #est_ratings = dot_product.dot_product(URM, self.W_sparse, k=120)
        # est_ratings = URM.dot(self.W_sparse)
        #est_ratings = est_ratings.tocsr()
        est_ratings = self.mine_est_ratings[user_id_array]
        recommendations = top_n_idx_sparse(est_ratings, cutoff, URM, exclude_seen=remove_seen_flag)
        return recommendations

    def __filter_seen(self, user_id, scores, newURM=None):
        URM = self.__get_right_URM(newURM)
        start_pos = URM.indptr[user_id]
        end_pos = URM.indptr[user_id + 1]
        user_profile = URM.indices[start_pos:end_pos]
        scores[user_profile] = -np.inf
        return scores

    def __get_right_URM(self, newURM):
        if newURM is None:
            URM = self.URM_train
        else:
            URM = newURM
        return URM

    def compute_score_item_based(self, user_id):
        raise NotImplementedError("Because it's just wrong")

    def compute_item_score(self, user_id):
        raise NotImplementedError("Because it's just wrong2")

    def loadModel(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Loading model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = pickle.load(open(folder_path + file_name, "rb"))

        self.mine_est_ratings = data_dict["mine_est_ratings"]

        print("{}: Loading complete".format(self.RECOMMENDER_NAME))

    def saveModel(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"mine_est_ratings": self.mine_est_ratings}

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))
