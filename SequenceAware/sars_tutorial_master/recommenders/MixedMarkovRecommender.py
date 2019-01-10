import logging
import numpy as np
import scipy.sparse as sps
import datetime

from recommenders.ISeqRecommender import ISeqRecommender
from recommenders.MarkovChainRecommender import MarkovChainRecommender

from Base.Recommender_utils import check_matrix

class MixedMarkovChainRecommender(ISeqRecommender):
    """
    Creates markov models with different values of k, and return recommendation by weighting the list of
    recommendation of each model.

    Reference: Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
    Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4
    """

    RECOMMENDER_NAME = "MixedMarkovChainRecommender"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    recommenders = {}

    def __init__(self, URM_train, train_sequential_df, seq_user_arr, min_order=1, max_order=1):
        """
        :param min_order: the minimum order of the Mixed Markov Chain
        :param max_order: the maximum order of the Mixed Markov Chain
        """
        super(MixedMarkovChainRecommender, self).__init__()
        self.min_order = min_order
        self.max_order = max_order
        self.compute_item_score = self.compute_markov_score

        self.URM_train = check_matrix(URM_train, "csr")
        self.train_sequential_df = train_sequential_df
        self.seq_user_arr = seq_user_arr
        # Convert data to sequence
        self.train_data = self.get_sequences_df(self.URM_train, self.train_sequential_df, seq_user_arr)

        # define the models
        orders = range(self.min_order, self.max_order + 1)
        for i in range(len(orders)):
            order = orders[i]
            self.recommenders[order] = MarkovChainRecommender(self.URM_train, self.train_sequential_df, self.seq_user_arr, order)

    def fit(self):
        for order in self.recommenders:
            self.recommenders[order].fit()

    # def recommend(self, user_id_arr=None):
    #     if user_id_arr is None:
    #         user_id_arr = self.seq_user_arr
    #     elif not np.all(np.isin(user_id_arr, self.seq_user_arr, assume_unique=True)):
    #         raise ValueError('user_id not present in initial seq_user_arr.')
    #     recommendations_arr = []
    #     for user_id in user_id_arr:
    #         recommendations = self._recommend_single(user_id)
    #         recommendations_arr.append(recommendations)
    #     return recommendations_arr
    #
    # def _recommend_single(self, user_id):
    #     # user_profile = self.train_data[self.train_data.user_id == user_id]['sequence'].values
    #     # user_profile = user_profile[0]
    #     rec_dict = {}
    #     recommendations = []
    #     sum_of_weights = 0
    #     for order, r in self.recommenders.items():
    #         rec_list = r.recommend([user_id])[0]
    #         print(rec_list)
    #         sum_of_weights += 1 / order
    #         for i in rec_list:
    #             if tuple(i[0]) in rec_dict:
    #                 rec_dict[tuple(i[0])] += 1 / order * i[1]
    #             else:
    #                 rec_dict[tuple(i[0])] = 1 / order * i[1]
    #     for k, v in rec_dict.items():
    #         recommendations.append((list(k), v / sum_of_weights))
    #
    #     return recommendations

    def compute_markov_score(self, user_id_array, k=160):

        if user_id_array is None:
            user_id_array = self.seq_user_arr
        elif not np.all(np.isin(user_id_array, self.seq_user_arr, assume_unique=True)):
            raise ValueError('user_id not present in initial seq_user_arr.')
        est_ratings_data = []
        est_ratings_users = []
        est_ratings_items = []
        for user_id in user_id_array:
            ###
            rec_dict = {}
            recommendations = []
            sum_of_weights = 0
            for order, r in self.recommenders.items():
                rec_list = r.recommend([user_id])[0]
                sum_of_weights += 1 / order
                for i in rec_list:
                    if tuple(i[0]) in rec_dict:
                        rec_dict[tuple(i[0])] += 1 / order * i[1]
                    else:
                        rec_dict[tuple(i[0])] = 1 / order * i[1]
            for k, v in rec_dict.items():
                # recommendations.append((list(k), v / sum_of_weights))
                est_ratings_users.append(user_id)
                est_ratings_items.append(k[0])
                est_ratings_data.append(v / sum_of_weights)

        est_ratings = sps.coo_matrix((est_ratings_data, (est_ratings_users, est_ratings_items)), shape=(len(user_id_array), self.URM_train.shape[1]))
        return est_ratings.tocsr()

    # #override
    # def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):
    #
    #     # If is a scalar transform it in a 1-cell array
    #     if np.isscalar(user_id_array):
    #         user_id_array = np.atleast_1d(user_id_array)
    #         single_user = True
    #     else:
    #         single_user = False
    #
    #
    #     if cutoff is None:
    #         cutoff = self.URM_train.shape[1] - 1
    #
    #     # Compute the scores using the model-specific function
    #     # Vectorize over all users in user_id_array
    #     scores_batch = self.compute_item_score(user_id_array)
    #
    #
    #     # if self.normalize:
    #     #     # normalization will keep the scores in the same range
    #     #     # of value of the ratings in dataset
    #     #     user_profile = self.URM_train[user_id]
    #     #
    #     #     rated = user_profile.copy()
    #     #     rated.data = np.ones_like(rated.data)
    #     #     if self.sparse_weights:
    #     #         den = rated.dot(self.W_sparse).toarray().ravel()
    #     #     else:
    #     #         den = rated.dot(self.W).ravel()
    #     #     den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
    #     #     scores /= den
    #
    #     return top_n_idx_sparse(scores_batch, cutoff, self.URM_train, userListInMatrix=user_id_array, exclude_seen=remove_seen_flag)

    def _set_model_debug(self, recommender, order):
        self.recommenders[order] = recommender

    def saveModel(self, folder_path, file_name=None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "URM_train": self.URM_train,
            "train_sequential_df": self.train_sequential_df,
            "seq_user_arr": self.seq_user_arr,
            "min_order": self.min_order,
            "max_order": self.max_order
        }
        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete")

        for order, r in self.recommenders.items():
            r_file_name = file_name + "_" + r.RECOMMENDER_NAME + "_order_" + str(order)
            r.saveModel(folder_path, file_name=r_file_name)

    def loadModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        super(MixedMarkovChainRecommender, self).loadModel(folder_path, file_name)

        # load submodels
        orders = range(self.min_order, self.max_order + 1)
        for i in range(len(orders)):
            order = orders[i]
            self.recommenders[order] = MarkovChainRecommender(self.URM_train, self.train_sequential_df,
                                                              self.seq_user_arr, order)
            r_file_name = file_name + "_" + self.recommenders[order].RECOMMENDER_NAME + "_order_" + str(order)
            self.recommenders[order].loadModel(folder_path, r_file_name)

