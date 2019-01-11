import scipy.sparse as sps
import numpy as np

from tqdm import tqdm

from SequenceAware.sars_tutorial_master.recommenders.ISeqRecommender import ISeqRecommender
from SequenceAware.sars_tutorial_master.util.fpmc.FPMC_numba import FPMC

from Base.Recommender_utils import check_matrix

from src.utils.top_n_idx_sparse import top_n_idx_sparse, top_n_idx_sparse_submatrix

class FPMCRecommender(ISeqRecommender):
    """
    Implementation of
    Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010). Factorizing personalized Markov chains for next-basket recommendation.
    Proceedings of the 19th International Conference on World Wide Web - WWW ’10, 811

    Based on the implementation available at https://github.com/khesui/FPMC
    """

    RECOMMENDER_NAME = "FPMCRecommender"

    def __init__(self, URM_train, train_sequential_df, seq_user_arr):
        """
        :param n_factor: (optional) the number of latent factors
        :param learn_rate: (optional) the learning rate
        :param regular: (optional) the L2 regularization coefficient
        :param n_epoch: (optional) the number of training epochs
        :param n_neg: (optional) the number of negative samples used in BPR learning
        """
        super(FPMCRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train, "csr")
        self.train_sequential_df = train_sequential_df
        self.seq_user_arr = seq_user_arr
        # Convert data to sequence
        self.train_data = self.get_sequences_df(self.URM_train, self.train_sequential_df, seq_user_arr)

        self._declare(self.train_data)

    def __str__(self):
        return 'FPMCRecommender(n_epoch={n_epoch}, ' \
               'n_neg={n_neg}, ' \
               'n_factor={n_factor}, ' \
               'learn_rate={learn_rate}, ' \
               'regular={regular})'.format(**self.__dict__)

    def fit(self, n_factor=32, learn_rate=0.01, regular=0.001, n_epoch=15, n_neg=10):
        self.n_epoch = n_epoch
        self.n_neg = n_neg
        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

        train_data_supervised = []

        for i, row in self.train_data.iterrows():
            u = self.user_mapping[row['user_id']]

            seq = []
            if len(row['sequence']) > 1:  # cannot use sequences with length 1 for supervised learning
                for item in row['sequence']:
                    i = self.item_mapping[item]
                    seq.append(i)

                train_data_supervised.append((u, seq[len(seq) - 1], seq[:len(seq) - 1]))

        self.fpmc = FPMC(n_user=len(self.user_mapping), n_item=len(self.item_mapping),
                         n_factor=self.n_factor, learn_rate=self.learn_rate, regular=self.regular)

        self.fpmc.user_set = set(self.user_mapping.values())
        self.fpmc.item_set = set(self.item_mapping.values())
        self.fpmc.init_model()

        self.fpmc.learnSBPR_FPMC(train_data_supervised, n_epoch=self.n_epoch, neg_batch_size=self.n_neg)

    # def compute_item_score(self, user_id_array, k=160):
    #
    #     est_ratings_data = []
    #     est_ratings_users = []
    #     est_ratings_items = []
    #     for user_id in tqdm(user_id_array):
    #         user_profile = self.train_data[self.train_data.user_id == user_id]['sequence'].values
    #         user_profile = user_profile[0]
    #         est_rat_user = self.fpmc.evaluation_recommender(user_id, user_profile) # ([indices], [data])
    #         k_to_take = k if len(est_rat_user[0]) >= k else len(est_rat_user[0])
    #         est_ratings_users.extend([user_id] * k_to_take)
    #         est_ratings_items.extend(est_rat_user[0][:k_to_take])
    #         est_ratings_data.extend(est_rat_user[1][:k_to_take])
    #
    #     est_ratings = sps.coo_matrix((est_ratings_data, (est_ratings_users, est_ratings_items)), shape=(self.URM_train.shape[0], self.URM_train.shape[1]))
    #     return est_ratings.tocsr()

    def compute_item_score(self, user_id_array, k=160):
        est_ratings_data = []
        est_ratings_users = []
        est_ratings_items = []
        for i in tqdm(range(len(user_id_array))):
            user_id = user_id_array[i]
            user_profile = self.train_data[self.train_data.user_id == user_id]['sequence'].values
            if user_profile:
                user_profile = user_profile[0]
                ##
                # context = []
                # for item in user_profile:
                #     context.append(self.item_mapping[item])
                context = list(map(lambda item: self.item_mapping[item], user_profile))
                items, est_rat_data = self.fpmc.evaluation_recommender_repeated(self.user_mapping[user_id], context, iter=i)
                # est_rat_indices = []
                # for i, it in enumerate(items):
                #     est_rat_indices.append(self.reverse_item_mapping[it])
                est_rat_indices = list(map(lambda item: self.reverse_item_mapping[item], items))
                ##
                est_rat_indices = np.array(est_rat_indices)
                est_rat_data = np.array(est_rat_data)
                est_rat_data[np.isnan(est_rat_data)] = -np.inf  # remove nan
                est_rat_sort_i = np.argsort(est_rat_data)[::-1]
                k_to_take = k if len(est_rat_indices) >= k else len(est_rat_indices)
                est_rat_sort_i_to_take = est_rat_sort_i[:k_to_take]
                est_ratings_users.extend([user_id] * k_to_take)
                est_ratings_items.extend(est_rat_indices[est_rat_sort_i_to_take].tolist())
                est_ratings_data.extend(est_rat_data[est_rat_sort_i_to_take].tolist())

        est_ratings = sps.coo_matrix((est_ratings_data, (est_ratings_users, est_ratings_items)), shape=(self.URM_train.shape[0], self.URM_train.shape[1]))
        return est_ratings.tocsr()

    def _get_user_est_rat(self, user_id, user_profile):
        context = []
        for item in user_profile:
            context.append(self.item_mapping[item])

        items, scores = self.fpmc.evaluation_recommender(self.user_mapping[user_id], context)
        recommendations = []

        for i, it in enumerate(items):
            recommendations.append(([self.reverse_item_mapping[it]], scores[i]))
        return recommendations

    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):
        return self.recommend_seq(user_id_array, cutoff = cutoff, remove_seen_flag=remove_seen_flag, remove_top_pop_flag = remove_top_pop_flag, remove_CustomItems_flag = remove_CustomItems_flag)

    def _declare(self, data):
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_item_mapping = {}

        user_counter = 0
        item_counter = 0
        for i, row in data.iterrows():
            if row['user_id'] not in self.user_mapping:
                self.user_mapping[row['user_id']] = user_counter
                user_counter += 1

            for item in row['sequence']:
                if item not in self.item_mapping:
                    self.item_mapping[item] = item_counter
                    self.reverse_item_mapping[item_counter] = item
                    item_counter += 1

    def saveModel(self, folder_path, file_name=None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "URM_train": self.URM_train,
            "train_sequential_df": self.train_sequential_df,
            "seq_user_arr": self.seq_user_arr,
            "n_epoch": self.n_epoch,
            "n_neg": self.n_neg,
            "n_factor": self.n_factor,
            "learn_rate": self.learn_rate,
            "regular": self.regular,
            "fpmc": self.fpmc
        }
        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("{}: Saving complete")
