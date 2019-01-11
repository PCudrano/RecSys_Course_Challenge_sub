import logging
import numpy as np
import pandas as pd

from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix

from src.utils.top_n_idx_sparse import top_n_idx_sparse, top_n_idx_sparse_submatrix

class ISeqRecommender(Recommender):
    """Abstract Recommender class"""

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def __init__(self):
        super(ISeqRecommender, self).__init__()

    #override from Recommender.recommend adapted for seq only
    def recommend_seq(self, user_id_array, cutoff = None, remove_seen_flag=True, remove_top_pop_flag = False, remove_CustomItems_flag = False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False


        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self.compute_item_score(user_id_array)

        return top_n_idx_sparse(scores_batch[user_id_array], cutoff, self.URM_train, userListInMatrix=user_id_array, exclude_seen=remove_seen_flag)


    @staticmethod
    def get_recommendation_list(recommendation):
        return list(map(lambda x: x[0], recommendation))

    @staticmethod
    def get_recommendation_confidence_list(recommendation):
        return list(map(lambda x: x[1], recommendation))

    def activate_debug_print(self):
        self.logger.setLevel(logging.DEBUG)

    def deactivate_debug_print(self):
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_sequences_df(URM, train_sequential_df, seq_user_arr):
        sequences_arr = []
        for user_id in seq_user_arr:
            URM_start_pos = URM.indptr[user_id]
            URM_end_pos = URM.indptr[user_id + 1]
            user_nnz = URM_end_pos - URM_start_pos

            original_order_items = np.array(
                train_sequential_df.loc[train_sequential_df['playlist_id'].isin([user_id])]['track_id'])
            original_order_items = np.array(
                [i for i in original_order_items if i in URM.indices[URM_start_pos:URM_end_pos]])  # filter items in urm
            # original_order_items_sorted_i = original_order_items.argsort()
            sequences_arr.append({'user_id': user_id, 'sequence': original_order_items.tolist()})
        sequences = pd.DataFrame(sequences_arr)
        sequences = sequences[['user_id', 'sequence']]
        return sequences