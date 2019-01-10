import gc
import logging
import numpy as np
import networkx as nx
import pickle

from recommenders.ISeqRecommender import ISeqRecommender
from util.markov.Markov import add_nodes_to_graph, add_edges, apply_skipping, apply_clustering

from SequenceAware.sars_tutorial_master.util.data_utils import create_seq_db_filter_top_k, sequences_to_spfm_format
from Base.Recommender_utils import check_matrix

class MarkovChainRecommender(ISeqRecommender):
    """
    Implementation from Shani, Guy, David Heckerman, and Ronen I. Brafman. "An MDP-based recommender system."
    Journal of Machine Learning Research 6, no. Sep (2005): 1265-1295. Chapter 3-4
    """

    RECOMMENDER_NAME = "MarkovChainRecommender"

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, URM_train, train_sequential_df, seq_user_arr, order):
        """
        :param order: the order of the Markov Chain
        """
        super(MarkovChainRecommender, self).__init__()
        self.order = order

        self.URM_train = check_matrix(URM_train, "csr")
        self.train_sequential_df = train_sequential_df
        self.seq_user_arr = seq_user_arr
        # Convert data to sequence
        self.train_data = self.get_sequences_df(self.URM_train, self.train_sequential_df, seq_user_arr)


    def fit(self):
        sequences = self.train_data['sequence'].values

        logging.info('Building Markov Chain model with k = ' + str(self.order))
        logging.info('Adding nodes')
        self.tree, self.count_dict, self.G = add_nodes_to_graph(sequences, self.order)
        logging.info('Adding edges')
        self.G = add_edges(self.tree, self.count_dict, self.G, self.order)
        logging.info('Applying skipping')
        self.G = apply_skipping(self.G, self.order, sequences)
        logging.info('Applying clustering')
        logging.info('{} states in the graph'.format(len(self.G.nodes())))
        self.G, _, _ = apply_clustering(self.G)
        # drop not useful resources
        self.tree = None
        self.count_dict = None
        gc.collect()

    def recommend(self, user_id_arr=None):
        if user_id_arr is None:
            user_id_arr = self.seq_user_arr
        elif not np.all(np.isin(user_id_arr, self.seq_user_arr, assume_unique=True)):
            raise ValueError('user_id not present in initial seq_user_arr.')
        recommendations_arr = []
        for user_id in user_id_arr:
            # user_profile = self.train_data.loc[self.train_data.user_id == user_id, 'sequence'].values
            user_profile = self.train_data[self.train_data.user_id == user_id]['sequence'].values
            user_profile = user_profile[0]

            # if the user profile is longer than the markov order, chop it keeping recent history
            state = tuple(user_profile[-self.order:])
            # print(user_profile, state)
            # see if graph has that state
            recommendations = []
            if self.G.has_node(state):
                # search for recommendations in the forward star
                rec_dict = {}
                for u, v in self.G.out_edges_iter([state]):
                    lastElement = tuple(v[-1:])
                    if lastElement in rec_dict:
                        rec_dict[lastElement] += self.G[u][v]['count']
                    else:
                        rec_dict[lastElement] = self.G[u][v]['count']
                for k, v in rec_dict.items():
                    recommendations.append((list(k), v))
                # print(recommendations)
                # print("---")
            recommendations_arr.append(recommendations)
        return recommendations_arr

    def _set_graph_debug(self, G):
        self.G = G

    def saveModel(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "URM_train": self.URM_train,
            "train_sequential_df": self.train_sequential_df,
            "seq_user_arr": self.seq_user_arr,
            "order": self.order
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        file_name_G = file_name + "_G"
        print("{}: Saving graph model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name_G))
        nx.write_gpickle(self.G, folder_path + file_name_G)

        print("{}: Saving complete")


    def loadModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        super(MarkovChainRecommender, self).loadModel(folder_path, file_name)

        file_name_G = file_name + "_G"
        print("{}: Loading graph model from file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name_G))
        self.G = nx.read_gpickle(folder_path + file_name_G)

        print("{}: Loading of graph model complete".format(self.RECOMMENDER_NAME))
