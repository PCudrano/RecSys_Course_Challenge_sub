from pymining import seqmining

from SequenceAware.sars_tutorial_master.recommenders.ISeqRecommender import ISeqRecommender
from SequenceAware.sars_tutorial_master.util.SPMFinterface import callSPMF
from SequenceAware.sars_tutorial_master.util.tree.Tree import SmartTree
from SequenceAware.sars_tutorial_master.util.data_utils import create_seq_db_filter_top_k, sequences_to_spfm_format

from Base.Recommender_utils import check_matrix

import numpy as np

class FSMRecommender(ISeqRecommender):
    """Frequent Sequence Mining recommender"""

    def __init__(self, URM_train, train_sequential_df, seq_user_arr, minsup, minconf, max_context=1, min_context=1, spmf_path='spmf/spmf.jar', tmp_path="tmp/"):
        """

        :param minsup: the minimum support threshold. It is interpreted as relative count if in [0-1],
                otherwise as an absolute count. NOTE: Relative count required for training with SPFM (faster).
        :param minconf: the minimum confidence threshold.
        :param max_context: (optional) the maximum number of items in the user profile (starting from the last) that will be used
                for lookup in the database of frequent sequences.
        :param min_context: (optional) the minimum number of items in the user profile (starting from the last) that will be used
                for lookup in the database of frequent sequences.
        :param spmf_path: (optional) path to SPMF jar file. If provided, SPFM library will be used for pattern extraction (algorithm: Prefix Span).
                Otherwise, use pymining, which can be significantly slower depending on the sequence database size.
        :param db_path: (optional) path to the sequence database file
        """

        super(FSMRecommender, self).__init__()
        self.minsup = minsup
        self.minconf = minconf
        self.max_context = max_context
        self.min_context = min_context
        self.recommendation_length = 1
        self.db_path = tmp_path + "sequences.txt"
        self.spmf_path = spmf_path
        self.spmf_algorithm = "PrefixSpan"
        self.output_path = tmp_path + "tmp_output.txt"

        self.URM_train = check_matrix(URM_train, "csr")
        self.train_sequential_df = train_sequential_df
        self.seq_user_arr = seq_user_arr
        # Convert data to sequence
        self.train_data = self.get_sequences_df(self.URM_train, self.train_sequential_df, seq_user_arr)
        # convert the training sequences to SPFM format first
        sequences_to_spfm_format(self.train_data['sequence'].values, tmp_path=self.db_path)


    def __str__(self):
        return 'FreqSeqMiningRecommender: ' \
               'minsup={minsup}, ' \
               'minconf={minconf}, ' \
               'max_context={max_context}, ' \
               'min_context={min_context}, ' \
               'spmf_path={spmf_path}, ' \
               'db_path={db_path}'.format(**self.__dict__)

    def fit(self, spmf=True):
        """
        Fit the model
        :param spmf: (optional) Bool
            If True, run FSM using SPFM over the sequence database stored in `self.db_path`.
            Otherwise, run FSM using `pymining.seqmining` (slower).
        """

        if spmf:
            if self.spmf_path is None or self.db_path is None:
                raise ValueError("You should set db_path and spfm_path before calling fit() without arguments.")

            self.logger.info('Using SPFM (Java) for Frequent Sequence Mining')
            if 0 <= self.minsup <= 1:
                percentage_min_sup = self.minsup * 100
            else:
                raise NameError("SPMF only accepts 0<=minsup<=1")

            # call spmf
            command = ' '.join([self.spmf_algorithm, self.db_path, self.output_path, str(percentage_min_sup) + '%'])
            callSPMF(self.spmf_path, command)

            # parse back output from text file
            self._parse_spfm_output()
        else:
            # use pymining
            self.logger.info('Using pymining.seqmining (python) for Frequent Sequence Mining')
            sequences = self.train_data['sequence'].values
            msup = int(self.minsup * len(sequences)) if 0 <= self.minsup <= 1 else self.minsup
            self.logger.info('Mining frequent sequences (minsup={})'.format(msup))
            self.freq_seqs = seqmining.freq_seq_enum(sequences, msup)

        self.logger.info('{} frequent sequences found'.format(len(self.freq_seqs)))
        self.logger.info('Building the prefix tree')
        self.tree = SmartTree()
        self.root_node = self.tree.set_root()
        for pattern, support in self.freq_seqs:
            if len(pattern) == 1:
                # add node to root
                self.tree.create_node(pattern[0], parent=self.root_node, data={"support": support})
            elif len(pattern) > 1:
                # add entire path starting from root
                self.tree.add_path(self.root_node, pattern, support)
            else:
                raise ValueError('Frequent sequence of length 0')
        self.logger.info('Training completed')

    def recommend(self, user_id_arr=None):
        if user_id_arr is None:
            user_id_arr = self.seq_user_arr
        elif not np.all(np.isin(user_id_arr, self.seq_user_arr, assume_unique=True)):
            raise ValueError('user_id not present in initial seq_user_arr.')
        recommendations_arr = []
        for user_id in user_id_arr:
            # user_profile = self.train_data.loc[self.train_data.user_id == user_id, 'sequence'].values
            user_profile = self.train_data[self.train_data.user_id == user_id]['sequence'].values
            if len(user_profile) > 0:
                user_profile = user_profile[0]
                if user_id == user_id_arr[0]:
                    print(user_profile)
                    print(user_profile[0])
                n = len(user_profile)
                c = min(n, self.max_context)
                match = []
                # iterate over decreasing context lengths until a match with sufficient confidence is found
                while not match and c >= self.min_context:
                    q = user_profile[n - c:n]
                    match = self._find_match(q, self.recommendation_length)
                    c -= 1
                if user_id == user_id_arr[0]:
                    print(match)
                recommendations = self.get_recommendation_list(match)
            else:
                recommendations = []
            recommendations_arr.append(recommendations)
        return recommendations_arr

    def _find_match(self, context, recommendation_length):
        # search context
        lastNode = self.tree.find_path(self.root_node, context)
        if lastNode == -1:
            return []
        else:  # context matched
            context_support = self.tree[lastNode].data['support']
            children = self.tree[lastNode].fpointer

            if not children:
                return []

            # find all path of length recommendation_length from match
            paths = self.tree.find_n_length_paths(lastNode, recommendation_length)
            return self._filter_confidence(context_support, paths)

    def _filter_confidence(self, context_support, path_list):
        goodPaths = []
        for p in path_list:
            confidence = self.tree[p[len(p) - 1]].data['support'] / float(context_support)
            if confidence >= self.minconf:
                goodPaths.append((self.tree.get_nodes_tag(p), confidence))
        return goodPaths

    def _set_tree_debug_only(self, tree):
        self.tree = tree
        self.root_node = tree.get_root()

    def get_freq_seqs(self):
        return self.freq_seqs

    def get_sequence_tree(self):
        return self.tree

    def show_tree(self):
        self.tree.show()

    def get_confidence_list(self, recommendation):
        return list(map(lambda x: x[1], recommendation))

    def _parse_spfm_output(self):
        with open(self.output_path, 'r') as fin:
            self.freq_seqs = []
            for line in fin:
                pieces = line.split('#SUP: ')
                support = pieces[1].strip()
                items = pieces[0].split(' ')
                seq = tuple(x for x in items if x != '' and x != '-1')
                seq_and_support = (seq, int(support))
                self.freq_seqs.append(seq_and_support)
