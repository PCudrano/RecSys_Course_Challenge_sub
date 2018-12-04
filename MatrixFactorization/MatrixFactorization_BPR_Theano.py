# theano-bpr
#
# Copyright (c) 2014 British Broadcasting Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import theano, numpy
import theano.tensor as T
from theano.printing import pprint
import time, pickle
import sys
from collections import defaultdict

import numpy as np
from Base.Recommender_utils import check_matrix

from Base.Recommender import Recommender

from SLIM_BPR.BPR_sampling import BPR_Sampling

from src.libs.similarity import dot_product

import random
import bisect as bis

class MatrixFactorization_BPR_Theano(Recommender, BPR_Sampling):
    RECOMMENDER_NAME = "MatrixFactorization_BPR_Theano"

    def __init__(self, URM_train):
        """
          Creates a new object for training and testing a Bayesian
          Personalised Ranking (BPR) Matrix Factorisation 
          model, as described by Rendle et al. in:

            http://arxiv.org/abs/1205.2618

          This model tries to predict a ranking of items for each user
          from a viewing history.  
          It's also used in a variety of other use-cases, such
          as matrix completion, link prediction and tag recommendation.

          `rank` is the number of latent features in the matrix
          factorisation model.

          `n_users` is the number of users and `n_items` is the
          number of items.

          The regularisation parameters can be overridden using
          `lambda_u`, `lambda_i` and `lambda_j`. They correspond
          to each three types of updates.

          The learning rate can be overridden using `learning_rate`.

          This object uses the Theano library for training the model, meaning
          it can run on a GPU through CUDA. To make sure your Theano
          install is using the GPU, see:

            http://deeplearning.net/software/theano/tutorial/using_gpu.html

          When running on CPU, we recommend using OpenBLAS.

            http://www.openblas.net/

          Example use (10 latent dimensions, 100 users, 50 items) for
          training:

          >>> from theano_bpr import BPR
          >>> bpr = BPR(10, 100, 50) 
          >>> from numpy.random import randint
          >>> train_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.train(train_data)

          This object also has a method for testing, which will return
          the Area Under Curve for a test set.

          >>> test_data = zip(randint(100, size=1000), randint(50, size=1000))
          >>> bpr.test(test_data)

          (This should give an AUC of around 0.5 as the training and
          testing set are chosen at random)
        """
        super(MatrixFactorization_BPR_Theano, self).__init__()
        self.URM_train = check_matrix(URM_train, format='csr', dtype=np.float32)
        self.train_data_u_ip = None

    def fit(self, num_factors=50, user_reg = 0.0025, positive_reg = 0.0025, negative_reg = 0.00025, lambda_bias = 0.0, learning_rate = 0.05, epochs=30, batch_size=1000):
        self.rank = num_factors
        self.n_users = self.URM_train.shape[0]
        self.n_items = self.URM_train.shape[1]
        self.lambda_u = user_reg
        self.lambda_i = positive_reg
        self.lambda_j = negative_reg
        self.lambda_bias = lambda_bias
        self.learning_rate = learning_rate
        self.train_users = set()
        self.train_items = set()
        self.train_dict = {}
        self._configure_theano()
        self._generate_train_model_function()
        self.train_data_u_ip_gen = self.get_rand_tuples_uip_generator()
        self.batch_size = batch_size
        self.epochs = epochs

        self.initializeFastSampling(positive_threshold=0.5)

        print(len(self.eligibleUsers))

        # run training
        self.train()

    def get_all_tuples_uip_generator(self):
        URM = self.URM_train # .tocsr()
        return ((u, URM.indices[ind]) for u in range(URM.shape[0]) for ind in range(URM.indptr[u], URM.indptr[u + 1]))

    def get_rand_tuples_uip_generator(self):
        def gen(URM):
            URM = URM.tocsr()
            URM.eliminate_zeros()
            ind = random.randint(0, len(URM.data)-1)
            u = bis.bisect_right(URM.indptr, ind) - 1
            ipos = URM.indices[ind]
            foundNeg = False
            while not foundNeg:
                rand_i = random.randint(0, URM.shape[1]-1)
                if rand_i not in URM.indices[URM.indptr[u]:URM.indptr[u+1]]:
                    ineg = rand_i
                    foundNeg = True
            return (u, ipos, ineg)

        URM = self.URM_train  # .tocsr()
        while True:
            yield gen(URM)

    def _configure_theano(self):
        """
          Configures Theano to run in fast mode
          and using 32-bit floats.
        """

        # Compile highly optimized code
        theano.config.mode = 'FAST_RUN'
        # Default float
        theano.config.floatX = 'float32'
        # Enable multicore
        theano.config.openmp = 'true'
        theano.config.exception_verbosity='high'
        #theano.config.optimizer = 'None'
        #theano.config.compute_test_value = 'off'
        #theano.config.cxx = ''
        theano.config.on_unused_input='ignore'

    def _generate_train_model_function(self):
        """
          Generates the train model function in Theano.
          This is a straight port of the objective function
          described in the BPR paper.

          We want to learn a matrix factorisation

            U = W.H^T

          where U is the user-item matrix, W is a user-factor
          matrix and H is an item-factor matrix, so that
          it maximises the difference between
          W[u,:].H[i,:]^T and W[u,:].H[j,:]^T, 
          where `i` is a positive item
          (one the user `u` has watched) and `j` a negative item
          (one the user `u` hasn't watched).
        """
        u = T.lvector('u')
        i = T.lvector('i')
        j = T.lvector('j')

        self.W = theano.shared(numpy.random.random((self.n_users, self.rank)).astype('float32'), name='W')
        self.H = theano.shared(numpy.random.random((self.n_items, self.rank)).astype('float32'), name='H')

        # Bias
        self.B = theano.shared(numpy.zeros(self.n_items).astype('float32'), name='B')

        x_ui = T.dot(self.W[u], self.H[i].T).diagonal()
        x_uj = T.dot(self.W[u], self.H[j].T).diagonal()

        x_uij = self.B[i] - self.B[j] + x_ui - x_uj

        obj = T.sum(T.log(T.nnet.sigmoid(x_uij)) - self.lambda_u * (self.W[u] ** 2).sum(axis=1) -
                    self.lambda_i * (self.H[i] ** 2).sum(axis=1) - self.lambda_j * (self.H[j] ** 2).sum(axis=1) -
                    self.lambda_bias * (self.B[i] ** 2 + self.B[j] ** 2))

        cost = - obj

        g_cost_W = T.grad(cost=cost, wrt=self.W)
        g_cost_H = T.grad(cost=cost, wrt=self.H)
        g_cost_B = T.grad(cost=cost, wrt=self.B)

        updates = [(self.W, self.W - self.learning_rate * g_cost_W),
                   (self.H, self.H - self.learning_rate * g_cost_H),
                   (self.B, self.B - self.learning_rate * g_cost_B)]

        self.train_model = theano.function(inputs=[u, i, j], outputs=cost, updates=updates)

    def train(self):
        """
          Trains the BPR Matrix Factorisation model using Stochastic
          Gradient Descent and minibatches over `train_data`.

          `train_data` is an array of (user_index, item_index) tuples.

          We first create a set of random samples from `train_data` for 
          training, of size `epochs` * size of `train_data`.

          We then iterate through the resulting training samples by
          batches of length `batch_size`, and run one iteration of gradient
          descent for the batch.
        """

        if self.URM_train.nnz < self.batch_size:
            sys.stderr.write("WARNING: Batch size is greater than number of training samples, switching to a batch size of %s\n" % str(len(train_data)))
            self.batch_size = len(self.URM_train.nnz)
        # self.train_dict, self.train_users, self.train_items = self._data_to_dict(train_data)

        # n_sgd_samples = batch_size * epochs # makes sense?
        # n_sgd_samples = (int(self.n_users / self.batch_size) + 1) * self.batch_size * self.epochs
        n_sgd_samples = self.batch_size * self.epochs

        # sgd_users, sgd_pos_items, sgd_neg_items = self._uniform_user_sampling(n_sgd_samples)
        # sgd_arr = []
        # for n in range(n_sgd_samples):
        #     sgd_arr.append(next(self.train_data_u_ip_gen))
        # sgd_users, sgd_pos_items, sgd_neg_items = zip(*sgd_arr)
        # sgd_users, sgd_pos_items, sgd_neg_items = next(self.train_data_u_ip_gen)

        z = 0
        t2 = t1 = t0 = time.time()
        while (z+1)*self.batch_size < n_sgd_samples:
            # get next samples

            # sgd_arr = []
            # for n in range(self.batch_size):
            #     sgd_arr.append(next(self.train_data_u_ip_gen))
            # sgd_users, sgd_pos_items, sgd_neg_items = zip(*sgd_arr)
            sgd_users, sgd_pos_items, sgd_neg_items = self.sampleBatch()


            # train with samples
            # self.train_model(
            #     sgd_users[z*batch_size: (z+1)*batch_size],
            #     sgd_pos_items[z*batch_size: (z+1)*batch_size],
            #     sgd_neg_items[z*batch_size: (z+1)*batch_size]
            # )
            x = self.train_model(sgd_users, sgd_pos_items, sgd_neg_items)
            z += 1
            t2 = time.time()
            sys.stderr.write("Processed %s ( %.2f%% ) in %.4f seconds\n" %(str(z*self.batch_size), 100.0 * float(z*self.batch_size)/n_sgd_samples, t2 - t1))
            sys.stderr.flush()
            t1 = t2
        if n_sgd_samples > 0:
            sys.stderr.write("Total training time %.2f seconds; %e per sample\n" % (t2 - t0, (t2 - t0)/n_sgd_samples))
            sys.stderr.flush()

    def _uniform_user_sampling(self, n_samples):
        """
          Creates `n_samples` random samples from training data for performing Stochastic
          Gradient Descent. We start by uniformly sampling users, 
          and then sample a positive and a negative item for each 
          user sample.
        """
        sys.stderr.write("Generating %s random training samples\n" % str(n_samples))
        sgd_users = numpy.array(list(self.train_users))[numpy.random.randint(len(list(self.train_users)), size=n_samples)]
        sgd_pos_items, sgd_neg_items = [], []
        for sgd_user in sgd_users:
            pos_item = self.train_dict[sgd_user][numpy.random.randint(len(self.train_dict[sgd_user]))]
            sgd_pos_items.append(pos_item)
            neg_item = numpy.random.randint(self.n_items)
            while neg_item in self.train_dict[sgd_user]:
                neg_item = numpy.random.randint(self.n_items)
            sgd_neg_items.append(neg_item)
        return sgd_users, sgd_pos_items, sgd_neg_items

    def compute_item_score(self, user_index):
        """
          Computes item predictions for `user_index`.
          Returns an array of prediction values for each item
          in the dataset.
        """
        w = self.W.get_value()
        h = self.H.get_value()
        b = self.B.get_value()
        user_vector = w[user_index,:]
        return user_vector.dot(h.T) + b

    def prediction(self, user_index, item_index):
        """
          Predicts the preference of a given `user_index`
          for a gven `item_index`.
        """
        return self.predictions(user_index)[item_index]

    def top_predictions(self, user_index, topn=10):
        """
          Returns the item indices of the top predictions
          for `user_index`. The number of predictions to return
          can be set via `topn`.
          This won't return any of the items associated with `user_index`
          in the training set.
        """
        return [ 
            item_index for item_index in numpy.argsort(self.predictions(user_index)) 
            if item_index not in self.train_dict[user_index]
        ][::-1][:topn]

    def test(self, test_data):
        """
          Computes the Area Under Curve (AUC) on `test_data`.

          `test_data` is an array of (user_index, item_index) tuples.

          During this computation we ignore users and items
          that didn't appear in the training data, to allow
          for non-overlapping training and testing sets.
        """
        test_dict, test_users, test_items = self._data_to_dict(test_data)
        auc_values = []
        z = 0
        for user in test_dict.keys():
            if user in self.train_users:
                auc_for_user = 0.0
                n = 0
                predictions = self.predictions(user)
                for pos_item in test_dict[user]:
                    if pos_item in self.train_items:
                        for neg_item in self.train_items:
                            if neg_item not in test_dict[user] and neg_item not in self.train_dict[user]:
                                n += 1
                                if predictions[pos_item] > predictions[neg_item]:
                                    auc_for_user += 1
                if n > 0:
                    auc_for_user /= n
                    auc_values.append(auc_for_user)
                z += 1
                if z % 100 == 0 and len(auc_values) > 0:
                    sys.stderr.write("\rCurrent AUC mean (%s samples): %0.5f" % (str(z), numpy.mean(auc_values)))
                    sys.stderr.flush()
        sys.stderr.write("\n")
        sys.stderr.flush()
        return numpy.mean(auc_values)

    def _data_to_dict(self, data):
        data_dict = defaultdict(list)
        items = set()
        for (user, item) in data:
            data_dict[user].append(item)
            items.add(item)
        return data_dict, set(data_dict.keys()), items

    def saveModel(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {"W": self.W,
                              "H": self.H,
                              "B": self.B}

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


        print("{}: Saving complete")