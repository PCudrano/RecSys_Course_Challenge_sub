
import sys
# sys.path.append('src/libs/RecSys_Course_2018')
sys.path.append('/home/stefano/git/recsys/recsys_challenge/src/libs/RecSys_Course_2018')
sys.path.append('/home/stefano/git/recsys/recsys_challenge')

import numpy as np
import pandas as pd
# import matplotlib.pyplot as pyplot
# import src.utils.build_icm as build_icm
import scipy.sparse as sps
from src.utils.data_splitter import train_test_holdout, train_test_user_holdout, train_test_row_holdout
# from src.utils.evaluation import evaluate_algorithm, evaluate_algorithm_recommendations
# from src.utils.top_n_idx_sparse import top_n_idx_sparse
# from src.utils.Compute_Similarity_Python import Compute_Similarity_Python
# from src.libs.similarity import cosine
# import src.utils.similarity_wrapper as sim
# import time

import random

import traceback, os
import datetime

from src.utils.csv_printer import print_to_csv

# import sys
# sys.path.append('src/libs/RecSys_Course_2018')


from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.recommenders.HybridLinCombEstRatings import HybridLinCombEstRatings
from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender, MultiThreadSLIM_ElasticNet

from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython, MatrixFactorization_FunkSVD_Cython
from MatrixFactorization.PureSVD import PureSVDRecommender
from MatrixFactorization.MatrixFactorization_BPR_Theano import MatrixFactorization_BPR_Theano

from Base.NonPersonalizedRecommender import TopPop, Random

#from KNN.UserKNNCFRecommender import UserKNNCFRecommender
#from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
#from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender
#from GraphBased.P3alphaRecommender import P3alphaRecommender

from MatrixFactorization.PureSVD import PureSVDRecommender
from src.recommenders.ItemCFKNNRecommender import ItemCFKNNRecommender
from src.recommenders.ItemCBFKNNRecommender import ItemCBFKNNRecommender
from src.recommenders.P3AlphaRecommender import P3AlphaRecommender
from src.recommenders.UserCFKNNRecommender import UserCFKNNRecommender
from src.recommenders.UserCBFKNNRecommender import UserCBFKNNRecommender
from src.recommenders.ImplicitALSRecommender import ImplicitALSRecommender
import src.utils.build_icm as build_icm
import time

from src.utils.evaluation import evaluate_algorithm, evaluate_algorithm_targets


from FW_Similarity.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg

# from data.Movielens_10M.Movielens10MReader import Movielens10MReader


if __name__ == '__main__':


    # dataReader = Movielens10MReader()
    #
    # URM_train = dataReader.get_URM_train()
    # URM_validation = dataReader.get_URM_validation()
    # URM_test = dataReader.get_URM_test()

    # #### Global vars


    JUPYTER = True
    sys.path.append('src/data/')

    # #### Load data
    if JUPYTER:
        # Jupyter
        tracks_csv_file = "../../../data/tracks.csv"
        interactions_csv_file = "../../../data/train.csv"
        playlist_id_csv_file = "../../../data/target_playlists.csv"
        sequential_csv_file = "../../../data/train_sequential.csv"
    else:
        # PyCharm
        tracks_csv_file = "data/tracks.csv"
        interactions_csv_file = "data/train.csv"
        playlist_id_csv_file = "data/target_playlists.csv"
        sequential_csv_file = "data/train_sequential.csv"

    tracks_df = pd.read_csv(tracks_csv_file)
    interactions_df = pd.read_csv(interactions_csv_file)
    playlist_id_df = pd.read_csv(playlist_id_csv_file)
    train_sequential_df = pd.read_csv(sequential_csv_file)
    userList = interactions_df["playlist_id"]
    itemList = interactions_df["track_id"]
    ratingList = np.ones(interactions_df.shape[0])
    targetsList = playlist_id_df["playlist_id"]
    targetsListList = targetsList.tolist()
    targetsListOrdered = targetsList[:5000].tolist()
    targetsListCasual = targetsList[5000:].tolist()
    userList_unique = pd.unique(userList)
    itemList_unique = tracks_df["track_id"]
    numUsers = len(userList_unique)
    numItems = len(itemList_unique)
    numberInteractions = interactions_df.size
    targetsListList = targetsList.tolist()

    # #### Build URM

    URM_all = sps.coo_matrix((ratingList, (userList, itemList)))
    URM_all_csr = URM_all.tocsr()

    # #### Train/test split: ratings and user holdout

    seed = 0
    # ratings holdout
    # URM_train, URM_test_pred = train_test_holdout(URM_all, train_perc=0.8, seed=seed)
    # URM_test_known = None

    # user holdout
    # URM_train, URM_test_known, URM_test_pred = train_test_user_holdout(URM_all, user_perc=0.8, train_perc=0.8, seed=seed)

    # row holdout
    #URM_train, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8, seed=seed, targetsListOrdered=targetsListOrdered, nnz_threshold=1)
    URM_test_known = None

    # row holdout - validation
    # URM_train_val, URM_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df, train_perc=0.8,
    #                                                       seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                       nnz_threshold=10)
    # URM_train, URM_valid = train_test_holdout(URM_train_val, train_perc=0.7, seed=seed)
    # URM_test_known = None

    ##############################################àà
    # URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, userList_unique, train_sequential_df,
    #                                                         train_perc=0.6,
    #                                                         seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                         nnz_threshold=2)
    # URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, userList_unique, train_sequential_df,
    #                                                   train_perc=0.5,
    #                                                   seed=seed, targetsListOrdered=targetsListOrdered,
    #                                                   nnz_threshold=1)
    # URM_train = URM_train
    # URM_validation = URM_valid
    # URM_test = URM_test_pred
    #####################################################
    usersNonOrdered = [i for i in userList_unique if i not in targetsListOrdered]
    URM_train, URM_valid_test_pred = train_test_row_holdout(URM_all, targetsListOrdered, train_sequential_df,
                                                            train_perc=0.6,
                                                            seed=seed, targetsListOrdered=targetsListOrdered,
                                                            nnz_threshold=2)
    URM_valid, URM_test_pred = train_test_row_holdout(URM_valid_test_pred, targetsListOrdered, train_sequential_df,
                                                      train_perc=0.5,
                                                      seed=seed, targetsListOrdered=targetsListOrdered,
                                                      nnz_threshold=1)
    URM_test_known = None

    URM_train = URM_train
    URM_validation = URM_valid
    URM_test = URM_test_pred

    #URM_train = URM_all.tocsr()

    ICM_all = build_icm.build_icm(tracks_df)

    IDF_ENABLED = True

    if IDF_ENABLED:
        num_tot_items = ICM_all.shape[0]
        # let's count how many items have a certain feature
        items_per_feature = (ICM_all > 0).sum(axis=0)
        IDF = np.array(np.log(num_tot_items / items_per_feature))[0]
        ICM_idf = ICM_all.copy()
        # compute the number of non-zeros in each col
        # NOTE: this works only if X is instance of sparse.csc_matrix
        col_nnz = np.diff(sps.csc_matrix(ICM_idf).indptr)
        # then normalize the values in each col
        ICM_idf.data *= np.repeat(IDF, col_nnz)
        ICM_all = ICM_idf  # use IDF features


    # URM_validation = URM_valid
    # URM_test = URM_test_pred.tocsr()

    recommender_class = HybridLinCombEstRatings


    from Base.Evaluation.Evaluator import SequentialEvaluator

    # evaluator = SequentialEvaluator(URM_test, [10], minRatingsPerUser=1, exclude_seen=True)


    # output_root_path = "result_experiments/"

    # If directory does not exist, create
    # if not os.path.exists(output_root_path):
    #     os.makedirs(output_root_path)


    # logFile = open(output_root_path + "result_{algtype:}_{date:%Y%m%d%H%M%S}.txt".format(algtype=recommender_class, date=datetime.datetime.now()), "a")


    try:
        print("Algorithm: {}".format(recommender_class))

        print("Starting initing the single recsys")


        N_cbf = 2
        N_cf = 6
        N_p3a = 1
        N_ucf = 1
        N_ucbf = 1
        N_rp3b = 1
        N_slim = 1
        N_als = 1
        N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_als
        recsys = []
        for i in range(N_cbf):
            recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
        for i in range(N_cf):
            recsys.append(ItemCFKNNRecommender(URM_train))
        for i in range(N_p3a):
            recsys.append(P3AlphaRecommender(URM_train))
        for i in range(N_ucf):
            recsys.append(UserCFKNNRecommender(URM_train))
        for i in range(N_ucbf):
            recsys.append(UserCBFKNNRecommender(URM_train, ICM_all))
        for i in range(N_rp3b):
            recsys.append(RP3betaRecommender(URM_train))
        for i in range(N_slim):
            recsys.append(SLIM_BPR_Cython(URM_train))
        recsys.append(ImplicitALSRecommender(URM_train))

        recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
        recsys_params2 = list((zip(np.linspace(5, 800, N_cf).tolist(), [12] * N_cf)))
        recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
        recsys_params4 = list((zip(np.linspace(170, 180, N_ucf).tolist(), [2] * N_ucf)))
        recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
        recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))

        print("Starting fitting single recsys")
        t = time.time()
        for i in range(N_cbf):
            # print("Training system {:d}...".format(i))
            topK = recsys_params[i][0]
            shrink = recsys_params[i][1]
            recsys[i].fit(topK=topK, shrink=shrink, type="tanimoto")
        for i in range(N_cf):
            # print("Training system {:d}...".format(i+N_cbf))
            topK = recsys_params2[i][0]
            shrink = recsys_params2[i][1]
            recsys[i + N_cbf].fit(topK=topK, shrink=shrink, type="cosine", alpha=0.3)
        for i in range(N_p3a):
            # print("Training system {:d}...".format(i+N_cbf))
            topK = recsys_params3[i][0]
            shrink = recsys_params3[i][1]
            recsys[i + N_cbf + N_cf].fit(topK=topK, shrink=shrink, alpha=0.31)
        for i in range(N_ucf):
            # print("Training system {:d}...".format(i+N_cbf))
            topK = recsys_params4[i][0]
            shrink = recsys_params4[i][1]
            recsys[i + N_cbf + N_cf + N_p3a].fit(topK=topK, shrink=shrink, type="jaccard")
        for i in range(N_ucbf):
            # print("Training system {:d}...".format(i+N_cbf))b
            topK = recsys_params5[i][0]
            shrink = recsys_params5[i][1]
            recsys[i + N_cbf + N_cf + N_p3a + N_ucf].fit(topK=topK, shrink=shrink, type="tanimoto")
        for i in range(N_rp3b):
            # print("Training system {:d}...".format(i+N_cbf))b
            topK = int(recsys_params6[i][0])
            shrink = recsys_params6[i][1]
            recsys[i + N_cbf + N_cf + N_p3a + N_ucf + N_ucbf].fit(topK=topK, alpha=0.5927789387679869,
                                                                  beta=0.009260542392306892)

        # load slim bpr
        slims_dir = "result_experiments/hyb_est_ratings_4/"
        # recsys[-3].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_100")
        recsys[-2].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_300")
        print("Load complete of slim bpr")
        el_t = time.time() - t
        print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        print("Starting fitting als")
        recsys[-1].fit(alpha=15, factors=495, regularization=0.04388, iterations=20)
        print("Ended fitting als")

        print("Starting recommending the est_ratings")
        t2 = time.time()
        recsys_est_ratings = []
        for i in range(0, N_hyb - 1):
            if i >= N_cbf + N_cf + N_p3a + N_ucf + N_ucbf:
                recsys_est_ratings.append(recsys[i].compute_item_score(userList_unique, 160))
            else:
                recsys_est_ratings.append(recsys[i].estimate_ratings(userList_unique, 160))
        el_t = time.time() - t2
        print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))

        print("Recommending als")
        recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))

        print("Starting hopefully the tuning")

        print("Building the alphas")

        # a = {'alphas0': 19.485713591551153, 'alphas1': 4.71319435578388, 'alphas10': 2.2227435851035993, 'alphas11': 1.9716421945640317, 'alphas12': 7.5238841058329875, 'alphas13': 0.4108412788439586, 'alphas14': 4.526224503200407, 'alphas15': 13.576966307999559, 'alphas16': 0.642187854373879, 'alphas17': 5.275515473418448, 'alphas18': 0.34228595410624507, 'alphas19': 0.34191569651452536, 'alphas2': 5.900450505346193, 'alphas20': 6.980737750232469, 'alphas21': 11.130807458091677, 'alphas22': 9.31624459508006, 'alphas23': 4.321048654234952, 'alphas24': 1.1310553103741783, 'alphas25': 9.604200085107115, 'alphas26': 0.5368035241482083, 'alphas27': 3.0960069679180546, 'alphas28': 3.864988337641473, 'alphas29': 16.706157748846, 'alphas3': 16.320530250453828, 'alphas30': 0.9494730681150165, 'alphas31': 0.7654428335341734, 'alphas32': 13.390283773301505, 'alphas33': 19.218172572103484, 'alphas34': 1.4466507699578846, 'alphas35': 19.8594906640407, 'alphas4': 6.6763597060138675, 'alphas5': 10.575281828871317, 'alphas6': 3.7900979297179593, 'alphas7': 4.420167469234189, 'alphas8': 13.929403030269114, 'alphas9': 1.3122328787554016}
        # a = {'alphas0': 12.12143084376925, 'alphas1': 6.329618178683852, 'alphas10': 9.102981891808348, 'alphas11': 0.9001485246078222, 'alphas12': 2.5682180852219316, 'alphas13': 3.2568188540005027, 'alphas14': 11.005268693216813, 'alphas15': 0.9062997275831175, 'alphas16': 13.188009484089187, 'alphas17': 16.618503886464694, 'alphas18': 0.4544583575813199, 'alphas19': 2.715027813148725, 'alphas2': 15.101248341753635, 'alphas20': 8.145981030784329, 'alphas21': 13.829234403692126, 'alphas22': 11.240521880378843, 'alphas23': 4.6187917393267615, 'alphas24': 5.012620582570997, 'alphas25': 6.7324854517393895, 'alphas26': 7.318581954616937, 'alphas27': 14.01494317182526, 'alphas28': 2.545384576903018, 'alphas29': 1.7902310004097965, 'alphas3': 18.224518011193695, 'alphas30': 2.360020536279188, 'alphas31': 0.8219167587550191, 'alphas32': 1.2032415188597079, 'alphas33': 4.910010585844271, 'alphas34': 0.5012633542572087, 'alphas35': 19.483848297642755, 'alphas4': 18.630449317979277, 'alphas5': 17.78788998244555, 'alphas6': 9.650344686826307, 'alphas7': 5.967145019861386, 'alphas8': 18.502041901497478, 'alphas9': 5.102867493087826}
        # original
        #        a = {'alphas0': 5.757775763196102, 'alphas1': 19.47022986534926, 'alphas10': 6.9077077022907885, 'alphas11': 10.760275351689499, 'alphas12': 5.439034921163273, 'alphas13': 14.500937788995314, 'alphas14': 4.609704094246663, 'alphas15': 19.1295618491843, 'alphas16': 9.69426602173553, 'alphas17': 18.484999960024137, 'alphas18': 2.4543601017987826, 'alphas19': 1.6876807886130085, 'alphas2': 16.657597909289752, 'alphas20': 0.3916057707187348, 'alphas21': 15.549547403572774, 'alphas22': 19.503935456689526, 'alphas23': 16.410209778202272, 'alphas24': 7.165060251240112, 'alphas25': 3.8453829956723085, 'alphas26': 3.7348363005530105, 'alphas27': 0.12072508130120285, 'alphas28': 2.4652898851925653, 'alphas29': 5.112848315383873, 'alphas3': 19.425848183503113, 'alphas30': 1.8116576556390163, 'alphas31': 0.05800503222769704, 'alphas32': 0.37728224164693014, 'alphas33': 19.41853432362186, 'alphas34': 7.970669953067522, 'alphas35': 15.4357424202083433, 'alphas4': 15.392945515390412, 'alphas5': 3.792209734069787, 'alphas6': 16.729303166800726, 'alphas7': 18.74144061007218, 'alphas8': 4.9151908367603525, 'alphas9': 7.8496827150821}
        # a = {'alphas0': 26.566251688696106, 'alphas1': 39.33170371797949, 'alphas2': 37.75837170002072, 'alphas3': 39.845731167478036, 'alphas4': 2.721872066155022, 'alphas5': 18.756069935719964, 'alphas6': 1.3492267973466676, 'alphas7': 37.41423593604899}
        # a = {'alphas0': 18.492065976239715, 'alphas1': 19.49580483022761, 'alphas10': 17.135227334336538, 'alphas11': 0.588217549437613, 'alphas12': 12.420886441387108, 'alphas13': 4.88160556741062, 'alphas14': 18.520096695330263, 'alphas15': 18.035041638012014, 'alphas16': 4.035762066972522, 'alphas17': 8.020746702676469, 'alphas18': 4.768031572121199, 'alphas19': 15.397020868768436, 'alphas2': 19.4015981895277, 'alphas20': 14.869113518032787, 'alphas21': 5.941433272722241, 'alphas22': 1.9282600051566878, 'alphas23': 19.211646340646674, 'alphas24': 2.1874842327868693, 'alphas25': 0.7605279804266196, 'alphas26': 6.295347442176274, 'alphas27': 8.495782331199749, 'alphas28': 1.0824912975066558, 'alphas29': 3.273585568505357, 'alphas3': 3.286716923697215, 'alphas30': 1.3350782620183033, 'alphas31': 0.5335154247882401, 'alphas32': 0.19330014778494942, 'alphas33': 5.58962659608061, 'alphas34': 0.7160633533466543, 'alphas35': 7.726210481505382, 'alphas36': 18.613828925323407, 'alphas37': 17.381410969783765, 'alphas38': 13.606899188230503, 'alphas4': 6.282866560877469, 'alphas5': 15.681784642861658, 'alphas6': 12.700047705244565, 'alphas7': 19.5181873389573, 'alphas8': 8.20262714025359, 'alphas9': 18.927557823468028}
        # a = {'alphas0': 16.46879337343726, 'alphas1': 19.290205549814253, 'alphas10': 1.3762974968040287,
        #      'alphas11': 10.963855108672512, 'alphas12': 5.938205986436033, 'alphas13': 0.528231427181256,
        #      'alphas14': 14.677043860326275, 'alphas15': 14.039165687639123, 'alphas16': 2.3617879772030914,
        #      'alphas17': 18.379651193888012, 'alphas18': 6.788288622140506, 'alphas19': 8.977720747163726,
        #      'alphas2': 14.465427418088714, 'alphas20': 19.744024874355297, 'alphas21': 4.751191262724268,
        #      'alphas22': 19.692190431266916, 'alphas23': 8.998101265227644, 'alphas24': 9.370468528673424,
        #      'alphas25': 19.31505596321069, 'alphas26': 3.4015057714127894, 'alphas27': 6.392417239076038,
        #      'alphas28': 19.011413147552744, 'alphas29': 10.09581795472873, 'alphas3': 0.8411115100800259,
        #      'alphas30': 0.18828475939425937, 'alphas31': 0.8573986471083117, 'alphas32': 2.7815518440985754,
        #      'alphas33': 11.683353476725317, 'alphas34': 19.65902556631257, 'alphas35': 0.12931436588237144,
        #      'alphas36': 19.461629673052958, 'alphas4': 0.498675778505524, 'alphas5': 1.9448304981920317,
        #      'alphas6': 1.338413714039508, 'alphas7': 2.8097382168845497, 'alphas8': 5.838428989584652,
        #      'alphas9': 16.305020660408967}
        a ={'alphas0': 0.0, 'alphas1': 40.0, 'alphas2': 40.0, 'alphas3': 40.0, 'alphas4': 40.0, 'alphas5': 40.0,
         'alphas6': 23.960436176277238, 'alphas7': 21.725866752522446, 'alphas8': 5.464415290150711, 'alphas9': 40.0,
         'alphas10': 0.0, 'alphas11': 16.500911764879305, 'alphas12': 40.0, 'alphas13': 40.0}


        # N_cbf = 2
        # N_cf = 6
        # N_p3a = 1
        # N_ucf = 1
        # N_ucbf = 1
        # N_rp3b = 1
        # N_slim = 0
        # N_als = 1
        # N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_als
        # recsys = []
        # for i in range(N_cbf):
        #     recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
        # for i in range(N_cf):
        #     recsys.append(ItemCFKNNRecommender(URM_train))
        # for i in range(N_p3a):
        #     recsys.append(P3AlphaRecommender(URM_train))
        # for i in range(N_ucf):
        #     recsys.append(UserCFKNNRecommender(URM_train))
        # for i in range(N_ucbf):
        #     recsys.append(UserCBFKNNRecommender(URM_train, ICM_all))
        # for i in range(N_rp3b):
        #     recsys.append(RP3betaRecommender(URM_train))
        # for i in range(N_slim):
        #     recsys.append(SLIM_BPR_Cython(URM_train))
        # recsys.append(ImplicitALSRecommender(URM_train))
        #
        # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
        # recsys_params2 = list((zip(np.linspace(5, 800, N_cf).tolist(), [12] * N_cf)))
        # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
        # recsys_params4 = list((zip(np.linspace(170, 180, N_ucf).tolist(), [2] * N_ucf)))
        # recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
        # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))
        # today
        # N_cbf = 2
        # N_cf = 4
        # N_p3a = 0
        # N_ucf = 2
        # N_ucbf = 1
        # N_rp3b = 1
        # N_slim = 1
        # N_als = 1
        # N_hyb_item_sim = 0
        # N_pure_svd = 0
        # N_hyb = N_cbf + N_cf + N_p3a + N_ucf + N_ucbf + N_rp3b + N_slim + N_als + N_hyb_item_sim + N_pure_svd
        # recsys = []
        # for i in range(N_cbf):
        #     recsys.append(ItemCBFKNNRecommender(URM_train, ICM_all))
        # for i in range(N_cf):
        #     recsys.append(ItemCFKNNRecommender(URM_train))
        # for i in range(N_p3a):
        #     recsys.append(P3AlphaRecommender(URM_train))
        # for i in range(N_ucf):
        #     recsys.append(UserCFKNNRecommender(URM_train))
        # for i in range(N_ucbf):
        #     recsys.append(UserCBFKNNRecommender(URM_train, ICM_all))
        # for i in range(N_rp3b):
        #     recsys.append(RP3betaRecommender(URM_train))
        # for i in range(N_slim):
        #     recsys.append(SLIM_BPR_Cython(URM_train))
        # for i in range(N_als):
        #     recsys.append(ImplicitALSRecommender(URM_train))
        # for i in range(N_pure_svd):
        #     recsys.append(PureSVDRecommender(URM_train))
        #
        # # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
        # # recsys_params2 = list((zip(np.linspace(5, 800, N_cf).tolist(), [12] * N_cf)))
        # # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
        # # recsys_params4 = list((zip(np.linspace(170, 180, N_ucf).tolist(), [2] * N_ucf)))
        # # recsys_params5 = list((zip(np.linspace(170, 180, N_ucbf).tolist(), [5] * N_ucbf)))
        # # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))
        # recsys_params = list(zip(np.linspace(10, 70, N_cbf).tolist(), [4] * N_cbf))
        # recsys_params2 = list((zip(np.linspace(5, 641, N_cf).tolist(), [12] * N_cf)))
        # recsys_params3 = list((zip(np.linspace(99, 101, N_p3a).tolist(), [1] * N_p3a)))
        # recsys_params4 = list((zip(np.linspace(10, 180, N_ucf).tolist(), [2] * N_ucf)))
        # recsys_params5 = list((zip(np.linspace(10, 180, N_ucbf).tolist(), [5] * N_ucbf)))
        # recsys_params6 = list((zip(np.linspace(99, 101, N_rp3b).tolist(), [0] * N_rp3b)))
        #
        # print("Starting fitting single recsys")
        # t = time.time()
        # for i in range(N_cbf):
        #     # print("Training system {:d}...".format(i))
        #     topK = recsys_params[i][0]
        #     shrink = recsys_params[i][1]
        #     recsys[i].fit(topK=topK, shrink=shrink, type="tanimoto")
        # for i in range(N_cf):
        #     # print("Training system {:d}...".format(i+N_cbf))
        #     topK = recsys_params2[i][0]
        #     shrink = recsys_params2[i][1]
        #     recsys[i + N_cbf].fit(topK=topK, shrink=shrink, type="cosine", alpha=0.3)
        # for i in range(N_p3a):
        #     # print("Training system {:d}...".format(i+N_cbf))
        #     topK = recsys_params3[i][0]
        #     shrink = recsys_params3[i][1]
        #     recsys[i + N_cbf + N_cf].fit(topK=topK, shrink=shrink, alpha=0.31)
        # for i in range(N_ucf):
        #     # print("Training system {:d}...".format(i+N_cbf))
        #     topK = recsys_params4[i][0]
        #     shrink = recsys_params4[i][1]
        #     recsys[i + N_cbf + N_cf + N_p3a].fit(topK=topK, shrink=shrink, type="jaccard")
        # for i in range(N_ucbf):
        #     # print("Training system {:d}...".format(i+N_cbf))b
        #     topK = recsys_params5[i][0]
        #     shrink = recsys_params5[i][1]
        #     recsys[i + N_cbf + N_cf + N_p3a + N_ucf].fit(topK=topK, shrink=shrink, type="tanimoto")
        # for i in range(N_rp3b):
        #     # print("Training system {:d}...".format(i+N_cbf))b
        #     topK = int(recsys_params6[i][0])
        #     shrink = recsys_params6[i][1]
        #     recsys[i + N_cbf + N_cf + N_p3a + N_ucf + N_ucbf].fit(topK=topK, alpha=0.5927789387679869,
        #                                                           beta=0.009260542392306892)
        #
        # # load slim bpr
        # slims_dir = "result_experiments/hyb_est_ratings_4/"
        # # recsys[-3].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_100")
        # recsys[-2].loadModel(slims_dir, "SLIM_BPR_Recommender_best_model_300")
        # print("Load complete of slim bpr")
        # el_t = time.time() - t
        # print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))
        #
        # print("Starting fitting als")
        # recsys[-1].fit(alpha=15, factors=495, regularization=0.04388, iterations=20)
        # print("Ended fitting als")
        #
        # print("Starting recommending the est_ratings")
        # t2 = time.time()
        # recsys_est_ratings = []
        # for i in range(0, N_hyb-1):
        #     if i >= N_cbf + N_cf + N_p3a + N_ucf + N_ucbf:
        #         recsys_est_ratings.append(recsys[i].compute_item_score(userList_unique, 160))
        #     else:
        #         recsys_est_ratings.append(recsys[i].estimate_ratings(userList_unique, 160))
        # el_t = time.time() - t2
        # print("Done. Elapsed time: {:02d}:{:06.3f}".format(int(el_t / 60), el_t - 60 * int(el_t / 60)))
        #
        # print("Recommending als")
        # recsys_est_ratings.append(recsys[-1].estimate_ratings(userList_unique, 160))
        #
        #
        # print("Building the alphas")
        #
        # #a = {'alphas0': 19.485713591551153, 'alphas1': 4.71319435578388, 'alphas10': 2.2227435851035993, 'alphas11': 1.9716421945640317, 'alphas12': 7.5238841058329875, 'alphas13': 0.4108412788439586, 'alphas14': 4.526224503200407, 'alphas15': 13.576966307999559, 'alphas16': 0.642187854373879, 'alphas17': 5.275515473418448, 'alphas18': 0.34228595410624507, 'alphas19': 0.34191569651452536, 'alphas2': 5.900450505346193, 'alphas20': 6.980737750232469, 'alphas21': 11.130807458091677, 'alphas22': 9.31624459508006, 'alphas23': 4.321048654234952, 'alphas24': 1.1310553103741783, 'alphas25': 9.604200085107115, 'alphas26': 0.5368035241482083, 'alphas27': 3.0960069679180546, 'alphas28': 3.864988337641473, 'alphas29': 16.706157748846, 'alphas3': 16.320530250453828, 'alphas30': 0.9494730681150165, 'alphas31': 0.7654428335341734, 'alphas32': 13.390283773301505, 'alphas33': 19.218172572103484, 'alphas34': 1.4466507699578846, 'alphas35': 19.8594906640407, 'alphas4': 6.6763597060138675, 'alphas5': 10.575281828871317, 'alphas6': 3.7900979297179593, 'alphas7': 4.420167469234189, 'alphas8': 13.929403030269114, 'alphas9': 1.3122328787554016}
        # #new
        # #a = {'alphas0': 16.9431042554559, 'alphas1': 4.644099562164927, 'alphas10': 9.790511891561398, 'alphas11': 6.128048426612316, 'alphas12': 9.53514404239681, 'alphas13': 9.902211129673606, 'alphas14': 15.455991038784067, 'alphas15': 0.7580787143633927, 'alphas16': 12.700059220322498, 'alphas17': 9.101615923590032, 'alphas18': 1.695541719218232, 'alphas19': 0.4503896520008954, 'alphas2': 1.58112207469844, 'alphas20': 14.468724241787374, 'alphas21': 3.690624747092459, 'alphas22': 9.836989886324046, 'alphas23': 14.492757157503558, 'alphas24': 3.0100555297168197, 'alphas25': 0.3267319282889791, 'alphas26': 4.445374260353121, 'alphas27': 2.280310105876955, 'alphas28': 3.337896350619496, 'alphas29': 0.3619126735978462, 'alphas3': 18.82246365026546, 'alphas30': 1.9871216544044223, 'alphas31': 1.4645937717755753, 'alphas32': 15.805123493865317, 'alphas33': 15.954066854241688, 'alphas34': 17.551287337467674, 'alphas35': 17.77998159521204, 'alphas4': 1.3959735370043913, 'alphas5': 13.537903382766192, 'alphas6': 3.905416743981318, 'alphas7': 5.468378132839908, 'alphas8': 3.981393478813855, 'alphas9': 2.1708067917427343}
        # #a = {'alphas0': 1, 'alphas1': 1, 'alphas2': 1, 'alphas3': 1, 'alphas4': 1, 'alphas5': 1, 'alphas6': 16.46166558659779}
        # #a = {'alphas0': 29.320616314914947, 'alphas1': 3.693662165804521, 'alphas10': 2.712023715017764, 'alphas11': 1.5801059996255729, 'alphas12': 29.73364247338825, 'alphas2': 29.95543186541617, 'alphas3': 11.342620661059287, 'alphas4': 21.98021113873687, 'alphas5': 17.255942788852753, 'alphas6': 29.88989122983309, 'alphas7': 8.72323633950845, 'alphas8': 4.686190019184634, 'alphas9': 27.594768586378013}
        # # a = {'alphas0': 3.8074892263744045, 'alphas1': 6.150219846332722, 'alphas2': 14.151408990046379, 'alphas3': 0.0,
        # #  'alphas4': 5.196499972793797, 'alphas5': 12.25261937360623, 'alphas6': 0.0, 'alphas7': 5.41602547654064,
        # #  'alphas8': 6.196313359091854, 'alphas9': 50.0, 'alphas10': 50.0, 'alphas11': 50.0}
        # a = {'alphas0': 8.940172641315279, 'alphas1': 2.4420214618701825, 'alphas2': 10.877618772020504,
        #  'alphas3': 22.5499796311901, 'alphas4': 0.0, 'alphas5': 22.46819184011107, 'alphas6': 0.0,
        #  'alphas7': 2.0236344507998414, 'alphas8': 0.0, 'alphas9': 0.0, 'alphas10': 40.0, 'alphas11': 40.0}
        print("Init recsys")
        recommender = recommender_class(URM_train, recsys_est_ratings)
        print("Fitting recsys")
        recommender.fit(**a)
        print("Hopefully done")

        #users_excluded_targets = [u for u in userList_unique if u not in targetsListList]
        # result_dict = evaluate_algorithm_targets(URM_validation, recommender, targets=userList_unique, at=10,
        #                                          ours=False)
        # result_dict = evaluate_algorithm_targets(URM_test, recommender, targets=userList_unique, at=10, ours=False)
        # result_dict = evaluate_algorithm_targets(URM_validation, recommender, targets=targetsListList, at=10, ours=False)
        # result_dict = evaluate_algorithm_targets(URM_test, recommender, targets=targetsListList, at=10, ours=False)
        result_dict = evaluate_algorithm_targets(URM_validation, recommender, targets=targetsListOrdered, at=10, ours=False)
        result_dict = evaluate_algorithm_targets(URM_test, recommender, targets=targetsListOrdered, at=10, ours=False)
        # from Base.Evaluation.Evaluator import FastEvaluator
        #
        # users_excluded_targets = [u for u in userList_unique if u not in targetsListList]
        # evaluator_validation_earlystopping = FastEvaluator(URM_validation, cutoff_list=[10], minRatingsPerUser=1,
        #                                                    exclude_seen=True, ignore_users=users_excluded_targets)
        # evaluator_test = FastEvaluator(URM_test, cutoff_list=[10], minRatingsPerUser=1, exclude_seen=True,
        #                                ignore_users=users_excluded_targets)

        #print(evaluator_validation_earlystopping(recommender))
        #print(evaluator_test(recommender))
        #recommender = recommender_class(URM_train, positive_threshold=0.5, URM_validation = None,
        #         recompile_cython = False, final_model_sparse_weights = True, train_with_sparse_weights = False,
        #         symmetric = True)
        # recommender.fit()
        #recommender.fit(epochs=400, logFile=None,
            # batch_size = 1000, lambda_i = 1e-6, lambda_j = 1e-6, learning_rate = 0.01, topK = 700,
            # sgd_mode='adagrad',
            # stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "MAP",
            # evaluator_object = None, validation_every_n = 1000)

        # results_run, results_run_string, user_map = evaluator.evaluateRecommender(recommender)

        # print("Algorithm: {}, results: \n{}".format(recommender.__class__, results_run_string))
        # logFile.write("Algorithm: {}, results: \n{}\n".format(recommender.__class__, results_run_string))
        # logFile.flush()
        #
        # # Submission
        # if JUPYTER:
        #     # Jupyter
        #     target_csv_file = "../../../data/target_playlists.csv"
        # else:
        #     # PyCharm
        #     target_csv_file = "data/target_playlists.csv"
        # target_df = pd.read_csv(target_csv_file)
        # targets = target_df["playlist_id"]
        #
        # #recommendations = recommender.recommend(targets, at=10)  #  recommend_submatrix(targets, at=10)
        # recommendations = recommender.recommend(targets, remove_seen_flag=True,
        #                                           cutoff=10, remove_top_pop_flag=False,
        #                                           remove_CustomItems_flag=False)
        # target_df["track_ids"] = recommendations
        # print(target_df[0:5])
        #
        # # Custom name
        # csv_filename = "hybrid_est_ratings_2"
        # # Default name
        # #csv_filename = "submission_{algtype:}_{date:%Y%m%d%H%M%S}".format(algtype=recommender_class, date=datetime.datetime.now())
        #
        # print_to_csv(target_df, csv_filename)
        #
        # # Save model
        # recommender.saveModel("../../../submissions/", file_name=csv_filename+"_model")

    except Exception as e:
        traceback.print_exc()
        # logFile.write("Algorithm: {} - Exception: {}\n".format(recommender_class, str(e)))
        # logFile.flush()
