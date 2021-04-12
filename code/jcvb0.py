from scipy.special import digamma, gammaln
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import norm
from code import Corpus
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold, StratifiedKFold
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from code.metrics import precision_recall_curve_metric, roc_curve_metric
import numpy as np
import pickle
import h5py
import os


class MixEHR():
    def __init__(self, K, numTypes, numVocab, out='.'):
        self.out = out  # folder to save experiments
        self.T = numTypes  # different data types number
        self.W = numVocab  # different words number
        self.K = K

        self.alpha = np.random.gamma(1, 10, self.K)  # hyperparameter for prior on weight vectors theta
        self.iota = np.random.gamma(1+0.001, 0.001, self.T)  # hyperparameter for prior on type vectors beta
        self.zeta = np.random.gamma(2, 100, (self.W, self.K))  # hyperparameter for prior on type vectors eta
        self.alpha_sum = np.sum(self.alpha)  # scalar value
        self.iota_sum = np.sum(self.iota)  # scalar value
        self.zeta_sum = np.sum(self.zeta, axis=0)  # sum over w, K dimensional
        self.tau = np.random.gamma(2, 0.5, self.K)

        # Model parameters to save
        # TODO: maybe we don't need all of them
        self.parameters = ['alpha', 'iota', 'zeta',
                           # 'tau',                   # save hyperparameters
                       'gamma',
                       # 'lambda_',                                            # save variational parameter
                       # 'exp_n', 'exp_m', 'exp_p',
                       # 'exp_n_sum', 'exp_m_sum', 'exp_p_sum',                # save expectations
                       # 'm_', 's',
                       # 'exp_g', 'exp_z_avg', 'exp_q_z',                      # save response parameters and expectation
                        'W', 'T']

    def init_variational_params(self):
        # variational parameters
        # self.lambda_ = np.zeros(self.D)
        # self.m_ = np.ones(self.K)
        # self.s = np.ones(self.K)
        # self.exp_g = np.random.normal(size=self.D)
        self.exp_z_avg = np.zeros((self.D, self.K))
        # self.exp_q_z = 0

    def init_expectations(self, update):
        # token variables
        self.exp_n = np.random.rand(self.D, self.K)
        if not update:
            self.exp_m = np.random.rand(self.T, self.K)
            self.exp_p = np.random.rand(self.T, self.W, self.K)
            for d in range(self.D):
                self.exp_n[d] /= np.sum(self.exp_n[d])
            for t in range(self.T):
                self.exp_m[t] /= np.sum(self.exp_m[t])
            for t in range(self.T):
                for w in range(self.W):
                    self.exp_p[t, w] /= np.sum(self.exp_p[t, w])
            self.exp_n_sum = np.sum(self.exp_n, axis=1) # sum over k, exp_n is [D K] dimensionality
            self.exp_m_sum = np.sum(self.exp_m, axis=0) # sum over t, exp_m is [T K] dimensionality
            self.exp_p_sum = np.sum(self.exp_p, axis=1) # sum over w, exp_p is [T W K] dimensionality

    def CVB0(self, patients, update_hyper):
        temp_exp_m = np.zeros((self.T, self.K))
        temp_exp_p = np.zeros((self.T, self.W, self.K))

        # E step
        for pat in patients:
            temp_gamma = np.zeros((len(pat.words_dict), self.K))
            temp_exp_n = np.zeros(self.K)
            for w_idx, ((t_i, w_i), freq) in enumerate(pat.words_dict.items()):
                temp_gamma[w_idx] = (self.alpha+self.exp_n[pat.patient_id]) \
                        * (self.iota[t_i-1]+self.exp_m[t_i-1]) \
                        / (self.iota_sum+self.exp_m_sum) \
                        * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1])\
                        / (self.zeta_sum+self.exp_p_sum[t_i-1])
                temp_gamma[w_idx] /= np.sum(temp_gamma[w_idx])
                temp_exp_n += temp_gamma[w_idx] * pat.word_freq[w_i]
                temp_exp_m[t_i-1] += temp_gamma[w_idx] * pat.type_freq[t_i]
                temp_exp_p[t_i-1, w_i-1] += temp_gamma[w_idx] * freq
                self.exp_n[pat.patient_id] = temp_exp_n
            self.gamma[pat.patient_id] = temp_gamma
            self.exp_z_avg[pat.patient_id] = temp_exp_n / pat.Cj

        # m step
        if not update_hyper:
            self.exp_m = temp_exp_m
            self.exp_p = temp_exp_p
            self.exp_n_sum = np.sum(self.exp_n, axis=1) # sum over k, exp_n is [D K] dimensionality
            self.exp_m_sum = np.sum(self.exp_m, axis=0) # sum over t, exp_m is [T K] dimensionality
            self.exp_p_sum = np.sum(self.exp_p, axis=1) # sum over w, exp_p is [T W K] dimensionality

        # update hyperparameters
        if not update_hyper:
            self.update_hyperparams()

        # print(self.exp_z_avg)
        # print(self.exp_z_avg.argmax(axis=1))
        # print("finish")


    def update_hyperparams(self):
        '''
        update hyperparameters alpha, iota and zeta
        '''
        # compute intermedaite alpha*
        self.alpha = (1 - 1 + self.alpha * np.sum([digamma(self.alpha + self.exp_n[j]) - digamma(self.alpha) for j in range(self.D)], axis=0)) /\
                     (10 + np.sum([digamma(self.alpha_sum + self.exp_n_sum[j]) - digamma(self.alpha_sum) for j in range(self.D)]))
        # compute intermedaite iota*
        self.iota = (0.001 + self.iota * np.sum([(digamma(self.iota + self.exp_m[:, k]) - digamma(self.iota)) for k in range(self.K)], axis=0)) /\
                    (0.001 + np.sum([(digamma(self.iota_sum + self.exp_m_sum[k]) - digamma(self.iota_sum)) for k in range(self.K)]))
        # compute intermedaite zeta*
        self.zeta = (2 - 1 + self.zeta * (np.sum([digamma(self.zeta + self.exp_p[t]) for t in range(self.T)], axis=0) - self.T * digamma(self.zeta))) /\
                    (100 + np.sum([digamma(self.zeta_sum + self.exp_p_sum[t]) for t in range(self.T)], axis=0) - self.T * digamma(self.zeta_sum))
        self.alpha_sum = np.sum(self.alpha) # scalar value
        self.iota_sum = np.sum(self.iota) # scalar value
        self.zeta_sum = np.sum(self.zeta, axis=0) # sum over w, zeta is [W, K] dimensionality


    def infer(self, corpus:Corpus, infer_only=False, predict=False, max_iter=500, tol=1e-4):
        elbo = [100, 0]
        iter = 0
        diff = 1

        # init containers
        self.C = corpus.C
        self.D = corpus.D
        self.init_variational_params()
        self.init_expectations(infer_only)

        # sample a full batch of corpus
        generator = Corpus.generator_full_batch(corpus)

        # init gamma uniformly
        for i, d in enumerate(generator):
            batch_patient, batch_i, M = d
            self.gamma = {pat.patient_id: np.random.rand(len(pat.words_dict), self.K) for pat in batch_patient}

        while iter < max_iter and diff > tol:
            for i, d in enumerate(generator):
                batch_patient, batch_index, M = d
                old_gamma = self.gamma.copy()

                # infer topics
                self.CVB0(batch_patient, infer_only)

                # test convergence
                # elbo.append(self.ELBO())
                iter += 1
                diff = np.mean([np.mean(np.abs(old_gamma[i] - self.gamma[i])) for i in range(len(batch_patient))])
                print("it %d. diff: %.5f " % (iter, diff))

                # predict
                if predict:
                    self.predict(corpus.labels)

                if (iter + 1) % 100 == 0:
                    self.save_model(iter + 1)
                if iter < max_iter and diff > tol:
                    break
        pickle.dump(elbo, open(os.path.join(self.out, 'elbo_training.pkl'), 'wb'))
        pickle.dump(self.gamma, open(os.path.join(self.out, 'gamma_train.pkl'), 'wb'))

        return self.gamma


    def predict(self, true_labels, nfolds=5):
        X = np.zeros((self.D, self.K))
        y = true_labels

        # for pat_ix, word_topic in self.gamma.items():
        #     X[pat_ix] = word_topic.mean(axis=0)

        X = self.exp_n

        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

        # k_folds = KFold(nfolds, shuffle=True, random_state=42)
        self.lasso = Lasso()
        skfold = StratifiedKFold(nfolds)
        cv_results = cross_validate(self.lasso, X, y, cv=skfold, return_train_score=True,
                                    scoring=['average_precision', 'roc_auc'])

        print("\tCV(%d) AP train: %.2f, test: %.2f - AUROC train: %.2f, test: %.2f" %
                    (nfolds,
                     cv_results['train_average_precision'].mean(),
                     cv_results['test_average_precision'].mean(),
                     cv_results['train_roc_auc'].mean(),
                     cv_results['test_roc_auc'].mean()))


    def save_model(self, iter):
        with h5py.File(os.path.join(self.out, 'model_mixehr_%s_%s.hdf5' % (self.K, iter)), 'w') as hf:
            for param in self.parameters:
                if param == 'gamma':
                    # TODO: Fix error caused for gamma since it is a dictionary
                    # Error "TypeError: Object dtype dtype('O') has no native HDF5 equivalent"
                    import pickle
                    pickle.dump(self.gamma, open(os.path.join(self.out, 'model_mixehr_%s_%s.hdf5' % (self.K, iter)), 'wb'))
                else:
                    hf.create_dataset(param, data=self.__getattribute__(param))


    def load_model(self, model_path):
        with h5py.File(os.path.join(model_path), 'r') as hf:
            for param in self.parameters:
                self.__setattr__(param, hf[param][...])




def train_cv():
    kf = StratifiedKFold(5, shuffle=True, random_state=42)
    folder = '/Users/cuent/Downloads/processed_new/delete1'
    corpus = Corpus.read_corpus_from_directory(folder + "/train")
    for train_index, test_index in kf.split(corpus, corpus.labels):
        for i in train_index:
            corpus.dataset[i].train = True
        for i in test_index:
            corpus.dataset[i].train = False

if __name__ == '__main__':
    folder = '/Users/cuent/Downloads/processed_new/delete1'
    c_train = Corpus.read_corpus_from_directory(folder + "/train")
    # c_test = Corpus.read_corpus_from_directory(folder + "/test")
    # c_train = Corpus.read_corpus_from_directory("../split/train")
    # c_test = Corpus.read_corpus_from_directory("../split/test")

    K = 21
    mixehr = MixEHR(K, c_train.T, c_train.W)
    gamma = mixehr.infer(c_train, predict=True)
    # code.predict(c_test)
    # train_cv()


