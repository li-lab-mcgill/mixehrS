from scipy.special import digamma, gammaln
from sklearn.metrics import  average_precision_score, roc_curve, auc
from scipy.stats import norm
from code import Corpus
import collections
from sklearn.linear_model import LogisticRegression
from code.metrics import precision_recall_curve_metric, roc_curve_metric
import numpy as np
import pickle
import h5py
import os


class MixEHR():
    def __init__(self, K, corpus:Corpus, out='.'):
        """
        Arguments:
            K: Number of topics
            corpus: A length of documents, documents represent as a class. Because doc length varies with each other,
                it's not a D*M matrix.
        """
        self.out = out  # folder to save experiments

        self.C = corpus.C
        self.generator = Corpus.generator_full_batch(corpus)
        self.D = corpus.D
        self.K = K
        self.T = corpus.T  # different data types number
        self.W = corpus.W  # different words number

        self.alpha = np.random.gamma(1, 10, self.K)  # hyperparameter for prior on weight vectors theta
        self.iota = np.random.gamma(1+0.001, 0.001, self.T)  # hyperparameter for prior on type vectors beta
        self.zeta = np.random.gamma(2, 100, (self.W, self.K))  # hyperparameter for prior on type vectors eta
        self.alpha_sum = np.sum(self.alpha)  # scalar value
        self.iota_sum = np.sum(self.iota)  # scalar value
        self.zeta_sum = np.sum(self.zeta, axis=0)  # sum over w, K dimensional
        self.tau = np.random.gamma(2, 0.5, self.K)

        # variational parameters
        self.lambda_ = np.zeros(self.D)
        self.m_ = np.ones(self.K)
        self.s = np.ones(self.K)
        self.exp_g = np.random.normal(size=self.D)
        self.exp_z_avg = np.zeros((self.D, self.K))
        self.exp_q_z = 0

        # token variables
        self.exp_n = np.random.rand(self.D, self.K)
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
        self.lasso = LogisticRegression()

        # Model parameters
        self.parameters = ['alpha', 'iota', 'zeta', 'gamma', 'm_', 's', 'W', 'T']

    def cal_exp_g_j(self, doc):
        '''
        calcualte the expected value of g_j
        '''
        eps = np.finfo(float).eps
        for pat in doc:
            j = pat.patient_id
            if pat.y == 1:
                self.exp_g[j] = self.lambda_[j] + norm.pdf(-self.lambda_[j]) / (eps if 1 - norm.cdf(-self.lambda_[j]) == 0
                                                                                else 1 - norm.cdf(-self.lambda_[j]))
            else:
                self.exp_g[j] = self.lambda_[j] - norm.pdf(-self.lambda_[j]) / (eps if norm.cdf(-self.lambda_[j]) == 0
                                                                                else norm.cdf(-self.lambda_[j]))

    def ELBO(self):
        eps = np.finfo(float).eps
        # E_q[log p(z | alpha)]
        ELBO =  self.batchsize * (gammaln(np.sum(self.alpha)) - np.sum(gammaln(self.alpha)))
        ELBO += np.sum([np.sum(gammaln(self.alpha + self.exp_n[pat.patient_id])) -
                    gammaln(np.sum(self.alpha + self.exp_n[pat.patient_id])) for pat in self.batch_patient])
        # E_q[log p(b | z, iota)]
        ELBO += np.sum([np.sum(gammaln(self.iota + self.exp_m[:, k])) - gammaln(np.sum((self.iota + self.exp_m[:, k])))
                        for k in range(self.K)])
        # E_q[log p(x | b, z, zeta)]
        ELBO += self.K * self.T * (gammaln(np.sum(self.zeta)) - np.sum(gammaln(self.zeta)))
        ELBO += np.sum(np.sum(gammaln(self.zeta + self.exp_p), axis=1) - gammaln(np.sum((self.zeta + self.exp_p), axis=1)))
        # np.sum([[np.sum(gammaln(self.zeta[:, k] + self.exp_p[t, :, k])) - gammaln(np.sum(self.zeta[:, k] + self.exp_p[t, :, k]))
            # for t in range(self.T)] for k in range(self.K)])
        # E_q[log p(g | z, w)] - E_q[log q(g | lambda)]
        for d_index, pat in enumerate(self.batch_patient):
            if not pat.isMissingLabel:
                ELBO += np.dot(self.m_, self.exp_z_avg[d_index])*self.exp_g[pat.patient_id]
                s_matrix = np.diag(self.s)
                ELBO -= 0.5*np.sum([np.dot(np.array([self.m_[k_p]*self.m_[k]+s_matrix[k_p, k] for k in range(self.K)]), self.exp_z_avg[d_index])
                                    *self.exp_z_avg[d_index] for k_p in range(self.K)])
                ELBO -= self.exp_g[pat.patient_id]*self.lambda_[pat.patient_id] - 0.5*self.lambda_[pat.patient_id]**2 + \
                        pat.y*np.log(eps if 1 - norm.cdf(-self.lambda_[pat.patient_id]) == 0 else 1 - norm.cdf(-self.lambda_[pat.patient_id])) \
                        + (1 - pat.y)*np.log(eps if norm.cdf(-self.lambda_[pat.patient_id]) == 0 else norm.cdf(-self.lambda_[pat.patient_id]))
        # E_q[log p(w | tau)] - E_q[log q(w | m, s)]
        ELBO += (0.5 * np.log(self.tau) - 0.5 * self.tau * (self.m_**2 + self.s)).sum() + 0.5 * np.log(self.s).sum()
        # E_q[log p(y | g)] = 0
        # - E_q[log q(z | gamma)]
        ELBO -= self.exp_q_z
        self.exp_q_z = 0
        return ELBO

    def CVB0(self, doc, iter):
        print("this is", iter)
        temp_exp_n = np.zeros((self.D, self.K))
        temp_exp_m = np.zeros((self.T, self.K))
        temp_exp_p = np.zeros((self.T, self.W, self.K))
        # E step
        for pat in doc:
            temp_gamma = np.zeros((len(pat.words_dict), self.K))
            for w_idx, ((t_i, w_i), freq) in enumerate(pat.words_dict.items()):
                temp_gamma[w_idx] = (self.alpha+self.exp_n[pat.patient_id]) \
                        * (self.iota[t_i-1]+self.exp_m[t_i-1]) \
                        / (self.iota_sum+self.exp_m_sum) \
                        * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1])\
                        / (self.zeta_sum+self.exp_p_sum[t_i-1])
                temp_gamma[w_idx] /= np.sum(temp_gamma[w_idx])
                temp_exp_n[pat.patient_id] += temp_gamma[w_idx] * pat.word_freq[w_i]
                temp_exp_m[t_i-1] += temp_gamma[w_idx] * pat.type_freq[t_i]
                temp_exp_p[t_i-1, w_i-1] += temp_gamma[w_idx] * freq
            self.gamma[pat.patient_id] = temp_gamma
            self.exp_z_avg[pat.patient_id] = temp_exp_n[pat.patient_id] / pat.Cj
        # m step
        self.exp_n = temp_exp_n
        self.exp_m = temp_exp_m
        self.exp_p = temp_exp_p
        self.exp_n_sum = np.sum(self.exp_n, axis=1) # sum over k, exp_n is [D K] dimensionality
        self.exp_m_sum = np.sum(self.exp_m, axis=0) # sum over t, exp_m is [T K] dimensionality
        self.exp_p_sum = np.sum(self.exp_p, axis=1) # sum over w, exp_p is [T W K] dimensionality
        self.SVB0_update_hyperparams() # update hyperparameters
        # ctr = collections.Counter(self.exp_z_avg.argmax(axis=1))
        # print(ctr.values())


    def CVB0_test(self, doc, iter):
        print("this is", iter)
        temp_exp_n = np.zeros((self.D, self.K))
        temp_exp_m = np.zeros((self.T, self.K))
        temp_exp_p = np.zeros((self.T, self.W, self.K))

        # E step
        for pat in doc:
            temp_gamma = np.zeros((len(pat.words_dict), self.K))
            for w_idx, counts in enumerate(pat.words_dict.items()):
                (t_i, w_i), freq = counts
                temp_gamma[w_idx] = (self.alpha+self.exp_n[pat.patient_id]) \
                        * (self.iota[t_i-1]+self.exp_m[t_i-1]) \
                        / (self.iota_sum+self.exp_m_sum) \
                        * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1])\
                        / (self.zeta_sum+self.exp_p_sum[t_i-1])
                temp_gamma[w_idx] /= np.sum(temp_gamma[w_idx])
                temp_exp_n[pat.patient_id] += temp_gamma[w_idx] * pat.word_freq[w_i]
                temp_exp_m[t_i-1] += temp_gamma[w_idx] * pat.type_freq[t_i]
                temp_exp_p[t_i-1, w_i-1] += temp_gamma[w_idx] * freq
            self.gamma[pat.patient_id] = temp_gamma
            self.exp_z_avg[pat.patient_id] = temp_exp_n[pat.patient_id] / pat.Cj
        # m step
        self.exp_n = temp_exp_n
        self.exp_m = temp_exp_m
        self.exp_p = temp_exp_p
        self.exp_n_sum = np.sum(self.exp_n, axis=1) # sum over k, exp_n is [D K] dimensionality
        self.exp_m_sum = np.sum(self.exp_m, axis=0) # sum over t, exp_m is [T K] dimensionality
        self.exp_p_sum = np.sum(self.exp_p, axis=1) # sum over w, exp_p is [T W K] dimensionality

    def SVB0_update_hyperparams(self):
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


    def inference_svb(self, max_iter=300, tol=0.01):
        elbo = [100, 0]
        iter =0
        for i, d in enumerate(self.generator):
            batch_patient, batch_i, M = d
            self.gamma = {pat.patient_id: np.random.rand(len(pat.words_dict), self.K) for pat in batch_patient}

        while iter < max_iter:
            for i, d in enumerate(self.generator):
                batch_patient, batch_i, M = d
                self.CVB0(batch_patient, iter)
                # elbo.append(self.ELBO())
                # print("%s elbo %s diff %s "%(iter , elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                iter += 1
                if (iter + 1) % 100 == 0:
                    self.save_model(iter + 1)
                if not iter < max_iter:
                    break
        pickle.dump(elbo, open(os.path.join(self.out, 'elbo_training.pkl'), 'wb'))
        pickle.dump(self.exp_g, open(os.path.join(self.out, 'exp_g_train.pkl'), 'wb'))
        pickle.dump(self.gamma, open(os.path.join(self.out, 'gamma_train.pkl'), 'wb'))

        return self.exp_g, self.gamma


    def predict(self, corpus, max_iter=100):
        self.D = corpus.D
        self.C = corpus.C
        self.generator = Corpus.generator_full_batch(corpus)
        self.batchsize = self.D # performa a full batch
        self.exp_n = np.random.rand(self.D, self.K)
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
        self.exp_z_avg = np.zeros((self.D, self.K))
        self.exp_q_z = 0
        self.exp_g = np.ones(self.D)
        self.lambda_ = np.zeros(self.D)

        elbo = [100, 0]
        iter = 0
        for i, d in enumerate(self.generator):
            batch_patient, batch_i, M = d
            self.gamma = {pat.patient_id: np.random.rand(len(pat.words_dict), self.K) for pat in batch_patient}
            for pat in batch_patient:
                pat.y = -1
                pat.isMissingLabel = True

            while iter < max_iter:
                self.CVB0_test(batch_patient, iter)
                pred_result = self.lasso.predict_proba(self.exp_n)[:, 1]
                avg_pr = average_precision_score(self.y_test, pred_result)
                fpr, tpr, threshold = roc_curve(self.y_test, pred_result, pos_label=1)
                roc_auc_rf = auc(fpr, tpr)
                pickle.dump([self.y_test, pred_result], open('prediction_y_p_test.pkl', 'wb'))
                if (iter + 1) % 50 == 0:
                    self.save_model(iter + 1)
                iter += 1
                if not iter < max_iter:
                    break

    def save_model(self, iter):
        with h5py.File(os.path.join(self.out, 'model_mixehr_%s_%s.hdf5' % (self.K, iter)), 'w') as hf:
            for param in self.parameters:
                if param == 'gamma':
                    pickle.dump(self.gamma, open(os.path.join(self.out, 'gamma%d.pkl' % iter), 'wb'))
                    pickle.dump(self.exp_n, open(os.path.join(self.out, 'exp_n_%d.pkl' % iter), 'wb'))
                    pickle.dump(self.exp_m, open(os.path.join(self.out, 'exp_m_%d.pkl' % iter), 'wb'))
                    pickle.dump(self.exp_p, open(os.path.join(self.out, 'exp_p_%d.pkl' % iter), 'wb'))
                else:
                    hf.create_dataset(param, data=self.__getattribute__(param))

    def load_model(self, model_path, iter=100):
        with h5py.File(os.path.join(model_path), 'r') as hf:
            for param in self.parameters:
                if param == 'gamma':
                    pass
                    import pickle
                    self.gamma = pickle.load(open(os.path.join(self.out, 'gamma%d.pkl' % iter), 'rb'))
                    self.exp_n = pickle.load(open(os.path.join(self.out, 'exp_n_%d.pkl' % iter), 'rb'))
                    self.exp_m = pickle.load(open(os.path.join(self.out, 'exp_m_%d.pkl' % iter), 'rb'))
                    self.exp_p = pickle.load(open(os.path.join(self.out, 'exp_p_%d.pkl' % iter), 'rb'))
                else:
                    self.__setattr__(param, hf[param][...])
        self.lasso.fit(self.exp_n, self.y_train)



if __name__ == '__main__':
    # c_train = Corpus.read_corpus_from_directory("../split/train")
    # c_test = Corpus.read_corpus_from_directory("../split/test")
    c_train = Corpus.read_corpus_from_directory("../dataset/cv1/train")
    c_test = Corpus.read_corpus_from_directory("../dataset/cv1/test")
    y_train_true = np.array([p[0].y for p in c_train])
    y_test_true = np.array([p[0].y for p in c_test])
    K = 100
    mixehr = MixEHR(K, c_train)
    mixehr.y_train = y_train_true
    mixehr.y_test = y_test_true
    # exp_g, gamma = code.inference_svb()
    mixehr.load_model("model_mixehr_100_100.hdf5")
    mixehr.predict(c_test)
