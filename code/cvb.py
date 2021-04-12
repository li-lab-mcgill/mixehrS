from scipy.special import digamma, gammaln
from sklearn.metrics import precision_recall_fscore_support
from scipy.stats import norm
from code import Corpus
from code.metrics import precision_recall_curve_metric, roc_curve_metric
import numpy as np
import pickle
import h5py
import os


class MixEHR():
    def __init__(self, K, corpus:Corpus, batchsize=32, tau0=1000, kappa=0.7, out='.'):
        """
        Arguments:
            K: Number of topics
            corpus: A length of documents, documents represent as a class. Because doc length varies with each other,
                it's not a D*M matrix.
        """
        self.out = out  # folder to save experiments

        self.C = corpus.C
        self.generator = Corpus.generator(corpus, batchsize)
        self.D = corpus.D
        self.K = K
        self.T = corpus.T  # different data types number
        self.W = corpus.W  # different words number
        self.batchsize = batchsize
        self.tau0 = tau0
        self.kappa = kappa
        self.updatect = 0
        self.updatect_test = 0
        self.rho = 0

        self.batch_patient = None
        self.doc_indices = None

        self.alpha = np.random.gamma(1, 10, self.K)  # hyperparameter for prior on weight vectors theta
        self.iota = np.random.gamma(1+0.001, 0.001, self.T)  # hyperparameter for prior on type vectors beta
        self.zeta = np.random.gamma(2, 100, (self.W, self.K))  # hyperparameter for prior on type vectors eta
        self.alpha_sum = np.sum(self.alpha)  # scalar value
        self.iota_sum = np.sum(self.iota)  # scalar value
        self.zeta_sum = np.sum(self.zeta, axis=0)  # sum over w
        self.tau = np.random.gamma(2, 0.5, self.K)

        # variational parameters
        self.gamma = np.random.uniform(0, 1, (self.W, self.K))
        self.lambda_ = np.zeros(self.D)
        self.m_ = np.ones(self.K)
        self.s = np.ones(self.K)
        self.avg_gamma = np.ones((batchsize, self.K))
        self.exp_g = np.random.normal(size=self.D)
        self.exp_z_avg = np.zeros((self.batchsize, self.K))
        self.exp_q_z = 0

        # token variables
        self.exp_n = np.zeros((self.D, self.K))
        self.exp_m = np.zeros((self.T, self.K))
        self.exp_p = np.zeros((self.T, self.W, self.K))
        self.exp_n_sum = np.zeros(self.D)
        self.exp_m_sum = np.zeros(self.K)
        self.exp_p_sum = np.zeros((self.T, self.K))

        # Model parameters
        # TODO: maybe we don't need all of them
        self.parameters = ['alpha', 'iota', 'zeta',
                           # 'tau',                   # save hyperparameters
                       'gamma',
                       # 'lambda_',                                            # save variational parameter
                       # 'exp_n', 'exp_m', 'exp_p',
                       # 'exp_n_sum', 'exp_m_sum', 'exp_p_sum',                # save expectations
                       'm_', 's',
                       # 'exp_g', 'exp_z_avg', 'exp_q_z',                      # save response parameters and expectation
                        'W', 'T']

    def cal_exp_g_j(self):
        '''
        calcualte the expected value of g_j
        '''
        eps = np.finfo(float).eps
        for pat in self.batch_patient:
            # TODO: does this expectation need to be calculated for missing values?
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

    def SCVB0(self, batch_patient, batch_i, M):
        self.batch_patient = batch_patient
        self.doc_indices = batch_i

        temp_exp_n = np.zeros((self.D, self.K))
        intermediate_m = np.zeros((self.T, self.K))
        intermediate_p = np.zeros((self.T, self.W, self.K))
        sum_avg_gamma_g = 0
        sum_avg_gamma_sq = 0
        if self.updatect == 0:
            self.rho = 1
        else:
            self.rho = np.power(self.tau0 + self.updatect, -self.kappa)

        for d_index, pat in enumerate(self.batch_patient):
            temp_gamma = np.zeros((self.W, self.K))
            # burn in process
            for (t_i, w_i), freq in pat.words_dict.items():
                temp_gamma[w_i-1] = (self.alpha+self.exp_n[pat.patient_id]) \
                        * (self.iota[t_i-1]+self.exp_m[t_i-1]) \
                        / (self.iota_sum+self.exp_m_sum) \
                        * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1]) \
                        / (self.zeta_sum+self.exp_p_sum[t_i-1])
                temp_gamma[w_i-1] /= np.sum(temp_gamma[w_i-1])

            for (t_i, w_i), freq in pat.words_dict.items():
                n_ij = temp_gamma[w_i-1] * pat.word_freq[w_i]
                m_ij = temp_gamma[w_i-1] * pat.type_freq[t_i]
                p_ij = temp_gamma[w_i-1] * freq
                gamma_not_i = np.sum([temp_gamma[new_w_i-1] for new_t_i, new_w_i in pat.words_dict if new_w_i != w_i], axis=0)
                temp_gamma[w_i-1] = (self.alpha+self.exp_n[pat.patient_id]-n_ij) \
                        * (self.iota[t_i-1]+self.exp_m[t_i-1]-m_ij) \
                        / (self.iota_sum+self.exp_m_sum-m_ij) \
                        * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1]-p_ij) \
                        / (self.zeta_sum+self.exp_p_sum[t_i-1]-p_ij)
                temp_gamma[w_i-1] *= np.exp((1/pat.Cj) * (self.m_ * self.exp_g[pat.patient_id])
                            - (0.5/pat.Cj**2) * (self.m_**2+self.s) * gamma_not_i)
                temp_gamma[w_i-1] /= np.sum(temp_gamma[w_i-1])
                # update patient topic variable for each document in the minibatch the same way as in CVB0
                temp_exp_n[pat.patient_id] += temp_gamma[w_i-1] * freq
                intermediate_m[t_i-1] += temp_gamma[w_i-1] * freq
                intermediate_p[t_i-1, w_i-1] += temp_gamma[w_i-1] * freq
            self.exp_z_avg[d_index] = temp_exp_n[pat.patient_id] / pat.Cj
            sum_avg_gamma_g += temp_gamma.sum(axis=0) / pat.Cj * self.exp_g[pat.patient_id]
            sum_avg_gamma_sq += (temp_gamma.sum(axis=0) / pat.Cj)**2
            self.exp_q_z += np.sum(temp_gamma * np.ma.log(temp_gamma).filled(0)) # mask non-exist words
        # update global feature-topic variables
        for d_index, pat in enumerate(self.batch_patient):
            self.exp_n[d_index] = temp_exp_n[pat.patient_id]
            # self.exp_n[d_index] = (1 - self.rho)*self.exp_n[d_index] + self.rho*temp_exp_n[pat.patient_id]
        self.exp_m = (1 - self.rho)*self.exp_m + self.rho*(self.C/M)*intermediate_m
        self.exp_p = (1 - self.rho)*self.exp_p + self.rho*(self.C/M)*intermediate_p
        self.exp_n_sum = np.sum(self.exp_n, axis=1) # sum over k, exp_n is [D K] dimensionality
        self.exp_m_sum = np.sum(self.exp_m, axis=0) # sum over t, exp_m is [T K] dimensionality
        self.exp_p_sum = np.sum(self.exp_p, axis=1) # sum over w, exp_p is [T W K] dimensionality
        intermediate_s = 1 / (self.tau + sum_avg_gamma_sq)
        intermediate_m_ = intermediate_s * sum_avg_gamma_g
        # update latent variable m_ and s for Gaussian parameter w
        self.s =  (1 - self.rho)*self.s + self.rho*intermediate_s
        self.m_ = (1 - self.rho)*self.m_ + self.rho*intermediate_m_
        for d_index, pat in enumerate(self.batch_patient):
            self.lambda_[pat.patient_id] = np.dot(self.m_, self.exp_z_avg[d_index])
        # update hyperparameters
        self.SVB0_update_hyperparams()
        self.cal_exp_g_j()
        self.updatect += 1
        print(self.exp_z_avg)
        print("___")
        # import time
        # time.sleep(10)

        # Check prediction
        # y_pred_expg = (self.exp_g>0).astype(np.int)
        # precision = np.sum(y_pred == self.y_train) / self.D
        # print("precison %s" % precision)
        n = self.exp_z_avg[batch_i].dot(self.m_)
        d = np.array([np.sqrt(1 + np.dot(z_avg.dot(np.diag(self.s)), z_avg)) for z_avg in self.exp_z_avg[batch_i]])
        p = norm.cdf(n / d)
        y_pred_bern = np.random.binomial(1, p).flatten()

        # precision_expg, recall_expg, _, _ = precision_recall_fscore_support(self.y_train, y_pred_expg, average='micro')
        # p_exp_g, r_exp_g, aucpr_exp_g = precision_recall_curve_metric(self.y_train, self.exp_g)
        p_bern, r_bern, aucpr_bern = precision_recall_curve_metric(np.array(self.y_train)[batch_i], p)
        # precision_bern, recall_bern, _, _ = precision_recall_fscore_support(self.y_train, y_pred_bern, average='micro')
        print("AUCPR", aucpr_bern)
        # with open(os.path.join(self.out, 'pr_tr_expg_bern.txt'), "a+") as f:
            # f.write(str(p_exp_g) + ' ' + str(r_exp_g))
            # f.write("%s,%s,%s" % (str(p_exp_g), str(r_exp_g), str(aucpr_exp_g)))
            # f.write('\n')
            # f.write("%s,%s,%s" % (str(p_bern), str(r_bern), str(aucpr_bern)))
            # f.write('\n\n')


    def SCVB0_test(self, batch_patient, batch_i, M):
        self.batch_patient = batch_patient
        self.doc_indices = batch_i

        temp_exp_n = np.zeros((self.D, self.K))
        intermediate_m = np.zeros((self.T, self.K))
        intermediate_p = np.zeros((self.T, self.W, self.K))
        if self.updatect_test == 0:
            self.rho = 1
        else:
            self.rho = np.power(self.tau0 + self.updatect, -self.kappa)
        for d_index, pat in enumerate(self.batch_patient):
            temp_gamma = np.zeros((self.W, self.K))
            for (t_i, w_i), freq in pat.words_dict.items():
                n_ij = temp_gamma[w_i-1] * pat.word_freq[w_i]
                m_ij = temp_gamma[w_i-1] * pat.type_freq[t_i]
                p_ij = temp_gamma[w_i-1] * freq
                temp_gamma[w_i-1] = (self.alpha+self.exp_n[pat.patient_id]-n_ij) \
                        * (self.iota[t_i-1]+self.exp_m[t_i-1]-m_ij) \
                        / (self.iota_sum+self.exp_m_sum-m_ij) \
                        * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1]-p_ij) \
                        / (self.zeta_sum+self.exp_p_sum[t_i-1]-p_ij)
                # temp_gamma[w_i-1] = (self.alpha+self.exp_n[pat.patient_id]) * (self.iota[t_i-1]+self.exp_m[t_i-1]) \
                #         / (self.iota_sum+self.exp_m_sum) * (self.zeta[w_i-1]+self.exp_p[t_i-1, w_i-1]) \
                #         / (self.zeta_sum+self.exp_p_sum[t_i-1])
                temp_gamma[w_i-1] /= np.sum(temp_gamma[w_i-1])
                # update patient topic variable for each document in the minibatch the same way as in CVB0
                temp_exp_n[pat.patient_id] += temp_gamma[w_i-1] * freq
                intermediate_m[t_i-1] += temp_gamma[w_i-1] * freq
                intermediate_p[t_i-1, w_i-1] += temp_gamma[w_i-1] * freq
            self.exp_z_avg[d_index] = temp_exp_n[pat.patient_id] / pat.Cj
            self.exp_q_z += np.sum(temp_gamma * np.ma.log(temp_gamma).filled(0)) # mask non-exist words
            self.lambda_[pat.patient_id] = np.dot(self.m_, temp_gamma.sum(axis=0) / pat.Cj)
        # update global feature-topic variables
        self.exp_n = temp_exp_n
        self.exp_m = (1 - self.rho)*self.exp_m + self.rho*(self.C/M)*intermediate_m
        self.exp_p = (1 - self.rho)*self.exp_p + self.rho*(self.C/M)*intermediate_p
        self.exp_n_sum = np.sum(self.exp_n, axis=1) # sum over k, exp_n is [D K] dimensionality
        self.exp_m_sum = np.sum(self.exp_m, axis=0) # sum over t, exp_m is [T K] dimensionality
        self.exp_p_sum = np.sum(self.exp_p, axis=1) # sum over w, exp_p is [T W K] dimensionality
        self.updatect_test += 1
        self.cal_exp_g_j()


    def SVB0_update_hyperparams(self):
        '''
        update hyperparameters alpha, iota and zeta
        '''
        # compute intermedaite alpha*
        intermediate_alpha = (1 - 1 + self.alpha * np.sum([digamma(self.alpha + self.exp_n[j]) - digamma(self.alpha) for j in self.doc_indices], axis=0)) /\
                     (10 + np.sum([digamma(self.alpha_sum + self.exp_n_sum[j]) - digamma(self.alpha_sum) for j in self.doc_indices]))
        # compute intermedaite iota*
        intermediate_iota = (0.001 + self.iota * np.sum([(digamma(self.iota + self.exp_m[:, k]) - digamma(self.iota)) for k in range(self.K)], axis=0)) /\
                    (0.001 + np.sum([(digamma(self.iota_sum + self.exp_m_sum[k]) - digamma(self.iota_sum)) for k in range(self.K)]))
        # compute intermedaite zeta*
        intermediate_zeta = (2 - 1 + self.zeta * (np.sum([digamma(self.zeta + self.exp_p[t]) for t in range(self.T)], axis=0) - self.T * digamma(self.zeta))) /\
                    (100 + np.sum([digamma(self.zeta_sum + self.exp_p_sum[t]) for t in range(self.T)], axis=0) - self.T * digamma(self.zeta_sum))
        self.alpha = (1 - self.rho)*self.alpha + self.rho*intermediate_alpha
        self.iota = (1 - self.rho)*self.iota + self.rho*intermediate_iota
        self.zeta = (1 - self.rho)*self.zeta + self.rho*intermediate_zeta
        self.alpha_sum = np.sum(self.alpha) # scalar value
        self.iota_sum = np.sum(self.iota) # scalar value
        self.zeta_sum = np.sum(self.zeta, axis=0) # sum over w, zeta is [W, K] dimensionality


    def inference_svb(self, max_iter=500, tol=0.01):
        elbo = [100, 0]
        iter =0
        while iter < max_iter:
            for i, d in enumerate(self.generator):
                batch_patient, batch_i, M = d
                self.SCVB0(batch_patient, batch_i, M)
                elbo.append(self.ELBO())
                print("%s elbo %s diff %s "%(iter , elbo[-1], np.abs(elbo[-1] - elbo[-2])))
                iter += 1
                if (iter + 1) % 100 == 0:
                    self.save_model(iter + 1)
                if not iter < max_iter:
                    break
        pickle.dump(elbo, open(os.path.join(self.out, 'elbo_training.pkl'), 'wb'))
        pickle.dump(self.exp_g, open(os.path.join(self.out, 'exp_g_train.pkl'), 'wb'))
        pickle.dump(self.gamma, open(os.path.join(self.out, 'gamma_train.pkl'), 'wb'))

        return self.exp_g, self.gamma


    def predict(self, corpus, max_iter=500):
        self.D = corpus.D
        self.C = corpus.C
        self.batchsize = self.D # performa a full batch

        self.exp_n = np.zeros((self.D, self.K))
        self.exp_m = np.zeros((self.T, self.K))
        self.exp_p = np.zeros((self.T, self.W, self.K))
        self.exp_n_sum = np.zeros(self.D)
        self.exp_m_sum = np.zeros(self.K)
        self.exp_p_sum = np.zeros((self.T, self.K))

        self.exp_z_avg = np.zeros((self.D, self.K))
        self.exp_q_z = 0

        self.exp_g = np.ones(self.D)
        self.lambda_ = np.zeros(self.D)

        elbo = [100, 0]
        iter =0

        true_labels = self.y_test
        for i, d in enumerate(Corpus.generator(corpus, batch_size=self.batchsize)):
            batch_patient, batch_i, M = d

            result_bernoulli = []
            results_exp_g = []

            for pat in batch_patient:
                # label = pat.y
                pat.y = -1
                pat.isMissingLabel = True
                # true_labels.append(label)
            while iter < max_iter:
                self.SCVB0_test(batch_patient, batch_i, M)
                elbo.append(self.ELBO())
                print("%s elbo %s diff %s" % (iter, elbo[-1], np.abs(elbo[-1] - elbo[-2])))

                # if (iter + 1) % 10 == 0:
                print(self.exp_z_avg)
                n = self.exp_z_avg.dot(self.m_)
                print(n)
                d = np.array([np.sqrt(1 + np.dot(z_avg.dot(np.diag(self.s)), z_avg)) for z_avg in self.exp_z_avg])
                print(d)
                p = norm.cdf(n / d)
                y = np.random.binomial(1, p).flatten()

                # stats_bern = precision_recall_fscore_support(true_labels, y, average='micro')
                # stats_exp_g = precision_recall_fscore_support(true_labels, (self.exp_g > 0).astype(np.int), average='micro')
                stats_bern = precision_recall_curve_metric(true_labels, p)
                print(stats_bern[2])
                # stats_exp_g = precision_recall_curve_metric(true_labels, self.exp_g)
                # print(stats_exp_g[2])
                # result_bernoulli.append((iter + 1, stats_bern))
                # results_exp_g.append((iter + 1, stats_exp_g))
                # import time
                # time.sleep(10)
                # print("\t exp g", self.exp_g)
                # print("\t n", n)
                # print("\t precision bern %.4f exp_g %.4f" %(stats_bern[0], stats_exp_g[0]))
                # print("\t recall bern %.4f p exp_g %.4f" %(stats_bern[1], stats_exp_g[1]))
                # print("\t aucpr bern %.4f aucpr exp_g %.4f" %(stats_bern[2], stats_exp_g[2]))
                # print("\t aucpr bern %.4f" %(stats_bern[2]))

                # ## save prediction
                # with open(os.path.join(self.out, 'pr_ts_expg_bern.txt'), "a+") as f:
                #     f.write("%s %s \n" % (stats_exp_g[0], stats_exp_g[1]))
                #     f.write("%s %s \n" % (stats_bern[0], stats_bern[1]))
                #     f.write('\n')

                iter += 1
                if not iter < max_iter:
                    break


    def save_model(self, iter):
        with h5py.File(os.path.join(self.out, 'model_mixehr_%s_%s.hdf5' % (self.K, iter)), 'w') as hf:
            for param in self.parameters:
                hf.create_dataset(param, data=self.__getattribute__(param))


    def load_model(self, model_path):
        with h5py.File(os.path.join(model_path), 'r') as hf:
            for param in self.parameters:
                print(param, hf[param][...].shape)
                self.__setattr__(param, hf[param][...])


if __name__ == '__main__':
    c_train = Corpus.read_corpus_from_directory("../split/train")
    c_test = Corpus.read_corpus_from_directory("../split/test")

    # Sample data
    # c = Corpus.build_corpus_sample(K, T=3, W=500, D=15, M=10)
    # c_test = Corpus.build_corpus_sample(K, T=3, W=500, D=10, M=10)

    y_train_true = [p[0].y for p in c_train]
    y_test_true = [p[0].y for p in c_test]

    print("y train true", y_train_true)
    print("y test true", y_test_true)

    K = 50
    mixehr = MixEHR(K, c_train)
    mixehr.y_train = y_train_true
    mixehr.y_test = y_test_true
    exp_g, gamma = mixehr.inference_svb()
    mixehr.predict(c_test)
    # code.load_model("./model_mixehr_50_500.hdf5")
    # code.predict(c_test)

