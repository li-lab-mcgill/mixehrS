import numpy as np
from scipy.stats import dirichlet, multinomial
import numpy as np
from numpy.random import dirichlet, multinomial
# from sklearn.decomposition import LatentDirichletAllocation


def sample_docs(D, W, K, length=10, alpha_init=1):
    beta = np.random.rand(K, W)
    beta /= beta.sum(axis=1, keepdims=1)
    assert np.isclose(beta.sum(), K), "{}".format(beta.sum())

    alpha = np.full((1, K), alpha_init)

    data = np.zeros((D, W))
    for d in range(D):
        theta = dirichlet.rvs(alpha.flatten())
        word_probs_doc_i = theta.dot(beta)
        data[d, :] = multinomial.rvs(length, word_probs_doc_i.flatten())
    return data, beta


def sample_docs(K=3, D=100, N=10, V=12):
    """
    Generate toy documents
    Args:
        K: number of topics
        D: number of documents
        N: number of words
        V: vocabulary size
    """
    np.random.seed(6004)

    alpha = np.array([2.0] * K)
    eta = np.array([2.0] * V)

    # 1) Generate topics-word proportion
    beta = np.array([[.9, .1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, .9, .1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, .9, .1, 0, 0, 0, 0, 0, 0]])
    # beta = dirichlet(eta, K)
    # 2) Generate topic propotions for D documents (theta_d)
    theta = dirichlet(alpha, D)

    w = np.zeros((D, V))
    for d in range(D):
        for n in range(N):
            # 3) Sample a topic z_dn and select z_dn^i=1
            z_dnk = multinomial(1, theta[d])
            k = np.argmax(z_dnk)
            # 4) Sample word w_dn from topic beta_{z_dnk}
            w_dnv = multinomial(1, beta[k])
            w[d] += w_dnv
    return w, beta


def generate_data(K, T, W, D, M, a=1, b=1, tau=4):
    '''
    Generates toy data using the generative model of MixEHR.
    :param K: number of topics
    :param T: number of types
    :param W: number of words in the vocabulary
    :param D: number of documents
    :param M: number of words. We assume that all documents have same length.
    :param a: shape parameter of gamma distribution; used to sample the hyper-parameters
    :param b: scale parameter of gamma distribution; used to sample the hyper-parameters
    :param tau: mean used to sample w
    :return: y: response b: types x: words z: topics-assignment g: response
    '''
    # TODO: make Mj variable, but first test results with same length.
    # 0) Generate hyper-parameters
    alpha = np.random.gamma(a, b, K)
    iota = np.random.gamma(a, b, T)
    zeta = np.random.gamma(a, b, (K, W))
    # 1) Generate topic-type proportions
    beta = np.random.dirichlet(iota, K)
    # 2) Generate topic-word-type proportions
    eta = np.zeros((K, T, W))
    for k in range(K):
        eta[k] = np.random.dirichlet(zeta[k], T)
    # 3) Generate document proportion
    theta = np.random.dirichlet(alpha, D)
    # 4) inner plate
    z = np.zeros((M, D, K))
    b = np.zeros((M, D), dtype=np.int32)
    x = np.zeros((M, D), dtype=np.int32)
    for j in range(D):
        for i in range(M):
            # 4.1) sample topic-document-word proportion
            z[i, j] = np.random.multinomial(1, theta[j])
            # 4.2) sample types
            k = np.argmax(z[i, j])
            b_ij = multinomial(1, beta[k])
            t = np.argmax(b_ij)
            b[i, j] = t + 1  # add one, because 0 means absence of type
            # 4.3 sample words
            x_ij = multinomial(1, eta[k, t])
            w = np.argmax(x_ij)
            x[i, j] = w + 1  # add one, because 0 means absence of word token
    # 5) Sample weights w
    z_bar = z.mean(axis=0)
    w = np.random.multivariate_normal(np.zeros(K), tau ** -1 * np.identity(K), 1)
    # 6) Generate response
    g = np.random.normal(z_bar.dot(w.T), 1)
    y = (g > 0).astype(int)
    return y, b, x, z, g


# def run_sample_example():
#     data, beta = sample_docs()
#
#     lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=2.0, topic_word_prior=2.0, learning_method='batch',
#                                     verbose=00, max_iter=100)
#     lda.fit(data)
#     newbeta = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
#
#     print('error:', np.mean(np.abs(beta - newbeta)))
#     print('data', data)
#     print('beta', beta)
#     print('newbeta', newbeta)
#     print(np.argmax(beta, axis=1))
#     print(np.argmax(newbeta, axis=1))


if __name__ == '__main__':
    # D = 10  # number of documents
    # W = 5  # number of words per document
    # K = 3  # number of topics
    # data, topics = sample_docs(D, W, K)
    # print("words counts ({}):\n{}\n\ntopics ({}):\n{}".format(data.shape, data, topics.shape, topics))
    # run_sample_example()
    K = 3
    T = 2
    W = 100
    D = 50
    M = 1000
    y, b, x, z, g = generate_data(K, T, W, D, M, a=1, b=1, tau=4)
