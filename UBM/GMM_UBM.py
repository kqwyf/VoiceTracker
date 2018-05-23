import numpy as np
from scipy import sparse


class GMM_UBM:
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.w = None
        self.mixture_num = 1

    def gmm_em(self, data, mixture_num, final_niter, ds_factor):
        '''
        :param data: 特征序列，其中每个的维度都是dim_num * frame_num
        :param mixture_num: 高斯分布个数
        :param final_niter: 最后一次分裂所需要的EM迭次数
        :param ds_factor:
        :return:
        '''
        global_mu, global_variance = self.comp_gm_gv(data)
        self.gmm_init(global_mu, global_variance)
        niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 15]
        max_pow = int(np.log2(mixture_num))
        niter[max_pow] = final_niter
        for c in range(max_pow + 1):
            mix = 1 << c
            if mix >= mixture_num / 2:
                ds_factor = 1
            for iter in range(niter[c]):
                N = 0; F = 0; S = 0; L = 0; frame_num = 0;
                for d in data:
                    mu = np.copy(self.mu)
                    sigma = np.copy(self.sigma)
                    w = np.copy(self.w)
                    n, f, s, l = self.expectation(d[:, 0:: ds_factor], mu, sigma, w)
                    N = N + n
                    F = F + f
                    S = S + s
                    L = L + l.sum()
                    frame_num = frame_num + len(l);
                self.maximization(N, F, S)
            if c < max_pow:
                self.mixup()

    def comp_gm_gv(self, data):
        frame_num = sum([d.shape[1] for d in data])
        mu = sum([d.sum(1) for d in data]).reshape(-1, 1) / frame_num
        sigma = sum([(d[:, i].reshape(-1, 1) - mu) ** 2 for d in data for i in range(d.shape[1])]) / (frame_num - 1)
        return mu, sigma

    def gmm_init(self, global_mu, global_sigma):
        self.mu = global_mu
        self.sigma = global_sigma
        self.w = np.array([[1]])

    def read_feature(self, filename):
        feateureDict = np.load(filename).item()
        self.mu = feateureDict['mu']
        self.sigma = feateureDict['sigma']
        self.w = feateureDict['w']
        self.mixture_num = self.mu.shape[1]

    def save_feature(self, filename):
        featureDict = dict()
        featureDict['sigma'] = self.sigma
        featureDict['mu'] = self.mu
        featureDict['w'] = self.w
        np.save(filename, featureDict)

    def map_adapt(self, data, ubm, tau=15, config = 'mvw'):
        N = 0
        F = 0
        S = 0
        ubm_mu = np.copy(ubm.mu)
        ubm_sigma = np.copy(ubm.sigma)
        ubm_w = np.copy(ubm.w)
        for d in data:
            n, f, s, llk = self.expectation(d, ubm_mu, ubm_sigma, ubm_w)
            N = N + n; F = F + f; S = S + s;

        dim_num = F.shape[0]
        alpha = N / (N + tau)
        m_ML = F / np.tile(N, (dim_num, 1))
        m = ubm_mu * np.tile(1 - alpha, (dim_num, 1)) + m_ML * np.tile(alpha, (dim_num, 1))
        self.mu = m

        v_ML = S / np.tile(N, (dim_num, 1))
        v = (ubm_sigma + ubm_mu ** 2) * np.tile(1 - alpha, (dim_num, 1)) + v_ML * np.tile(alpha, (dim_num, 1)) - m ** 2
        self.sigma = v

        w_ML = N / sum(N)
        w = ubm_w * (1 - alpha) + w_ML * alpha
        w = w / sum(w)
        self.w = w

    def expectation(self, data, mu, sigma, w):
        post, llk = self.postprob(data, mu, sigma, w)
        N = post.sum(1).T
        F = data @ post.T
        S = (data * data) @ post.T
        return N, F, S, llk

    def postprob(self, data, mu, sigma, w):
        post = self.lgmmprob(data, mu, sigma, w)
        llk = self.logsumexp(post, 0)
        post = np.exp(post - np.tile(llk, (post.shape[0], 1)))
        return post, llk

    def lgmmprob(self, data, mu, sigma, w):
        dim_num, frame_num = data.shape
        c = (mu * mu / sigma).sum(0).reshape(1, -1) + np.log(sigma).sum(0).reshape(1, -1)
        d = (1 / sigma).T @ (data * data) - 2 * (mu / sigma).T @ data + dim_num * np.log(2 * np.pi)
        log_prob = -0.5 * (np.tile(c.T, (1, frame_num)) + d)
        log_prob = log_prob + np.tile(np.log(w).reshape(-1, 1), (1, frame_num))
        return log_prob

    def logsumexp(self, x, dim):
        dim_num, frame_num = x.shape
        xmax = x.max(dim)
        if dim == 0:
            xmax = xmax.reshape(1, -1)
            y = xmax + np.log(np.exp(x - np.tile(xmax, (dim_num, 1))).sum(dim))
        elif dim == 1:
            xmax = xmax.reshape(-1, 1)
            y = xmax + np.log(np.exp(x - np.tile(xmax, (1, frame_num))).sum(dim))
        ind = np.where(np.logical_not(np.isfinite(xmax)))
        y[ind] = xmax[ind]
        return y

    def maximization(self, N, F, S):
        ndims = F.shape[0]
        w = N / N.sum()
        mu = F / np.tile(N, (ndims, 1))
        sigma = S / np.tile(N, (ndims, 1)) - mu * mu
        sigma = self.apply_var_floors(w, sigma, 0.1)
        self.w = w; self.mu = mu; self.sigma = sigma;

    def apply_var_floors(self, w, sigma, floor_const):
        vFloor = (sigma @ w.T).reshape(-1, 1) * floor_const
        sigma = np.maximum(sigma, np.tile(vFloor, (1, sigma.shape[1])))
        return sigma

    def mixup(self):
        mu = self.mu; sigma = self.sigma; w = self.w
        dim_num, frame_num = sigma.shape
        sig_max, arg_max = np.max(sigma, 0), np.argmax(sigma, 0)
        eps = sparse.csc_matrix((np.sqrt(sig_max), (arg_max, range(frame_num))), shape=(dim_num, frame_num))
        mu = np.concatenate((mu - eps, mu + eps), 1)
        sigma = np.concatenate((sigma, sigma), 1)
        w = np.concatenate((w, w)).reshape(1, -1) * 0.5
        self.sigma = sigma; self.mu = mu; self.w = w;

    def compute_llk(self, data, mu, sigma, w):
        post = self.lgmmprob(data, mu, sigma, w)
        llk = self.logsumexp(post, 0)
        return llk

    def score_gmm_trials(self, models, test_files, trials, ubm):
        trial_num = trials.shape[0]
        llr = np.zeros([trial_num, 1])
        for i in range(trial_num):
            gmm = models[trials[i, 0]]
            fea = test_files[trials[i, 1]]
            ubm_llk = self.compute_llk(fea, ubm.mu, ubm.sigma, ubm.w)
            gmm_llk = self.compute_llk(fea, gmm.mu, gmm.sigma, gmm.w)
            llr[i] = np.mean(gmm_llk - ubm_llk)
        return llr


def main():
    pass

if __name__ == '__main__':
    pass
