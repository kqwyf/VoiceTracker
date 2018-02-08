import numpy as np
from scipy import sparse
class GMM_UBM:
    def __init__(self):
        self.mu = None
        self.sigma = None
        self.w = None
        self.nmix = 1
    def fitgmm(self, data, nmix, final_niter, ds_factor):
        # 初始化GMM参数
        nframes = sum([d.shape[1] for d in data])
        mu = sum([d.sum(1) for d in data]).reshape(-1, 1) / nframes
        sigma = sum([(d[:, i].reshape(-1, 1) - mu) ** 2 for d in data for i in range(d.shape[1])]) / (nframes - 1)
        self.mu = mu; self.sigma = sigma; self.w = np.array([[1]]);
        niter = [1, 2, 4, 4, 4, 4, 6, 6, 10, 10, 15]
        maxpow = int(np.log2(nmix))
        niter[maxpow] = final_niter
        for c in range(maxpow + 1):
            mix = 1 << c
            if mix >= nmix / 2:
                ds_factor = 1
            for iter in range(niter[c]):
                N = 0; F = 0; S = 0; nframes = 0;
                for d in data:
                    n, f, s = self.expectation(d[:, 0 : : ds_factor])
                    N = N + n; F = F + f; S = S + s; nframes = nframes + d.shape[1];
                self.maximization(N, F, S)
            if c < maxpow:
                self.mixup()
    def readFeature(self, filename):
        feateureDict = np.load(filename).item()
        self.mu = feateureDict['mu']
        self.sigma = feateureDict['sigma']
        self.w = feateureDict['w']
        self.nmix = self.mu.shape[1]
    def saveFeature(self, filename):
        featureDict = dict()
        featureDict['sigma'] = self.sigma
        featureDict['mu'] = self.mu
        featureDict['w'] = self.w
        np.save(filename, featureDict)
    def mapAdapt(self, data, config = 'm', tau = 15):
        N = 0; F = 0; S = 0;
        for d in data:
            n, f, s = self.expectation(d)
            N = N + n; F = F + f; S = S + s;
        ndims = F.shape[0]
        gmmMu = self.mu.copy(); gmmSigma = self.sigma.copy(); gmmW = self.w.copy();
        alpha = N / (N + tau)
        if 'm' in config:
            m_ML = F / np.tile(N, (ndims, 1))
            m = gmmMu * np.tile(1 - alpha, (ndims, 1)) + m_ML * np.tile(alpha, (ndims, 1))
            self.mu = m
        if 'v' in config:
            v_ML = S / np.tile(N, (ndims, 1))
            v = (gmmSigma + gmmMu ** 2) * np.tile(1 - alpha, (ndims, 1)) + v_ML * np.tile(alpha, (ndims, 1)) - m ** 2
            self.sigma = v
        if 'w' in config:
            w_ML = N / sum(N)
            w = gmmW * (1 - alpha) + w_ML * alpha
            w = w / sum(w)
            self.w = w
    def expectation(self, data):
        post = self.lgmmprob(data)
        llk = self.logsumexp(post, 0)
        post = np.exp(post - np.tile(llk, (post.shape[0], 1)))
        N = post.sum(1).T
        F = data @ post.T
        S = (data * data) @ post.T
        return N, F, S
    def lgmmprob(self, data):
        ndim, nframe = data.shape
        sigma = np.copy(self.sigma); mu = np.copy(self.mu); w = np.copy(self.w)
        C = (mu ** 2 / sigma).sum(0).reshape(1, -1) + np.log(sigma).sum(0).reshape(1, -1)
        D = (1 / sigma).T @ (data ** 2) - 2 * (mu / sigma).T @ data\
        + ndim * np.log(2 * np.pi)
        logprob = -0.5 * (np.tile(C.T, (1, nframe)) + D)
        logprob = logprob + np.tile(np.log(w).reshape(-1, 1), (1, nframe))
        return logprob
    def logsumexp(self, x, dim):
        ndims, nframes = x.shape
        xmax = x.max(dim)
        if dim == 0:
            xmax = xmax.reshape(1, -1)
            y = xmax + np.log(np.exp(x - np.tile(xmax, (ndims, 1))).sum(dim))
        elif dim == 1:
            xmax = xmax.reshape(-1, 1)
            y = xmax + np.log(np.exp(x - np.tile(xmax, (1, nframes))).sum(dim))
        ind = np.where(np.logical_not(np.isfinite(xmax)))
        y[ind] = xmax[ind]
        return y
    def maximization(self, N, F, S):
        ndims = F.shape[0]
        w = N / N.sum()
        mu = F / np.tile(N, (ndims, 1))
        sigma = S / np.tile(N, (ndims, 1)) - mu * mu
        vFloor = (sigma @ w.T).reshape(-1, 1) * 0.1
        sigma = np.maximum(sigma, np.tile(vFloor, (1, sigma.shape[1])))
        self.w = w; self.mu = mu; self.sigma = sigma;
    def mixup(self):
        mu = self.mu; sigma = self.sigma; w = self.w
        ndims, nframes = sigma.shape
        sig_max, arg_max = np.max(sigma, 0), np.argmax(sigma, 0)
        eps = sparse.csc_matrix((np.sqrt(sig_max), (arg_max, range(nframes))), shape=(ndims, nframes))
        mu = np.concatenate((mu - eps, mu + eps), 1)
        sigma = np.concatenate((sigma, sigma), 1)
        w = np.concatenate((w, w)).reshape(1, -1) * 0.5
        self.sigma = sigma; self.mu = mu; self.w = w;
def main():
    gmm = GMM_UBM();
    data = []
    data.append(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]))
    data.append(np.array([[3,3],[2,2],[4,4],[1,1]]))
    data.append(np.array([[5,3,3,1],[0,3,6,9],[3,5,7,9],[6,5,4,3]]))
    data.append(np.array([[4,5,6,7],[6,5,4,3],[3,4,5,6],[4,3,2,1]]))
    data.append(np.array([[1,1],[2,2],[2,2],[1,1]]))
    gmm.fitgmm(data, nmix=8, final_niter=10, ds_factor=1)
    print(gmm.sigma)
    print(gmm.mu)
    print(gmm.w)
if __name__ == '__main__':
    main()