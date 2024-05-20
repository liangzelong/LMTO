from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import coo_matrix


class FilterABC(metaclass=ABCMeta):
    def __init__(self, nelx, nely, rmin) -> None:
        self.nelx = nelx
        self.nely = nely
        nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * nely + l
                        fac = rmin - np.sqrt(((i - k) * (i - k) + (j - l) * (j - l)))
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc = cc + 1
        self.H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
        self.Hs = self.H.sum(1)

    @abstractmethod
    def filter_dc(self):
        pass

    @abstractmethod
    def filter_x(self):
        pass


class DensityFilter(FilterABC):
    def __init__(self, nelx, nely, rmin, **kwargs) -> None:
        super().__init__(nelx, nely, rmin)

    def filter_dc(self, dc, dv, x):
        dc[:] = np.asarray(self.H * (dc[np.newaxis].T / self.Hs))[:, 0]
        dv[:] = np.asarray(self.H * (dv[np.newaxis].T / self.Hs))[:, 0]
        return dc, dv

    def filter_x(self, x):
        x[:] = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]
        return x


class SensivityFilter(FilterABC):
    def __init__(self, nelx, nely, rmin, **kwargs) -> None:
        super().__init__(nelx, nely, rmin)

    def filter_dc(self, dc, dv, x):
        dc[:] = np.asarray((self.H * (x * dc))[np.newaxis].T / self.Hs)[
            :, 0
        ] / np.maximum(0.001, x)

        return dc, dv

    def filter_x(self, x):
        return x


class BESOFilter(FilterABC):
    def __init__(self, nelx, nely, rmin, **kwargs) -> None:
        super().__init__(nelx, nely, rmin)

    def filter_dc(self, dc, dv, x):
        dc[:] = np.asarray(self.H * (dc[np.newaxis].T / self.Hs))[:, 0]
        return dc, dv

    def filter_x(self, x):
        return x


class FilterProxyABC(metaclass=ABCMeta):
    @abstractmethod
    def create_filter(self):
        pass


class FilterProxy(FilterProxyABC):
    def __init__(
        self,
    ):
        self.filter_map = {
            "sensivity": SensivityFilter,
            "density": DensityFilter,
            "conv": BESOFilter,
        }

    def create_filter(self, filter_method) -> FilterABC:
        return self.filter_map[filter_method]
