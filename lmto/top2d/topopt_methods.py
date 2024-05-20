from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .boundary_condition import BoundaryCondituonProxy
from .FEA import FEA88
from .filter import FilterProxy
from .optimizer import OptimizerProxy
from .stiffness_matrix import lk


class TopOptProxyABC(metaclass=ABCMeta):
    @abstractmethod
    def create_topopt(self, method):
        pass


class TopOptProxy(TopOptProxyABC):
    def __init__(
        self,
    ) -> None:
        self.topopt_map = {
            "SIMP": SIMP,
            "BESO": BESO,
        }

    def create_topopt(self, method):
        return self.topopt_map[method]


class TopOptABC(metaclass=ABCMeta):
    def __init__(
        self, nelx, nely, volfrac, bc, ft, opt_method, symmetry, rmin, E=1, nu=0.3
    ):
        self.nelx = nelx
        self.nely = nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.KE = lk(E, nu)
        self.bc = BoundaryCondituonProxy().create_bc(bc)(nelx, nely)
        self.filter = FilterProxy().create_filter(ft)(nelx, nely, rmin)
        self.optimizer = OptimizerProxy().create_optimizer(opt_method)(
            self.bc, volfrac, symmetry
        )
        self.history = []

    @abstractmethod
    def compute_cv(self):
        pass

    @abstractmethod
    def compute_dcdv(self, x, c):
        pass

    @abstractmethod
    def update_x(self, x):
        pass

    @abstractmethod
    def if_meet_criterion(self):
        pass

    # def update_x(self, x):
    #     c, v, ce = self.compute_cv(x)
    #     dc, dv = self.compute_dcdv(x, ce)
    #     dc, dv = self.filter.filter_dc(dc, dv, x)
    #     xnew = self.optimize_x(x, dc, dv)
    #     xnew = self.filter.filter_x(xnew)
    #     return xnew, c, v


class SIMP(TopOptABC):
    def __init__(
        self,
        nelx,
        nely,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        ft="sensivity",
        bc="BridgeForceTop",
        opt_method="BESO_OC",
        symmetry={},
    ) -> None:
        super().__init__(nelx, nely, volfrac, bc, ft, opt_method, symmetry, rmin)
        self.Emin = 1e-9
        self.Emax = 1.0
        self.penal = penal
        self.FEA = FEA88(self.bc)
        self.x = np.ones(nelx * nely)
        self.xnew = np.zeros(nelx * nely)
        self.edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array(
                    [
                        2 * n1 + 2,
                        2 * n1 + 3,
                        2 * n2 + 2,
                        2 * n2 + 3,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n1,
                        2 * n1 + 1,
                    ]
                )

    def compute_K(self, x):
        iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()
        sK = (
            (self.KE.flatten()[np.newaxis]).T
            * (self.Emin + x**self.penal * (self.Emax - self.Emin))
        ).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()
        return K

    def compute_cv(self, x):
        K = self.compute_K(x)
        u = self.FEA.solve(K)
        ce = np.ones((self.nely * self.nelx))
        ce[:] = (
            np.dot(u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)
        c = ((self.Emin + x**self.penal * (self.Emax - self.Emin)) * ce).sum()
        v = x.sum() / (self.nelx * self.nely)
        self.c = c
        self.v = v
        return ce

    def compute_dcdv(self, x, ce):
        dc = (-self.penal * x ** (self.penal - 1) * (self.Emax - self.Emin)) * ce
        dv = np.ones(self.nely * self.nelx)
        return dc, dv

    def update_x(self, iter, x):
        self.x = x
        ce = self.compute_cv(x)
        dc, dv = self.compute_dcdv(x, ce)
        dc, dv = self.filter.filter_dc(dc, dv, x)
        xnew, _ = self.optimizer.update_x(x, dc, dv)
        self.xnew = self.filter.filter_x(xnew)
        return self.xnew

    def if_meet_criterion(self, iter):
        self.change = np.linalg.norm(self.x - self.xnew, np.inf)
        if (self.change > 0.001) and (iter < 2000):
            return True
        else:
            return False


class BESO(TopOptABC):
    def __init__(
        self,
        nelx,
        nely,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        ft="sensivity",
        bc="BridgeForceTop",
        opt_method="BESO_OC",
        symmetry={},
    ) -> None:
        super().__init__(nelx, nely, volfrac, bc, ft, opt_method, symmetry, rmin)
        self.er = 0.02
        self.M = 5
        self.volstep = 1.0
        self.penal = penal
        self.volfrac = volfrac
        self.FEA = FEA88(self.bc)
        self.x = np.ones(nelx * nely)
        self.xnew = np.zeros(nelx * nely)
        self.edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array(
                    [
                        2 * n1 + 2,
                        2 * n1 + 3,
                        2 * n2 + 2,
                        2 * n2 + 3,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n1,
                        2 * n1 + 1,
                    ]
                )

    def compute_K(self, x):
        iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()
        sK = ((self.KE.flatten()[np.newaxis]).T * x).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()
        return K

    def compute_cv(self, x):
        K = self.compute_K(x)
        u = self.FEA.solve(K)
        ce = np.ones((self.nely * self.nelx))
        ce[:] = (
            np.dot(u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)
        c = (0.5 * x**self.penal * ce).sum()
        v = x.sum() / (self.nelx * self.nely)
        self.c = c
        self.history.append(c)
        self.v = v
        return ce

    def compute_dcdv(self, x, ce):
        dc = (0.5 * x ** (self.penal - 1)) * ce
        dv = np.ones(self.nely * self.nelx)
        return dc, dv

    def update_x(self, iter, x):
        if iter > 1:
            self.olddc = self.dc
        self.x = x
        ce = self.compute_cv(x)
        dc, dv = self.compute_dcdv(x, ce)
        dc, dv = self.filter.filter_dc(dc, dv, x)
        if iter > 1:
            dc = (self.olddc + dc) / 2
        self.dc = dc

        self.volstep = max(self.volstep * (1 - self.er), self.volfrac)
        self.optimizer.volfrac = self.volstep
        xnew, _ = self.optimizer.update_x(x, dc, dv)
        self.xnew = self.filter.filter_x(xnew)
        return self.xnew

    def if_meet_criterion(self, iter):
        if iter > self.M * 2:
            self.change = (
                np.abs(
                    np.array(self.history[iter - self.M * 2 : iter - self.M]).sum()
                    - np.array(self.history[iter - self.M : iter]).sum()
                )
                / np.array(self.history[iter - self.M : iter]).sum()
            )
        else:
            self.change = 1.0
        if (self.change > 0.001) and (iter < 2000):
            return True
        else:
            return False


class LMTO(TopOptABC):
    def __init__(
        self,
        nelx,
        nely,
        volfrac=0.5,
        penal=3.0,
        rmin=1.5,
        ft="sensivity",
        bc="BridgeForceTop",
        opt_method="BESO_OC",
        symmetry={},
    ) -> None:
        super().__init__(nelx, nely, volfrac, bc, ft, opt_method, symmetry, rmin)
        self.er = 0.02
        self.M = 5
        self.volstep = 1.0
        self.penal = penal
        self.volfrac = volfrac
        self.FEA = FEA88(self.bc)
        self.x = np.ones(nelx * nely)
        self.xnew = np.zeros(nelx * nely)
        self.edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                self.edofMat[el, :] = np.array(
                    [
                        2 * n1 + 2,
                        2 * n1 + 3,
                        2 * n2 + 2,
                        2 * n2 + 3,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n1,
                        2 * n1 + 1,
                    ]
                )

    def compute_K(self, x):
        iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()
        sK = ((self.KE.flatten()[np.newaxis]).T * x).flatten(order="F")
        K = coo_matrix((sK, (iK, jK)), shape=(self.ndof, self.ndof)).tocsc()
        return K

    def compute_cv(self, x):
        K = self.compute_K(x)
        u = self.FEA.solve(K)
        ce = np.ones((self.nely * self.nelx))
        ce[:] = (
            np.dot(u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE)
            * u[self.edofMat].reshape(self.nelx * self.nely, 8)
        ).sum(1)
        c = (0.5 * x**self.penal * ce).sum()
        v = x.sum() / (self.nelx * self.nely)
        self.c = c
        self.history.append(c)
        self.v = v
        return ce

    def compute_dcdv(self, x, ce, alpha, sdf):
        dc = (0.5 * x ** (self.penal - 1)) * ce
        # dc = (dc - dc.min()) / (dc.max() - dc.min())
        # dc = alpha * dc + (1 - alpha) * sdf
        dv = np.ones(self.nely * self.nelx)
        return dc, dv

    def update_x(self, iter, x, alpha, sdf):
        if iter > 1:
            self.olddc = self.dc
        self.x = x
        ce = self.compute_cv(x)
        dc, dv = self.compute_dcdv(x, ce, alpha, sdf)
        dc, dv = self.filter.filter_dc(dc, dv, x)
        # dc = (dc - dc.min()) / (dc.max() - dc.min())
        
        if iter > 1:
            dc = (self.olddc + dc) / 2
        # dc = (dc - dc.min()) / (dc.max() - dc.min())
        self.dc = dc
        
        dc = (dc - dc.min()) / (dc.max() - dc.min())
        dc = alpha * dc + (1 - alpha) * sdf
        
        self.volstep = max(self.volstep * (1 - self.er), self.volfrac)
        self.optimizer.volfrac = self.volstep
        xnew, _ = self.optimizer.update_x(x, dc, dv)
        self.xnew = self.filter.filter_x(xnew)
        return self.xnew

    def if_meet_criterion(self, iter):
        if iter > self.M * 2:
            self.change = (
                np.abs(
                    np.array(self.history[iter - self.M * 2 : iter - self.M]).sum()
                    - np.array(self.history[iter - self.M : iter]).sum()
                )
                / np.array(self.history[iter - self.M : iter]).sum()
            )
        else:
            self.change = 1.0
        if (self.change > 0.001) and (iter < 2000):
            return True
        else:
            return False
