from abc import ABCMeta, abstractmethod

import numpy as np

from .boundary_condition import BoundaryConditionABC


class OptimizerProxyABC(metaclass=ABCMeta):
    def create_optimizer(self, method):
        pass


class OptimizerProxy(OptimizerProxyABC):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer_map = {
            "SIMP_OC": SIMP_OC,
            "BESO_OC": BESO_OC,
        }

    def create_optimizer(self, method):
        return self.optimizer_map[method]


class OptimizerABC(metaclass=ABCMeta):
    def __init__(
        self, boundary_condition: BoundaryConditionABC, volfrax, symmetry
    ) -> None:
        self.volfrac = volfrax
        self.nelx = boundary_condition.nelx
        self.nely = boundary_condition.nely
        self.symmetry = symmetry  # xflip
        self.passive = boundary_condition.passive

    def symmetry_transform(self, x):
        if self.symmetry is None:
            return x
        x = x.reshape(self.nelx, self.nely)
        xflip = np.flip(x, self.symmetry["dim"])
        if self.symmetry["method"] == "max":
            xnew = np.where(x > xflip, x, xflip)
            return xnew.flatten()
        elif self.symmetry["method"] == "mean":
            xnew = (x + xflip) / 2.0
            return xnew.flatten()
        else:
            raise KeyError(f"symmetry method {self.symmetry['method']} not supported")

    def passive_transform(self, x):
        if self.passive is None:
            return x
        else:
            xnew = x.copy()
            xnew[self.passive > 0.5] = 1
            xnew[self.passive < 0.5] = 0
            return xnew

    @abstractmethod
    def update_x(self, x, dc, dv):
        pass


class SIMP_OC(OptimizerABC):
    def __init__(self, boundary_condition, volfrac, symmetry) -> None:
        super().__init__(boundary_condition, volfrac, symmetry)
        self.l1 = 0
        self.l2 = 1e9
        self.move = 0.2

    def update_x(self, x, dc, dv):
        l1 = self.l1
        l2 = self.l2
        # reshape to perform vector operations
        xnew = np.zeros(self.nelx * self.nely)
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew[:] = np.maximum(
                0.0,
                np.maximum(
                    x - self.move,
                    np.minimum(
                        1.0, np.minimum(x + self.move, x * np.sqrt(-dc / dv / lmid))
                    ),
                ),
            )
            xnew = self.symmetry_transform(xnew)
            xnew = self.passive_transform(xnew)
            volfrac_new = np.sum(xnew) / (self.nelx * self.nely)
            if volfrac_new > self.volfrac:
                l1 = lmid
            else:
                l2 = lmid
        return xnew, volfrac_new


class BESO_OC(OptimizerABC):
    def __init__(self, boundary_condition, volfrac, symmetry) -> None:
        super().__init__(boundary_condition, volfrac, symmetry)

    def update_x(self, x, dc, dv):
        l1 = min(dc)
        l2 = max(dc)
        while (l2 - l1) / l2 > 1e-5:
            lmid = 0.5 * (l2 + l1)
            xnew = np.maximum(0.001, np.sign(dc - lmid))
            xnew = self.symmetry_transform(xnew)
            xnew = self.passive_transform(xnew)
            volfrac_new = np.sum(xnew) / (self.nelx * self.nely)
            if volfrac_new > self.volfrac:
                l1 = lmid
            else:
                l2 = lmid
        # (long)worse
        # xnew = 0.5 * x + 0.5 * xnew
        return xnew, volfrac_new
