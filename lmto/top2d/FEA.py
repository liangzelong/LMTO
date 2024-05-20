from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse.linalg import spsolve

from .boundary_condition import BoundaryConditionABC, BoundaryCondituonProxy


class baseFEA(metaclass=ABCMeta):
    def __init__(self, bc) -> None:
        self.nelx = bc.nelx
        self.nely = bc.nely
        self.ndof = bc.ndof
        self.fixeddofs = bc.fixed
        self.freedofs = self.get_freedofs()
        self.F = bc.force
        self.U = np.zeros((self.ndof, 1))

    def get_freedofs(
        self,
    ):
        alldofs = np.arange(2 * (self.nelx + 1) * (self.nely + 1))
        return np.setdiff1d(alldofs, self.fixeddofs)

    @abstractmethod
    def solve(self):
        pass


class FEA88(baseFEA):
    def __init__(self, bc):
        super().__init__(bc)

    def solve(self, K):
        self.U[self.freedofs, 0] = spsolve(
            K[self.freedofs, :][:, self.freedofs], self.F[self.freedofs, 0]
        )
        return self.U
