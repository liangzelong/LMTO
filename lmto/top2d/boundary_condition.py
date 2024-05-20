from abc import ABCMeta, abstractmethod

import numpy as np


class BoundaryCondituonProxyABC(metaclass=ABCMeta):
    def create_bc(self):
        pass


class BoundaryCondituonProxy(BoundaryCondituonProxyABC):
    def __init__(self) -> None:
        super().__init__()
        self.bc_map = {
            "BridgeForceTop": BridgeForceTop,
            "BridgeForceMid": BridgeForceMid,
        }

    def create_bc(self, bc_name):
        return self.bc_map[bc_name]


class BoundaryConditionABC(metaclass=ABCMeta):
    def __init__(
        self,
        nelx: int,
        nely: int,
    ):
        self.nelx = nelx
        self.nely = nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.fixed = self.get_fixed()
        self.force = self.get_force()
        self.passive = self.get_passive()

    @abstractmethod
    def get_fixed(self):
        pass

    @abstractmethod
    def get_force(self):
        pass

    @abstractmethod
    def get_passive(self):
        pass


class BridgeForceTop(BoundaryConditionABC):
    def __init__(self, nelx, nely, **kwargs):
        super().__init__(nelx, nely)

    def get_fixed(self):
        fixed = np.array(
            [
                2 * int((self.nelx + 1) / 4) * (self.nely + 1) - 1,
                2 * int((self.nelx + 1) / 4) * (self.nely + 1) - 2,
                2 * int((self.nelx + 1) * 3 / 4) * (self.nely + 1) - 1,
                2 * int((self.nelx + 1) * 3 / 4) * (self.nely + 1) - 2,
            ]
        )
        return fixed

    def get_force(self):
        f = np.zeros((self.ndof, 1))
        f[1 :: 2 * (self.nely + 1), 0] = -1
        return f

    def get_passive(self):
        """_summary_
        ndarray:(nelx*nely),if pix==0 (undesigned) pix==1(designed) pix==0.5(remained)
        Returns:
            _type_: _description_
        """
        passive = np.ones((self.nelx, self.nely)) * 0.5
        passive[:, 0] = 1.0
        return passive.flatten()

class BridgeForceMid(BoundaryConditionABC):
    def __init__(self, nelx, nely, **kwargs):
        super().__init__(nelx, nely)

    def get_fixed(self):
        fixed = np.array(
            [
                2 * int((self.nelx + 1) / 8) * (self.nely + 1) - 1,
                2 * int((self.nelx + 1) / 8) * (self.nely + 1) - 2,
                2 * int((self.nelx + 1) * 7 / 8) * (self.nely + 1) - 1,
                2 * int((self.nelx + 1) * 7 / 8) * (self.nely + 1) - 2,
            ]
        )
        return fixed

    def get_force(self):
        f = np.zeros((self.ndof, 1))
        f[1+2*((self.nely + 1)//2):: 2 * (self.nely + 1), 0] = -1
        return f

    def get_passive(self):
        """_summary_
        ndarray:(nelx*nely),if pix==0 (undesigned) pix==1(designed) pix==0.5(remained)
        Returns:
            _type_: _description_
        """
        passive = np.ones((self.nelx, self.nely)) * 0.5
        passive[:, (self.nely + 1)//2-1:(self.nely + 1)//2+1] = 1.0
        return passive.flatten()