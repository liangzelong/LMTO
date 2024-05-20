import os
import sys
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lmto.top2d import LMTO, BridgeForceTop, SensivityFilter, init_one, init_load
from lmto.utils import generate_sdf_out, generate_sdf_singlepk, generate_sdf_doublepk


class resize_cash:
    def __init__(self):
        self.design_cash = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nelx": (
                    "INT",
                    {
                        "default": 200,
                        "min": 1,  # Minimum value
                        "max": 2000,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "nely": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,  # Minimum value
                        "max": 2000,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "inputs": ("IMAGE",),
                "cash_intput": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    # OUTPUT_NODE = False

    CATEGORY = "TopologyOptimization"

    def generate(
        self,
        nelx=400,
        nely=100,
        inputs=None,
        cash_intput=False,
    ):
        if (self.design_cash is not None) and cash_intput:
            pass
        else:
            outputs = torch.nn.functional.interpolate(
                inputs.permute(0, 3, 1, 2), (nely, nelx)
            ).permute(0, 2, 3, 1)
            self.design_cash = outputs.cpu().clone()
        return (self.design_cash,)


class LMTO_process:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nelx": (
                    "INT",
                    {
                        "default": 200,
                        "min": 1,  # Minimum value
                        "max": 2000,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "nely": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,  # Minimum value
                        "max": 2000,  # Maximum value
                        "step": 1,  # Slider's step
                        "display": "number",  # Cosmetic only: display as "number" or "slider"
                    },
                ),
                "volfrac": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "rmin": (
                    "FLOAT",
                    {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "bc": (["BridgeForceTop", "BridgeForceMid"],),
                "alpha": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "refpath": (
                    "STRING",
                    {
                        "multiline": False,
                    },
                ),
            },
            "optional": {"refimg": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "generate"

    # OUTPUT_NODE = False

    CATEGORY = "TopologyOptimization"

    def generate(
        self,
        nelx=400,
        nely=100,
        volfrac=0.3,
        rmin=1.5,
        bc="BridgeForceMid",
        alpha=1.0,
        refpath=None,
        refimg=None,
    ):
        penal = 3.0
        ft = "conv"
        opt_method = "BESO_OC"
        symmerty = {"dim": 0, "method": "max"}
        # symmerty=None

        topopt = LMTO(nelx, nely, volfrac, penal, rmin, ft, bc, opt_method, symmerty)
        x = init_one(nelx, nely)
        if refimg is not None:
            sdf_ref = refimg.numpy()[0].mean(-1).T
            sdf = generate_sdf_out(sdf_ref)
        elif len(refpath) > 3:
            sdf_ref = init_load(refpath, nelx, nely)
            sdf = generate_sdf_out(sdf_ref)
        else:
            sdf = np.zeros((nelx * nely))

        iter = 0
        if alpha == 10:
            sdf_weight = 1.0
        else:
            sdf_weight = 1 - 10 ** (-alpha)

        while topopt.if_meet_criterion(iter):
            iter += 1
            x = topopt.update_x(iter, x, sdf_weight, sdf)
            print(f"iter:{iter},c:{topopt.c},v:{topopt.v},change:{topopt.change}")

        self.design_cash = x
        imgtensor = (
            torch.tensor(1 - x.reshape(nelx, nely).T)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(1, 3, 1, 1)
        )


        return (imgtensor.permute(0, 2, 3, 1),)



NODE_CLASS_MAPPINGS = {"LMTO": LMTO_process, "resize_cash": resize_cash}
NODE_DISPLAY_NAME_MAPPINGS = {"LMTO": "LMTO", "resize_cash": "resize_cash"}
