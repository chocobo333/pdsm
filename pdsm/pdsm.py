
from __future__ import annotations

import numpy as np
# import numpy.typing as npt
import sporco.cnvrep as cr
import sporco.util as util
import sporco as sp
import sporco.common as sc


from dataclasses import dataclass

from typing import Any, Union, Literal


class Pdsm(sc.IterativeSolver):
    r"""Base class for Primal-Dual Splitting Method (PDSM).

    Solve an optimisation problem of the form

    .. math::
        \newcommand{\argmin}{\mathop{\rm arg~min}\limits}
        \newcommand{\prox}{\mathop{\mathrm{prox}}\nolimits}

        \argmin_{\mathbf{x}} F(\mathbf{x}) + G(\mathbf{x}) + H(\mathbf{L}\mathbf{x})

    This class is intended to be a base class of other classes that
    specialise to specific optimisation problems.
    """

    @dataclass
    class Option:
        rho_p: float
        rho_d: float
        theta: float

    timer: util.Timer
    opt: Pdsm.Option

    def __new__(cls, *args: Any, **kwargs: Any) -> Pdsm:
        """Create an Pdsm object and start its initialisation timer."""

        instance = super(Pdsm, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_func', 'solve_wo_rsdl'])
        instance.timer.start('init')
        return instance

    def step(self):
        r"""Evaluate one step of below iteration

        .. math::
            \tilde{\mathbf{x}}_{n+1} &= \prox_{\tau G}(\mathbf{x}_n - \tau\nabla F(\mathbf{x}_n) - \tau L^{\dagger}\mathbf{y}_n)\\
            \tilde{\mathbf{y}}_{n+1} &= \prox_{\sigma H^{*}}(\mathbf{y}_n - \sigma L(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n)\\
            (\mathbf{x}_{n+1}, \mathbf{y}_{n+1}) &= \rho_n(\tilde{\mathbf{x}}_{n+1}, \tilde{\mathbf{y}}_{n+1}) + (1 - \rho_n)(\mathbf{x}_{n}, \mathbf{y}_{n})

        """
        self._primal_step()
        self._dual_step()

    def primal_step(self):
        r"""

        .. math::
            \tilde{\mathbf{x}}_{n+1} = \prox_{\tau G}(\mathbf{x}_n - \tau\nabla F(\mathbf{x}_n) - \tau L^{\dagger}\mathbf{y}_n)
        """
        pass

    def dual_step(self):
        r"""

        .. math::
            \tilde{\mathbf{y}}_{n+1} = \prox_{\sigma H^{*}}(\mathbf{y}_n - \sigma L(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n)
        """
        pass


class GenericConvBPDN(Pdsm):

    @dataclass
    class Option(Pdsm.Option):
        pass


class ConvBPDNL1(GenericConvBPDN):
    @dataclass
    class Option(GenericConvBPDN.Option):
        h: None


class ConvBPDNL1L1(GenericConvBPDN):
    r"""

    Solve the optimization problem

    .. math::
        \argmin_{\mathbf{x}} \| \mathbf{D}\mathbf{x} - \mathbf{s} \|_{1} + \| \mathbf{x} \|_{1}
    """

    cri: cr.CSC_ConvRepIndexing

    @dataclass
    class Option(GenericConvBPDN.Option):
        pass

    def __init__(self, D: np.ndarray, S: np.ndarray, dimK: Union[Literal[1, 2], None, int] = None, dimN: int = 2) -> None:
        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

    def primal_step(self):
        r"""

        .. math::
            \tilde{\mathbf{x}}_{n+1} &= \prox_{\tau \|\bullet\|_{1}}(\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n)\\
            &= \mathcal{S}_{\tau}(\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n)\\
            &= \mathrm{sign}(\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n) \odot \max(0, |\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n| - \tau)
        """
        pass

    def dual_step(self):
        r"""

        .. math::
            \tilde{\mathbf{y}}_{n+1} &= \prox_{\sigma \|\bullet\|_{1}^{*}}(\mathbf{y}_n - \sigma \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n))\\
            &= \mathbf{y}_n - \sigma \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n) - \sigma\prox_{\frac{1}{\sigma}\|\bullet\|_{1}}(\frac{\mathbf{y}_n}{\sigma} - \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n))\\
            &= \mathbf{y}_n - \sigma \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n) - \sigma\mathcal{S}_{\frac{1}{\sigma}}(\frac{\mathbf{y}_n}{\sigma} - \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n))\\
        """
        pass
