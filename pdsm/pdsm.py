
from __future__ import annotations

import numpy as np
# import numpy.typing as npt
import sporco.cnvrep as cr
import sporco.util as util
import sporco as sp
import sporco.common as sc
import sporco.linalg as sl
import sporco.prox as spr


from dataclasses import dataclass

from typing import Any, Union, Literal, Optional


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
        tau: float = 1
        sigma: float = 1
        rho: float = 1

    timer: util.Timer
    opt: Pdsm.Option

    dtype: np.dytpe

    D: np.ndarray
    """Dictionaries"""
    _X: np.ndarray
    """Coefficients maps"""
    S: np.ndarray
    """Signal images"""
    Y: np.ndarray
    """Dual variable"""

    def __new__(cls, *args: Any, **kwargs: Any) -> Pdsm:
        """Create an Pdsm object and start its initialisation timer."""

        instance = super(Pdsm, cls).__new__(cls)
        instance.timer = util.Timer(['init', 'solve', 'solve_wo_func', 'solve_wo_rsdl'])
        instance.timer.start('init')
        return instance

    def __init__(self, X: Optional[np.ndarray] = None):
        if X is None:
            self._X = self.X_prev = self._xinit()
        else:
            self._X = self.X_prev = X

    def _xinit(self) -> np.ndarray:
        """Initialize primal variable `X`"""
        pass

    def _yinit(self) -> np.ndarray:
        """Initialize primal variable `Y`"""
        pass

    def step(self):
        r"""Evaluate one step of below iteration

        .. math::
            \tilde{\mathbf{x}}_{n+1} &= \prox_{\tau G}(\mathbf{x}_n - \tau\nabla F(\mathbf{x}_n) - \tau L^{\dagger}\mathbf{y}_n)\\
            \tilde{\mathbf{y}}_{n+1} &= \prox_{\sigma H^{*}}(\mathbf{y}_n - \sigma L(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n)\\
            (\mathbf{x}_{n+1}, \mathbf{y}_{n+1}) &= \rho_n(\tilde{\mathbf{x}}_{n+1}, \tilde{\mathbf{y}}_{n+1}) + (1 - \rho_n)(\mathbf{x}_{n}, \mathbf{y}_{n})

        """
        self._primal_step()
        self._dual_step()

    def _primal_step(self):
        r"""

        .. math::
            \tilde{\mathbf{x}}_{n+1} = \prox_{\tau G}(\mathbf{x}_n - \tau\nabla F(\mathbf{x}_n) - \tau L^{\dagger}\mathbf{y}_n)
        """
        pass

    def _dual_step(self):
        r"""

        .. math::
            \tilde{\mathbf{y}}_{n+1} = \prox_{\sigma H^{*}}(\mathbf{y}_n - \sigma L(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n)
        """
        pass

    @property
    def X(self) -> np.ndarray:
        return self._X

    @X.setter
    def X(self, value):
        self.X_prev = self._X
        self._X = value


class GenericConvBPDN(Pdsm):

    @dataclass
    class Option(Pdsm.Option):
        pass

    Df: np.ndarray
    """`D` in FFT domain"""
    _Xf: np.ndarray
    """`X` in FFT domain"""
    Sf: np.ndarray
    """`S` in FFT domain"""
    Yf: np.ndarray
    """`Y` in FFT domain"""

    def __init__(self, D: np.ndarray, S: np.ndarray, X: Optional[np.ndarray], dimK: Union[Literal[1, 2], None, int] = None, dimN: int = 2):
        self.dtype = np.float

        # Infer problem dimensions and set relevant attributes of self
        self.cri = cr.CSC_ConvRepIndexing(D, S, dimK=dimK, dimN=dimN)

        # Reshape D and S to standard layout
        self.D = np.asarray(D.reshape(self.cri.shpD), dtype=self.dtype)
        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)

        # Compute signal in DFT domain
        self.Sf = sl.rfftn(self.S, None, self.cri.axisN)
        self.setdict()

        super().__init__(X=X)
        self._Xf = self.Xf_prev = sl.rfftn(self.X, self.cri.Nv, self.cri.axisN)

    def reconstruct(self) -> np.ndarray:
        return sl.irfftn(self.Df * self.Xf, self.cri.Nv, self.cri.axisN)

    def setdict(self, D: Optional[np.ndarray] = None):
        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = sl.rfftn(self.D, self.cri.Nv, self.cri.axisN)
        # Compute D^H S
        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)

    def _xinit(self) -> np.ndarray:
        """Initialize primal variable `X`"""
        Xf = np.conj(self.Df) / sl.inner(self.Df, np.conj(self.Df)) * self.Sf
        return sl.irfftn(Xf, self.cri.Nv, self.cri.axisN)

    @property
    def X(self):
        """Getter of `X`"""
        return super().X

    @X.setter
    def X(self, value):
        """Update `X`, `X_prev` and `Xf`"""
        super(GenericConvBPDN, type(self)).X.fset(self, value)
        self.Xf_prev = self.Xf
        self._Xf = sl.rfftn(self.X, self.cri.Nv, self.cri.axisN)

    @property
    def Xf(self) -> np.ndarray:
        """Getter of `Xf`"""
        return self._Xf

    @Xf.setter
    def Xf(self, value):
        """Update `X`, `X_prev` and `Xf`"""
        self.Xf_prev = self._Xf
        self._Xf = value
        self.X_prev = self.X
        self._X = sl.irfftn(self.Xf, self.cri.Nv, self.cri.axisN)


class ConvBPDNL1(GenericConvBPDN):
    @dataclass
    class Option(GenericConvBPDN.Option):
        h = None


class ConvBPDNL1L1(GenericConvBPDN):
    r"""

    Solve the optimization problem

    .. math::
        \argmin_{\mathbf{x}} \| \mathbf{D}\mathbf{x} - \mathbf{s} \|_{1} + \lambda\| \mathbf{x} \|_{1}
    """

    @dataclass
    class Option(Pdsm.Option):
        lambda_: float = 0.2

    cri: cr.CSC_ConvRepIndexing
    opt: ConvBPDNL1L1.Option

    def __init__(self, D: np.ndarray, S: np.ndarray, X: Optional[np.ndarray] = None, opt: Optional[ConvBPDNL1L1.Option] = None, dimK: Union[Literal[1, 2], None, int] = None, dimN: int = 2) -> None:
        self.opt = ConvBPDNL1L1.Option() if opt is None else opt

        # Initialize `X` and `Xf`
        super().__init__(D, S, X=X)
        print(np.sum(self.X[self.X != 0]))

        # Initialize `Y` and `Yf`
        self.Y = self._yinit()
        self.Yf = sl.rfftn(self.Y, self.cri.Nv, self.cri.axisN)

        self.step()

    def _yinit(self) -> np.ndarray:
        """Initialize dual variable `Y` with zeros (adhocly)"""
        return np.zeros(self.cri.shpX)

    def _primal_step(self):
        r"""

        .. math::
            \tilde{\mathbf{x}}_{n+1}=  &\prox_{\tau\lambda \|\bullet\|_{1}}(\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n)\\
            &= \mathcal{S}_{\tau\lambda}(\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n)\\
            &= \mathrm{sign}(\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n) \odot \max(0, |\mathbf{x}_n - \tau \mathbf{D}^{\dagger}\mathbf{y}_n| - \tau\lambda)
        """
        DYf: np.ndarray = np.conj(self.Df) * self.Yf
        tDYf: np.ndarray = self.opt.tau * DYf
        XtDYf: np.ndarray = self.Xf - tDYf
        XtDY: np.ndarray = sl.irfftn(XtDYf, self.cri.Nv, self.cri.axisN)
        self.X = spr.prox_l1(XtDY, self.opt.tau * self.opt.lambda_)

    def _dual_step(self):
        r"""

        .. math::
            \tilde{\mathbf{y}}_{n+1} &= \prox_{\sigma\lambda \|\bullet\|_{1}^{*}}(\mathbf{y}_n - \sigma \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n))\\
            &= \mathbf{y}_n - \sigma \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n) - \sigma\prox_{\frac{1}{\sigma\lambda}\|\bullet\|_{1}}(\frac{\mathbf{y}_n}{\sigma\lambda} - \frac{1}{\lambda}\mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n))\\
            &= \mathbf{y}_n - \sigma \mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n) - \sigma\mathcal{S}_{\frac{1}{\sigma\lambda}}(\frac{\mathbf{y}_n}{\sigma\lambda} - \frac{1}{\lambda}\mathbf{D}(2\tilde{\mathbf{x}}_{n+1} - \mathbf{x}_n))\\
        """
        pass
