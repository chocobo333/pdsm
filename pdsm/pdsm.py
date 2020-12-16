
from sporco import common
from sporco import admm
from sporco.admm import cbpdn

from dataclasses import dataclass


class Pdsm(common.IterativeSolver):
    r"""Base class for Primal-Dual Splitting Method (PDSM).

    Solve an optimisation problem of the form

    .. math::
       \mathrm{argmin}_{\mathbf{x},\mathbf{y}} \;
       f(\mathbf{x}) + g(\mathbf{y}) \;\mathrm{such\;that}\;
       A\mathbf{x} + B\mathbf{y} = \mathbf{c} \;\;.

    This class is intended to be a base class of other classes that
    specialise to specific optimisation problems.
    """

    @dataclass
    class Option:
        rho: int

class GenericConvBPDN(Pdsm):

    @dataclass
    class Option(Pdsm.Option):
        pass

# class ConvBPDNL1L1
