
from sporco import common

from dataclasses import dataclass


class Cp(common.IterativeSolver):
    r"""
    # CP method
    """

    @dataclass
    class Option:
        rho: int

class GenericConvBPDN(Cp):

    @dataclass
    class Option(Cp.Option):
        pass

class ConvBPDN(GenericConvBPDN):

    @dataclass
    class Option(GenericConvBPDN.Option):
        pass
