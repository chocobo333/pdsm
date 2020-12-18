
import numpy as np
import sporco.util as util

import pdsm.pdsm as pdsm


def run():
    img = util.ExampleImages().image('kodim23.png', scaled=True, gray=True, idxexp=np.s_[160:416, 60:316])
    npd = 16
    fltlmbd = 10
    sl, sh = util.tikhonov_filter(img, fltlmbd, npd)
    D = util.convdicts()['G:12x12x36']

    pdsm.ConvBPDNL1L1(D, sh)
