# This file is responsible for the hidden Markov model of Attal et al. ( 2015 ) whose variances are kept above a floor after every update
# %%
# Importing Libraries
import numpy as np
from hmmlearn.hmm import GMMHMM


# %%
# Floored Mixture HMM
class flooredGmmHmm(GMMHMM):
    def runFlooredMstep(self, stats):
        """
        This method runs the M-step of hmmlearn's GMMHMM and then floors every covariance at min_covar, which hmmlearn applies only at initialisation

        Arguments
        =========
        stats : Sufficient statistics collected in the E-step

        Output
        ======
        None
        """
        # Update The Parameters, Then Keep Every Variance Above The Floor
        super()._do_mstep(stats)
        self.covars_ = np.maximum(self.covars_, self.min_covar)


# %%
# Hook Registration
# hmmlearn calls its M-step through this fixed name, so the floored step is registered under it
flooredGmmHmm._do_mstep = flooredGmmHmm.runFlooredMstep
