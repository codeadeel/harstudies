# This file is responsible for the singular spectrum analysis class that joins its decomposition and sine fit methods
# %%
# Importing Libraries
from smartphoneExploration.ssaDecomposition import ssaDecomposition
from smartphoneExploration.ssaSineFit import ssaSineFit


# %%
# Singular Spectrum Analysis, With The Decomposition And Sine Fit Methods Inherited From Their Files
class singularSpectrumAnalysis(ssaDecomposition, ssaSineFit):
    def __init__(self, windowLength):
        """
        This class initializes the singular spectrum analysis with its embedding window length

        Arguments
        =========
        windowLength : Number of lagged values in each column of the trajectory matrix

        Output
        ======
        None
        """
        # Store The Embedding Window Length
        self.windowLength = windowLength
