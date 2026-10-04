# This file is responsible for turning the raw acceleration windows of one bout into continuous signals for the cadence estimators
# %%
# Importing Libraries
import numpy as np


# %%
# Cadence Signals
class cadenceSignals:
    def stitchBout(self, boutWindows, windowStepSamples):
        """
        This method rebuilds the continuous raw signal of one bout from its overlapping windows

        Arguments
        =========
        boutWindows : Consecutive windows of one bout, shape ( windows , samples per window )
        windowStepSamples : Number of new samples each window adds to the previous one

        Output
        ======
        Continuous one dimensional signal of the bout
        """
        # Keep The First Window And Append The New Samples Of Every Following Window
        return np.concatenate([boutWindows[0]] + [boutWindow[-windowStepSamples:] for boutWindow in boutWindows[1:]])

    def computeMagnitude(self, axisSignals):
        """
        This method returns the mean removed euclidean norm of the acceleration axes

        Arguments
        =========
        axisSignals : Sequence of equally long axis signals

        Output
        ======
        Magnitude signal with its mean removed
        """
        # Combine The Axes And Remove The Mean
        magnitude = np.sqrt(np.sum(np.square(axisSignals), axis=0))
        return magnitude - magnitude.mean()

    def computeVerticalAcceleration(self, bodyAxisSignals, totalAxisSignals):
        """
        This method projects the body acceleration onto the gravity direction ( total minus body acceleration ) sample by sample

        Arguments
        =========
        bodyAxisSignals : Sequence of the body acceleration axis signals
        totalAxisSignals : Sequence of the total acceleration axis signals of the same samples

        Output
        ======
        Vertical body acceleration with its mean removed
        """
        # Estimate The Gravity Direction
        bodyAcceleration = np.asarray(bodyAxisSignals)
        gravityAcceleration = np.asarray(totalAxisSignals) - bodyAcceleration
        gravityDirection = gravityAcceleration / np.linalg.norm(gravityAcceleration, axis=0)

        # Project The Body Acceleration And Remove The Mean
        verticalAcceleration = np.sum(bodyAcceleration * gravityDirection, axis=0)
        return verticalAcceleration - verticalAcceleration.mean()
