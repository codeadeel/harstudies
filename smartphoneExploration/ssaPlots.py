# This file is responsible for plotting the singular spectrum decomposition and the sine fit of one participant
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np

from common.plotDefaults import saveFigure


# %%
# SSA Plots
class ssaPlots:
    def plotSsaDecomposition(self, participantDetails):
        """
        This method draws the walking window means, their SSA components, the sine fit and the singular value shares

        Arguments
        =========
        participantDetails : Series, components, singular value shares, sine fit, model and pair correlation of the notebook participant

        Output
        ======
        None
        """
        # Prepare The Spacing And The Component Numbers
        windowStepSeconds = self.loader.windowStepSeconds
        trendIndex = self.ssaTrendIndex
        firstIndex, secondIndex = self.ssaPairIndices

        # Plot The Notebook Participant's Decomposition And Fit
        timeSeconds, meanNorm, reconstructedComponents, singularShares, sineFit, ssaModel, pairCorrelation = participantDetails
        figure, (seriesAxis, fitAxis, shareAxis) = plt.subplots(3, 1, figsize=(10, 11))
        seriesAxis.plot(timeSeconds, meanNorm, label="norm of tBodyAcc-mean XYZ")
        seriesAxis.plot(timeSeconds, reconstructedComponents[trendIndex], label=f"SSA component {trendIndex + 1} ( trend )")
        seriesAxis.set_title(f"Participant {self.ssaParticipant}: walking window means, one value every {windowStepSeconds} s")
        seriesAxis.set_xlabel("Seconds ( concatenated walking windows )")
        seriesAxis.legend()
        fittedWave = ssaModel.sineWave(timeSeconds, sineFit["amplitude"], sineFit["frequency"], sineFit["phase"], sineFit["offset"])
        fitAxis.plot(timeSeconds, reconstructedComponents[firstIndex] + reconstructedComponents[secondIndex], label=f"SSA components {firstIndex + 1} + {secondIndex + 1}")
        fitAxis.plot(timeSeconds, fittedWave, linestyle="--", label=f"sine fit, {sineFit['frequency']:.4f} Hz")
        fitAxis.set_xlabel("Seconds ( concatenated walking windows )")
        fitAxis.set_title(f"SSA components {firstIndex + 1} + {secondIndex + 1} ( w-correlation {pairCorrelation:.2f} ) and their sine fit")
        fitAxis.legend()
        shownComponents = min(10, len(singularShares))
        shareAxis.bar(np.arange(1, shownComponents + 1), singularShares[:shownComponents])
        shareAxis.set_yscale("log")
        shareAxis.set_xlabel("SSA component")
        shareAxis.set_ylabel("Share of squared singular values")
        shareAxis.set_title(f"Singular value shares ( component {trendIndex + 1} carries the series mean level )")
        figure.tight_layout()
        saveFigure(figure, self.figuresDirectory / "ssaWalkingFrequency.png")
