# This file is responsible for estimating the walking frequency of every participant with singular spectrum analysis on the window means
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv
from smartphoneExploration.singularSpectrumAnalysis import singularSpectrumAnalysis
from smartphoneExploration.ssaPlots import ssaPlots


# %%
# SSA Sections, With The Plot Method Inherited From ssaPlots
class ssaSections(ssaPlots):
    def estimateWalkingFrequency(self):
        """
        This method applies singular spectrum analysis and a sine fit to the walking window means of every participant ( section 6 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the SSA step frequency and the pairing check per participant
        """
        # Prepare The Walking Rows
        walkingFrame = self.featureFrame[self.featureFrame["Activity"] == self.walkingActivity]

        # Decompose The Mean Norm Series Of Every Participant
        ssaRows = []
        participantDetails = None
        for subject in sorted(walkingFrame["subject"].unique().tolist()):
            ssaRow, subjectDetails = self.decomposeParticipant(walkingFrame, subject)
            ssaRows.append(ssaRow)

            # Keep The Series Of The Notebook's Participant For The Figure
            if subject == self.ssaParticipant:
                participantDetails = subjectDetails
        self.ssaTable = pd.DataFrame(ssaRows)
        writeCsv(self.ssaTable, self.resultsDirectory / "ssaWalkingFrequency.csv")

        # Record The SSA Settings
        self.recordSsaSettings()

        # Plot The Notebook Participant's Decomposition And Fit
        self.plotSsaDecomposition(participantDetails)
        return self.ssaTable

    def decomposeParticipant(self, walkingFrame, subject):
        """
        This method decomposes the walking window mean series of one participant and fits a sine to its oscillating pair

        Arguments
        =========
        walkingFrame : Rows of the walking windows
        subject : Participant number

        Output
        ======
        Tuple of the table row of the participant and the series, components and fit kept for the figure
        """
        # Prepare The Spacing, The Spacing Limit And The Component Numbers
        windowStepSeconds = self.loader.windowStepSeconds
        nyquistFrequency = 0.5 / windowStepSeconds
        trendIndex = self.ssaTrendIndex
        firstIndex, secondIndex = self.ssaPairIndices

        # Decompose The Mean Norm Series
        meanNorm = np.linalg.norm(walkingFrame.loc[walkingFrame["subject"] == subject, self.ssaMeanFeatures].to_numpy(), axis=1)
        timeSeconds = np.arange(len(meanNorm)) * windowStepSeconds
        ssaModel = singularSpectrumAnalysis(windowLength=len(meanNorm) // 2)
        reconstructedComponents, singularValues = ssaModel.decomposeSeries(meanNorm)

        # Fit A Sine To The Components Taken As The Oscillating Pair
        oscillation = reconstructedComponents[firstIndex] + reconstructedComponents[secondIndex]
        sineFit = ssaModel.fitSine(timeSeconds, oscillation, nyquistFrequency)

        # Check Whether Those Two Components Can Be Separated
        pairCorrelation = ssaModel.weightedCorrelation(reconstructedComponents[firstIndex], reconstructedComponents[secondIndex])

        # Collect The Participant Row
        singularShares = singularValues ** 2 / np.sum(singularValues ** 2)
        ssaRow = {
            "subject": subject,
            "walkingWindows": len(meanNorm),
            "ssaWindowLength": ssaModel.windowLength,
            "trendSingularShare": singularShares[trendIndex],
            "pairSingularShare": singularShares[firstIndex] + singularShares[secondIndex],
            "pairVarianceRatio": np.var(oscillation) / np.var(meanNorm),
            "pairWCorrelation": pairCorrelation,
            "wCorrelated": bool(pairCorrelation >= self.pairCorrelationThreshold),
            "firstComponentPeakHz": ssaModel.findPeriodogramPeak(reconstructedComponents[firstIndex], windowStepSeconds),
            "secondComponentPeakHz": ssaModel.findPeriodogramPeak(reconstructedComponents[secondIndex], windowStepSeconds),
            "periodogramStartHz": sineFit["startFrequency"],
            "ssaStepsPerSecond": sineFit["frequency"],
            "sineAmplitude": sineFit["amplitude"],
            "sineRSquared": sineFit["rSquared"],
            "nyquistHz": nyquistFrequency,
        }
        return ssaRow, (timeSeconds, meanNorm, reconstructedComponents, singularShares, sineFit, ssaModel, pairCorrelation)

    def recordSsaSettings(self):
        """
        This method records the SSA settings for runParameters.csv

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Prepare The Component Labels And The Spacing Limit
        trendIndex = self.ssaTrendIndex
        firstIndex, secondIndex = self.ssaPairIndices
        pairLabel = f"{firstIndex + 1} and {secondIndex + 1}"
        nyquistFrequency = 0.5 / self.loader.windowStepSeconds

        # Record The SSA Settings
        self.recordParameter("ssa", "participant", self.ssaParticipant)
        self.recordParameter("ssa", "series", "Euclidean norm of " + ", ".join(self.ssaMeanFeatures))
        self.recordParameter("ssa", "windowLengthRule", "half the series length, rounded down")
        self.recordParameter("ssa", "trendComponent", trendIndex + 1)
        self.recordParameter("ssa", "oscillationComponents", pairLabel)
        self.recordParameter("ssa", "pairCorrelationThreshold", self.pairCorrelationThreshold)
        self.recordParameter("ssa", "nyquistHz", nyquistFrequency)
