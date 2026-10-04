# This file is responsible for the SONAR window class that reads the converted recordings, holds the window settings and cuts the recordings into windows
# %%
# Importing Libraries
from pathlib import Path

from common.csvWriters import readCsv
from windowSelectionPilot.windowFeatures import windowFeatures
from windowSelectionPilot.windowRecording import windowRecording
from windowSelectionPilot.windowScores import windowScores
from windowSelectionPilot.windowStreams import windowStreams


# %%
# SONAR Windows
class sonarWindowBuilder(windowStreams, windowFeatures, windowScores, windowRecording):
    def __init__(self, dataRoot, windowSeconds=2.0, gapFactor=1.5, bandEdges=(0.5, 1.0, 2.0, 4.0, 8.0, 16.0)):
        """
        This class initializes the window class with the converted SONAR data, the window length, the gap rule and the frequency bands

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets
        windowSeconds : Window length in seconds ( default : 2.0 )
        gapFactor : A time step longer than this multiple of the sampling step starts a new stream ( default : 1.5 )
        bandEdges : Lower edges in Hz of the band power features, the last band reaching the Nyquist frequency ( default : ( 0.5, 1, 2, 4, 8, 16 ) )

        Output
        ======
        None
        """
        # Read The Converted Data Tables
        self.sonarDirectory = Path(dataRoot) / "sonar"
        self.recordingDirectory = self.sonarDirectory / "recordings"
        self.recordingIndex = readCsv(self.sonarDirectory / "recordingIndex.csv")
        self.labelNames = readCsv(self.sonarDirectory / "labelNames.csv")["label"].tolist()
        self.channelNames = readCsv(self.sonarDirectory / "channelNames.csv")["channel"].tolist()

        # Name The Published Source Of The Data
        self.sourceEntry = {"url": "https://zenodo.org/api/records/7881952/files/SONAR_ML.zip/content"}

        # Derive The Window Length From The Sampling Step Shared By Every Recording
        stepValues = self.recordingIndex["medianStepMicroseconds"].unique()
        if len(stepValues) != 1:
            raise ValueError(f"The recordings do not share one sampling step: {sorted(stepValues)}")
        self.stepMicroseconds = float(stepValues[0])
        self.samplingRate = 1e6 / self.stepMicroseconds
        self.windowSeconds = windowSeconds
        self.windowSamples = int(round(windowSeconds * self.samplingRate))
        self.gapFactor = gapFactor

        # Map The Band Edges To Frequency Bins Of One Window
        self.binWidth = self.samplingRate / self.windowSamples
        self.bandEdges = list(bandEdges)
        bandBins = [int(round(bandEdge / self.binWidth)) for bandEdge in self.bandEdges] + [self.windowSamples // 2 + 1]
        self.bandRanges = list(zip(bandBins[:-1], bandBins[1:]))
        bandLabels = [f"{lowEdge:g}to{highEdge:g}Hz" for lowEdge, highEdge in zip(self.bandEdges[:-1], self.bandEdges[1:])] + [f"above{self.bandEdges[-1]:g}Hz"]

        # Store The Channel Groups And The Feature Names
        self.accelerationColumns = [channelIndex for channelIndex, channelName in enumerate(self.channelNames) if "_acc" in channelName]
        self.sensorCount = len(self.accelerationColumns) // 3
        self.statisticNames = ["mean", "variance", *[f"power{bandLabel}" for bandLabel in bandLabels], "dominantFrequency"]
        self.featureNames = [f"{channelName}_{statisticName}" for channelName in self.channelNames for statisticName in self.statisticNames]
        self.scoreNames = ["accelerationVariance", "windowChange", "spectralEntropy"]
