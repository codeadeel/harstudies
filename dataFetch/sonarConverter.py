# This file is responsible for converting each SONAR machine learning csv file into float32 acceleration and angular rate arrays of the five sensors
# %%
# Importing Libraries
import zipfile
from pathlib import PurePosixPath

import numpy as np
import pandas as pd


# %%
# SONAR Converter
class sonarConverter:
    def __init__(self, sensorNames=("LF", "LW", "ST", "RW", "RF"), gapFactor=1.5):
        """
        This class initializes the converter with the sensor codes, the expected column layout and the gap rule

        Arguments
        =========
        sensorNames : Column suffixes of the five sensors ( default : ( LF, LW, ST, RW, RF ) )
        gapFactor : A time step longer than this multiple of the median step counts as a gap ( default : 1.5 )

        Output
        ======
        None
        """
        # Store The Column Layout Of The Machine Learning Version
        self.sensorNames = list(sensorNames)
        self.streamNames = ["Quat_W", "Quat_X", "Quat_Y", "Quat_Z", "dq_W", "dq_X", "dq_Y", "dq_Z", "dv[1]", "dv[2]", "dv[3]", "Mag_X", "Mag_Y", "Mag_Z"]
        self.timeColumn = "SampleTimeFine"
        self.labelColumn = "activity"
        self.expectedHeader = [f"{streamName}_{sensorName}" for sensorName in self.sensorNames for streamName in self.streamNames] + [self.timeColumn, self.labelColumn]

        # Store The Increment Columns And The Output Channel Names
        self.velocityColumns = {sensorName: [f"dv[{axisNumber}]_{sensorName}" for axisNumber in (1, 2, 3)] for sensorName in self.sensorNames}
        self.rotationColumns = {sensorName: [f"dq_{partName}_{sensorName}" for partName in "WXYZ"] for sensorName in self.sensorNames}
        self.channelNames = [f"{sensorName}_{quantityName}{axisName}" for sensorName in self.sensorNames for quantityName in ("acc", "gyro") for axisName in "XYZ"]
        self.channelUnits = {"acc": "m/s^2", "gyro": "rad/s"}

        # Store The Time Counter Settings
        self.microsecondsPerSecond = 1e6
        self.gapFactor = gapFactor

    def angularRate(self, rotationIncrements, samplePeriod):
        """
        This method turns unit quaternion rotation increments into angular rate vectors

        Arguments
        =========
        rotationIncrements : Array of increments with the columns w, x, y and z
        samplePeriod : Seconds between samples

        Output
        ======
        Array of angular rates in rad/s with the columns x, y and z
        """
        # Put Every Increment On The Hemisphere With A Non Negative Scalar Part
        hemisphereSigns = np.where(rotationIncrements[:, :1] < 0, -1.0, 1.0)
        scalarParts = rotationIncrements[:, 0] * hemisphereSigns[:, 0]
        vectorParts = rotationIncrements[:, 1:] * hemisphereSigns

        # Scale The Rotation Axis By The Rotation Angle Over The Sample Period
        vectorNorms = np.linalg.norm(vectorParts, axis=1)
        rotationAngles = 2.0 * np.arctan2(vectorNorms, scalarParts)
        safeNorms = np.where(vectorNorms > 0, vectorNorms, 1.0)
        return vectorParts * (rotationAngles / safeNorms / samplePeriod)[:, None]

    def convertMember(self, archivePath, memberName, recordingDirectory):
        """
        This method reads one csv member of the archive, converts its increments to acceleration and angular rate and saves the signal and time arrays

        Arguments
        =========
        archivePath : Path of the SONAR zip archive
        memberName : Name of the csv member inside the archive
        recordingDirectory : Directory receiving the recording arrays

        Output
        ======
        Tuple of the index row, the sorted label names of the recording and the label position of every sample among them
        """
        # Check The Header Of The Member
        keptColumns = [columnName for sensorName in self.sensorNames for columnName in self.velocityColumns[sensorName] + self.rotationColumns[sensorName]]
        with zipfile.ZipFile(archivePath) as archive:
            with archive.open(memberName) as memberHandle:
                headerNames = memberHandle.readline().decode("utf-8").strip().split(",")
            if headerNames != self.expectedHeader:
                raise ValueError(f"{memberName} has an unexpected header")

            # Stream The Member, Parsing Only The Increment, Time And Label Columns, With Empty Increment Fields As Missing
            with archive.open(memberName) as memberHandle:
                memberFrame = pd.read_csv(
                    memberHandle, usecols=keptColumns + [self.timeColumn, self.labelColumn],
                    dtype={**{columnName: np.float64 for columnName in keptColumns}, self.timeColumn: np.int64, self.labelColumn: str},
                    keep_default_na=False, na_values={columnName: [""] for columnName in keptColumns},
                )

        # Derive The Sample Period From The Median Positive Step Of The Microsecond Counter, Which This Version Stores Unwrapped
        sampleTimes = memberFrame[self.timeColumn].to_numpy()
        timeSteps = np.diff(sampleTimes)
        if np.any(timeSteps < 0):
            raise ValueError(f"{memberName} has time stamps that go backwards")
        medianStep = float(np.median(timeSteps[timeSteps > 0]))
        samplePeriod = medianStep / self.microsecondsPerSecond

        # Convert Velocity Increments To Acceleration And Rotation Increments To Angular Rate In Float64, Then Store Float32 With Missing Values Kept
        channelArrays = []
        for sensorName in self.sensorNames:
            channelArrays.append(memberFrame[self.velocityColumns[sensorName]].to_numpy() / samplePeriod)
            channelArrays.append(self.angularRate(memberFrame[self.rotationColumns[sensorName]].to_numpy(), samplePeriod))
        signalArray = np.hstack(channelArrays).astype(np.float32)

        # Save The Signal And Time Arrays Of The Recording
        recordingName = PurePosixPath(memberName).stem
        np.save(recordingDirectory / f"{recordingName}_signals.npy", signalArray)
        np.save(recordingDirectory / f"{recordingName}_time.npy", sampleTimes)

        # Find The Runs Of Samples With A Missing Value
        missingSamples = np.isnan(signalArray).any(axis=1)
        runEdges = np.diff(np.concatenate([[0], missingSamples.astype(np.int8), [0]]))
        runLengths = np.flatnonzero(runEdges == -1) - np.flatnonzero(runEdges == 1)

        # Describe The Recording
        labelNames, labelPositions = np.unique(memberFrame[self.labelColumn].to_numpy(dtype=str), return_inverse=True)
        accelerationColumns = [channelIndex for channelIndex, channelName in enumerate(self.channelNames) if "_acc" in channelName]
        accelerationMagnitudes = np.linalg.norm(signalArray[:, accelerationColumns].reshape(len(signalArray), len(self.sensorNames), 3), axis=2)
        indexRow = {
            "recording": recordingName,
            "participant": int(recordingName.rsplit("_sub", 1)[1]),
            "samples": len(signalArray),
            "medianStepMicroseconds": medianStep,
            "repeatedTimes": int(np.sum(timeSteps == 0)),
            "gaps": int(np.sum(timeSteps > self.gapFactor * medianStep)),
            "missingSamples": int(missingSamples.sum()),
            "missingRuns": len(runLengths),
            "longestMissingRun": int(runLengths.max()) if len(runLengths) else 0,
            "labels": len(labelNames),
            "labelChanges": int(np.sum(labelPositions[1:] != labelPositions[:-1])),
            "medianAccelerationMagnitude": float(np.nanmedian(accelerationMagnitudes)),
        }
        return indexRow, labelNames.tolist(), labelPositions
