# This file is responsible for listing the filled frames of the published data and measuring what their replacement changes
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from activityRecognitionChain.chainEvaluation import chainEvaluator
from common.csvWriters import writeCsv


# %%
# Filled Frames
class filledFrames:
    def inventoryFills(self):
        """
        This method lists the runs of filled frames per participant and IMU and checks whether every filled value equals the mean of its frame's class

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per run of filled frames
        """
        # Find The Filled Frames Of Every IMU In The Published Data
        loader = self.evaluator.loader
        fillRows = []
        for subjectNumber, recording in self.evaluator.recordings.items():
            filledFrames = loader.findFilledFrames(recording["data"])
            for imuIndex, (imuNumber, placementName) in enumerate(loader.placementNames.items()):
                imuChannels = loader.selectChannels([f"acc_{imuNumber}", f"gyr_{imuNumber}"])
                imuData = recording["data"][:, imuChannels]
                classMeans = {classLabel: imuData[recording["labels"] == classLabel].mean(axis=0) for classLabel in loader.classLabels}

                # Split The Filled Frames Into Runs And Compare Each Frame With Its Class Mean
                runEdges = np.flatnonzero(np.diff(np.concatenate([[0], filledFrames[:, imuIndex].astype(np.int64), [0]])))
                for runStart, runEnd in zip(runEdges[0::2], runEdges[1::2]):
                    runLabels = recording["labels"][runStart:runEnd]
                    expectedValues = np.stack([classMeans[classLabel] for classLabel in runLabels])
                    labelValues, labelCounts = np.unique(runLabels, return_counts=True)
                    fillRows.append({
                        "subject": subjectNumber, "imu": imuNumber, "placement": placementName, "runStart": int(runStart), "runEnd": int(runEnd),
                        "frames": int(runEnd - runStart), "labelCounts": " ".join(f"{labelValue}:{labelCount}" for labelValue, labelCount in zip(labelValues, labelCounts)),
                        "equalsClassMean": bool(np.allclose(imuData[runStart:runEnd], expectedValues, rtol=0, atol=1e-6)),
                    })
        fillTable = pd.DataFrame(fillRows, columns=["subject", "imu", "placement", "runStart", "runEnd", "frames", "labelCounts", "equalsClassMean"])
        writeCsv(fillTable, self.resultsDirectory / "dataFills.csv")

        # Record How The Filled Frames Are Found
        self.recordParameter("dataFills", "detectionRule", "every channel of that IMU holds a fractional value, although the sensors record whole numbers")
        return fillTable

    def testFillSensitivity(self, basicTable, placementTable):
        """
        This method reruns the hand accelerometer of the basic ARC and every placement with the filled frames replaced without labels, once by the IMU means and once by interpolation ( extra c )

        Arguments
        =========
        basicTable : Section 5.1 averages
        placementTable : Section 5.4 averages

        Output
        ======
        Dataframe comparing the published data with both label free replacements
        """
        # Pair Every Configuration With Its Published Rows
        mainBasic = basicTable[(basicTable["stepVariant"] == "paperStep") & (basicTable["classifier"] == "kNearestNeighbour")]
        comparedSets = [("handAccelerometer", self.sensorSets["handAccelerometer"], mainBasic[mainBasic["sensorSet"] == "handAccelerometer"])]
        comparedSets += [(placementName, sensorKeys, placementTable[placementTable["placement"] == placementName]) for placementName, sensorKeys in self.placementSets.items()]

        # Evaluate The Same Configurations Under Each Replacement
        sensitivityRows = []
        for replacementName in self.fillReplacements:
            fillFreeEvaluator = chainEvaluator(self.dataRoot, randomSeed=self.randomSeed, nJobs=self.nJobs, fillReplacement=replacementName)
            for configurationName, sensorKeys, publishedRows in comparedSets:
                fillFreeRows = self.runConfiguration(f"fillSensitivity {replacementName}", configurationName, self.buildConfiguration(sensorKeys), evaluator=fillFreeEvaluator, keepRounds=False)
                for schemeName in self.evaluator.schemeNames:
                    publishedRow = publishedRows[publishedRows["scheme"] == schemeName].iloc[0]
                    fillFreeRow = fillFreeRows[fillFreeRows["scheme"] == schemeName].iloc[0]
                    sensitivityRow = {"replacement": replacementName, "configuration": configurationName, "sensors": "+".join(sensorKeys), "scheme": schemeName}
                    for metricName in ["precisionAll", "recallAll", "precisionNonNull", "recallNonNull"]:
                        sensitivityRow[f"{metricName}Published"] = publishedRow[metricName]
                        sensitivityRow[f"{metricName}FillFree"] = fillFreeRow[metricName]
                        sensitivityRow[f"{metricName}Change"] = fillFreeRow[metricName] - publishedRow[metricName]
                    sensitivityRows.append(sensitivityRow)
        sensitivityTable = pd.DataFrame(sensitivityRows)
        writeCsv(sensitivityTable, self.resultsDirectory / "extraFillSensitivity.csv")

        # Record Both Replacements
        self.recordParameter("dataFills", "replacementMean", "every filled frame of an IMU gets that IMU's per channel mean over its unfilled frames")
        self.recordParameter("dataFills", "replacementInterpolate", "every filled frame of an IMU gets, per channel, the straight line between the nearest unfilled frames before and after it, or the nearest unfilled value at a recording edge")
        return sensitivityTable
