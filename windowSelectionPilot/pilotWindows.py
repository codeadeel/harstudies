# This file is responsible for the pilot steps that build the windows of every recording and write the data inventory
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common.csvWriters import writeCsv


# %%
# Pilot Windows
class pilotWindows:
    def buildRecordingParts(self):
        """
        This method builds the windows of every recording in worker processes

        Arguments
        =========
        None

        Output
        ======
        List with one result dictionary per recording
        """
        # Build The Windows Of Every Recording In Worker Processes
        recordingIndex = self.builder.recordingIndex
        return Parallel(n_jobs=self.nJobs)(delayed(self.builder.buildRecording)(recordingName) for recordingName in recordingIndex["recording"])

    def joinRecordingParts(self, recordingParts):
        """
        This method joins the windows of all recordings, numbering the streams across recordings, and adds up the CPU seconds

        Arguments
        =========
        recordingParts : List with one result dictionary per recording

        Output
        ======
        None
        """
        # Join The Windows, Numbering The Streams Across Recordings
        recordingIndex = self.builder.recordingIndex
        participantNumbers = dict(zip(recordingIndex["recording"], recordingIndex["participant"]))
        tableParts = []
        streamOffset = 0
        for recordingPart in recordingParts:
            tableParts.append(pd.DataFrame({
                "recording": recordingPart["recording"],
                "participant": participantNumbers[recordingPart["recording"]],
                "stream": recordingPart["streams"] + streamOffset,
                "label": recordingPart["labels"],
            }))
            streamOffset += recordingPart["counts"]["streams"]
        self.windowTable = pd.concat(tableParts, ignore_index=True)
        self.featureMatrix = np.concatenate([recordingPart["features"] for recordingPart in recordingParts])
        self.scoreMatrix = np.concatenate([recordingPart["scores"] for recordingPart in recordingParts])
        self.cpuSeconds = {cpuName: sum(recordingPart["cpuSeconds"][cpuName] for recordingPart in recordingParts) for cpuName in ["features", *self.builder.scoreNames]}

    def findStreamRanges(self):
        """
        This method finds the first and the end window of every stream

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Find Where Every Stream Starts And Ends
        streamNumbers = self.windowTable["stream"].to_numpy()
        streamBreaks = np.flatnonzero(np.diff(streamNumbers)) + 1
        self.streamRanges = list(zip(np.concatenate([[0], streamBreaks]).tolist(), np.concatenate([streamBreaks, [len(streamNumbers)]]).tolist()))

    def recordDataCounts(self, recordingParts):
        """
        This method records the sample, recording, window and cleaning counts for runParameters.csv

        Arguments
        =========
        recordingParts : List with one result dictionary per recording

        Output
        ======
        None
        """
        # Record The Data Counts
        recordingIndex = self.builder.recordingIndex
        sampleCounts = {countName: sum(recordingPart["counts"][countName] for recordingPart in recordingParts) for countName in recordingParts[0]["counts"]}
        for countName, countValue in sampleCounts.items():
            self.recordParameter("data", countName, countValue)
        self.recordParameter("data", "tailSamples", sampleCounts["samples"] - sampleCounts["repeatedRows"] - sampleCounts["droppedStreamSamples"] - sampleCounts["windowedSamples"])
        self.recordParameter("data", "recordings", len(recordingIndex))
        self.recordParameter("data", "participants", recordingIndex["participant"].nunique())
        self.recordParameter("data", "windows", len(self.windowTable))
        self.recordParameter("data", "windowStreams", len(self.streamRanges))
        self.recordParameter("data", "missingSamples", int(recordingIndex["missingSamples"].sum()))
        self.recordParameter("data", "missingRuns", int(recordingIndex["missingRuns"].sum()))
        self.recordParameter("data", "longestMissingRun", int(recordingIndex["longestMissingRun"].max()))
        self.recordParameter("data", "gaps", int(recordingIndex["gaps"].sum()))
        self.recordParameter("data", "medianAccelerationMagnitude", float(recordingIndex["medianAccelerationMagnitude"].median()))
        self.recordParameter("data", "lowestRecordingAccelerationMagnitude", float(recordingIndex["medianAccelerationMagnitude"].min()))
        self.recordParameter("data", "highestRecordingAccelerationMagnitude", float(recordingIndex["medianAccelerationMagnitude"].max()))
        self.recordParameter("data", "labels", len(self.builder.labelNames))
        self.recordParameter("data", "sourceUrl", self.builder.sourceEntry["url"])

    def writeInventory(self):
        """
        This method writes the number and the share of windows per activity

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per activity
        """
        # Write The Windows Per Activity
        windowLabels = self.windowTable["label"].to_numpy()
        windowCounts = np.bincount(windowLabels, minlength=len(self.builder.labelNames))
        inventoryTable = pd.DataFrame({"activity": self.builder.labelNames, "windows": windowCounts, "windowShare": windowCounts / len(windowLabels)})
        writeCsv(inventoryTable, self.resultsDirectory / "dataInventory.csv")
        return inventoryTable

    def buildWindows(self):
        """
        This method builds the windows of every recording in worker processes, numbers the streams and writes the data inventory

        Arguments
        =========
        None

        Output
        ======
        Dataframe with one row per window
        """
        # Build The Windows Of Every Recording In Worker Processes
        recordingParts = self.buildRecordingParts()

        # Join The Windows, Numbering The Streams Across Recordings
        self.joinRecordingParts(recordingParts)

        # Find Where Every Stream Starts And Ends
        self.findStreamRanges()

        # Record The Data Counts And Write The Windows Per Activity
        self.recordDataCounts(recordingParts)
        self.writeInventory()
        print(f"[ PILOT : WINDOWS ] : {len(self.windowTable)} windows in {len(self.streamRanges)} streams")
        return self.windowTable
