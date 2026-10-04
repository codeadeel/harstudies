# This file is responsible for splitting SONAR recordings into streams, filling missing values and labelling windows
# %%
# Importing Libraries
import numpy as np


# %%
# Window Streams
class windowStreams:
    def splitStreams(self, sampleTimes):
        """
        This method splits a recording into streams wherever the time step exceeds the gap rule

        Arguments
        =========
        sampleTimes : Microsecond time stamps of the recording

        Output
        ======
        List of ( first sample , end sample ) pairs
        """
        # Break After Every Long Step
        breakPositions = np.flatnonzero(np.diff(sampleTimes) > self.gapFactor * self.stepMicroseconds) + 1
        streamStarts = np.concatenate([[0], breakPositions])
        streamEnds = np.concatenate([breakPositions, [len(sampleTimes)]])
        return list(zip(streamStarts.tolist(), streamEnds.tolist()))

    def holdLastValid(self, streamSignals):
        """
        This method fills missing values of a stream with the last valid value of the same channel, and leading missing values with the first valid one

        Arguments
        =========
        streamSignals : Samples by channels array of one stream

        Output
        ======
        Tuple of the filled array, or None when a channel has no valid value, and the number of filled values
        """
        # Find The Missing Values
        missingValues = np.isnan(streamSignals)
        missingCount = int(missingValues.sum())
        if missingCount == 0:
            return streamSignals, 0
        if missingValues.all(axis=0).any():
            return None, missingCount

        # Carry The Last Valid Row Of Every Channel Forward
        channelPositions = np.arange(streamSignals.shape[1])
        validRows = np.where(~missingValues, np.arange(len(streamSignals))[:, None], -1)
        np.maximum.accumulate(validRows, axis=0, out=validRows)
        firstValidRows = np.argmax(~missingValues, axis=0)
        filledRows = np.where(validRows < 0, firstValidRows[None, :], validRows)
        return streamSignals[filledRows, channelPositions[None, :]], missingCount

    def majorityLabels(self, windowLabels):
        """
        This method labels every window with its most frequent sample label, the smaller code winning a tie

        Arguments
        =========
        windowLabels : Windows by samples array of label codes

        Output
        ======
        Label code of every window
        """
        # Count The Codes Of Every Window
        labelCounts = np.zeros((len(windowLabels), len(self.labelNames)), dtype=np.int64)
        np.add.at(labelCounts, (np.repeat(np.arange(len(windowLabels)), windowLabels.shape[1]), windowLabels.ravel()), 1)
        return np.argmax(labelCounts, axis=1).astype(np.uint8)
