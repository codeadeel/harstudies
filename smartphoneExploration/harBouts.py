# This file is responsible for finding the continuous recording bouts inside the UCI HAR windows
# %%
# Importing Libraries
import numpy as np


# %%
# Recording Bouts
class recordingBouts:
    def findContinuedWindows(self, signalArrays, subjects, activities, splits):
        """
        This method marks each pair of neighbouring rows where the next window repeats the second half of the current window

        Arguments
        =========
        signalArrays : Raw signal arrays in the combined frame row order
        subjects : Subject of every row
        activities : Activity of every row
        splits : Split name of every row

        Output
        ======
        Boolean array whose entry i is true when row i + 1 continues row i within one recording
        """
        # Compare The Second Half Of Each Window With The First Half Of The Next
        overlapSamples = self.windowSamples - self.windowStepSamples
        continuesNext = np.ones(len(subjects) - 1, dtype=bool)
        for signalArray in signalArrays:
            continuesNext &= np.all(signalArray[:-1, self.windowStepSamples:] == signalArray[1:, :overlapSamples], axis=1)

        # Require The Same Subject, Activity And Split
        subjects, activities, splits = np.asarray(subjects), np.asarray(activities), np.asarray(splits)
        continuesNext &= (subjects[:-1] == subjects[1:]) & (activities[:-1] == activities[1:]) & (splits[:-1] == splits[1:])
        return continuesNext

    def labelBouts(self, continuesNext):
        """
        This method numbers the recording bouts, starting a new bout wherever a row does not continue the previous row

        Arguments
        =========
        continuesNext : Boolean array from findContinuedWindows

        Output
        ======
        Integer bout number for every row
        """
        # Count The Breaks Before Each Row
        return np.concatenate([[0], np.cumsum(~continuesNext)])
