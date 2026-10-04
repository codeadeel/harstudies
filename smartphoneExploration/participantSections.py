# This file is responsible for identifying participants within each activity and counting the seconds of data per participant
# %%
# Importing Libraries
import pandas as pd

from common.csvWriters import writeCsv
from smartphoneExploration.participantPlots import participantPlots
from smartphoneExploration.participantScoring import participantScoring


# %%
# Participant Sections, With The Scoring And Plot Methods Inherited From Their Files
class participantSections(participantScoring, participantPlots):
    def identifyParticipants(self):
        """
        This method identifies participants within each activity and adds the seconds of data per participant ( sections 5.1 and 5.2 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of identification scores and durations per activity
        """
        # Prepare The Shared Arrays
        featureMatrix = self.featureFrame[self.featureNames].to_numpy()
        subjects = self.featureFrame["subject"].to_numpy()
        activities = self.featureFrame["Activity"].to_numpy()

        # Find Participant Durations
        self.findParticipantDurations()

        # Train One Participant Classifier Per Activity
        identificationTable = self.scoreActivities(featureMatrix, subjects, activities)

        # Record The Split Settings
        self.recordSplitSettings()

        # Plot Accuracy Next To The Seconds Per Participant
        self.plotParticipantIdentification(identificationTable)

        # Plot The Seconds Of Every Participant And Activity
        self.plotParticipantDurations()
        return identificationTable

    def findParticipantDurations(self):
        """
        This method counts the seconds of data of every participant in every activity and writes them

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Find Participant Durations
        windowCounts = pd.crosstab(self.featureFrame["subject"], self.featureFrame["Activity"])
        self.durationTable = windowCounts.reindex(columns=self.activityOrder, fill_value=0) * self.loader.windowStepSeconds
        self.durationTable.columns.name = None
        writeCsv(self.durationTable.reset_index(), self.resultsDirectory / "participantDurations.csv")

    def scoreActivities(self, featureMatrix, subjects, activities):
        """
        This method scores participant identification in every activity and writes the score tables

        Arguments
        =========
        featureMatrix : Feature values of every window
        subjects : Subject of every window
        activities : Activity of every window

        Output
        ======
        Dataframe of identification scores and durations per activity
        """
        # Train One Participant Classifier Per Activity
        identificationRows = []
        heldOutFoldRows = []
        for activityName in self.activityOrder:
            identificationRow, foldRows = self.scoreActivity(activityName, featureMatrix, subjects, activities)
            identificationRows.append(identificationRow)
            heldOutFoldRows += foldRows
        identificationTable = pd.DataFrame(identificationRows)
        writeCsv(identificationTable, self.resultsDirectory / "participantIdentification.csv")
        writeCsv(pd.DataFrame(heldOutFoldRows), self.resultsDirectory / "participantIdentificationFolds.csv")
        return identificationTable

    def recordSplitSettings(self):
        """
        This method records how the participant identification splits its windows for runParameters.csv

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Record The Split Settings
        self.recordParameter("participantIdentification", "testFraction", self.identificationTestFraction)
        self.recordParameter("participantIdentification", "stratifiedBy", "subject")
        self.recordParameter("participantIdentification", "splitFunction", "sklearn train_test_split within each activity")
        self.recordParameter("participantIdentification", "boutHeldOutSplit", f"all {self.boutHeldOutFolds} folds of sklearn StratifiedGroupKFold, stratified by subject, grouped by recording bout, shuffled with the run seed")
        self.recordParameter("participantIdentification", "boutHeldOutSpread", "mean and sample standard deviation ( ddof 1 ) over the folds")
