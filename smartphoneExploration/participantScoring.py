# This file is responsible for scoring participant identification with a random window split and with whole recording bouts held out
# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# %%
# Participant Scoring
class participantScoring:
    def measureNeighbourLeakage(self, trainIndices, testIndices):
        """
        This method counts test windows that share raw samples with a neighbouring window of the same bout in the training rows

        Arguments
        =========
        trainIndices : Row indices of the training windows
        testIndices : Row indices of the test windows

        Output
        ======
        Number of test windows with an overlapping neighbour among the training windows
        """
        # Mark The Training Rows
        isTrain = np.zeros(len(self.featureFrame), dtype=bool)
        isTrain[trainIndices] = True

        # Check The Previous And The Next Window Of The Same Bout
        sameBoutAsNext = self.boutNumbers[:-1] == self.boutNumbers[1:]
        previousInTrain = np.concatenate([[False], sameBoutAsNext & isTrain[:-1]])
        nextInTrain = np.concatenate([sameBoutAsNext & isTrain[1:], [False]])
        return int(np.sum((previousInTrain | nextInTrain)[testIndices]))

    def scoreHeldOutBouts(self, featureMatrix, subjects, activityIndices):
        """
        This method scores participant identification on every fold of a stratified group split that keeps whole recording bouts on one side

        Arguments
        =========
        featureMatrix : Feature values of every window
        subjects : Subject of every window
        activityIndices : Row indices of the windows of one activity

        Output
        ======
        Tuple of the fold summary ( mean and sample standard deviation of accuracy and macro F1, fold count, test subjects without a training bout ) and one row per fold
        """
        # Split The Activity Into Folds That Keep Every Bout On One Side
        foldSplitter = StratifiedGroupKFold(n_splits=self.boutHeldOutFolds, shuffle=True, random_state=self.randomSeed)
        foldSplits = foldSplitter.split(activityIndices, subjects[activityIndices], groups=self.boutNumbers[activityIndices])

        # Train And Score Every Fold
        foldRows = []
        for foldNumber, (trainPositions, testPositions) in enumerate(foldSplits, start=1):
            trainIndices = activityIndices[trainPositions]
            testIndices = activityIndices[testPositions]
            foldModel = self.buildClassifier().fit(featureMatrix[trainIndices], subjects[trainIndices])
            foldPredictions = foldModel.predict(featureMatrix[testIndices])
            foldRows.append({
                "fold": foldNumber,
                "trainWindows": len(trainIndices),
                "testWindows": len(testIndices),
                "testSubjectsWithoutTrainingBouts": len(np.setdiff1d(np.unique(subjects[testIndices]), np.unique(subjects[trainIndices]))),
                "accuracy": accuracy_score(subjects[testIndices], foldPredictions),
                "macroF1": f1_score(subjects[testIndices], foldPredictions, average="macro"),
            })

        # Summarise The Folds
        return self.summariseHeldOutFolds(foldRows), foldRows

    def summariseHeldOutFolds(self, foldRows):
        """
        This method summarises the folds of the bout held out split with their mean and sample standard deviation

        Arguments
        =========
        foldRows : One dictionary of scores per fold

        Output
        ======
        Dictionary of the fold count, the mean and standard deviation of accuracy and macro F1 and the test subjects without a training bout
        """
        # Take The Fold Table And Summarise It
        foldTable = pd.DataFrame(foldRows)
        foldSummary = {
            "boutHeldOutFolds": len(foldTable),
            "boutHeldOutAccuracyMean": foldTable["accuracy"].mean(),
            "boutHeldOutAccuracyStd": foldTable["accuracy"].std(ddof=1),
            "boutHeldOutMacroF1Mean": foldTable["macroF1"].mean(),
            "boutHeldOutMacroF1Std": foldTable["macroF1"].std(ddof=1),
            "testSubjectsWithoutTrainingBouts": int(foldTable["testSubjectsWithoutTrainingBouts"].sum()),
        }
        return foldSummary

    def scoreActivity(self, activityName, featureMatrix, subjects, activities):
        """
        This method identifies the participants of one activity with a random window split and with whole bouts held out

        Arguments
        =========
        activityName : Activity whose windows are used
        featureMatrix : Feature values of every window
        subjects : Subject of every window
        activities : Activity of every window

        Output
        ======
        Tuple of the score row of the activity and the held out fold rows of the activity
        """
        # Train One Participant Classifier On The Activity
        activityIndices = np.flatnonzero(activities == activityName)
        trainIndices, testIndices = train_test_split(
            activityIndices, test_size=self.identificationTestFraction,
            stratify=subjects[activityIndices], random_state=self.randomSeed,
        )
        participantModel = self.buildClassifier().fit(featureMatrix[trainIndices], subjects[trainIndices])
        predictedSubjects = participantModel.predict(featureMatrix[testIndices])

        # Keep The Walking Model For The Sensor Comparison
        if activityName == self.walkingActivity:
            self.walkingModel = participantModel

        # Score The Same Activity With Whole Bouts Held Out
        heldOutSummary, heldOutFolds = self.scoreHeldOutBouts(featureMatrix, subjects, activityIndices)

        # Collect The Scores Next To The Durations
        activitySeconds = self.durationTable[activityName]
        identificationRow = {
            "Activity": activityName,
            "windows": len(activityIndices),
            "trainWindows": len(trainIndices),
            "testWindows": len(testIndices),
            "participants": len(np.unique(subjects[activityIndices])),
            "accuracy": accuracy_score(subjects[testIndices], predictedSubjects),
            "macroF1": f1_score(subjects[testIndices], predictedSubjects, average="macro"),
            "testWindowsWithTrainNeighbour": self.measureNeighbourLeakage(trainIndices, testIndices),
            **heldOutSummary,
            "meanSecondsPerParticipant": activitySeconds.mean(),
            "medianSecondsPerParticipant": activitySeconds.median(),
            "minSecondsPerParticipant": activitySeconds.min(),
            "maxSecondsPerParticipant": activitySeconds.max(),
        }
        print(f"[ SMARTPHONE : PARTICIPANT ACCURACY {activityName} ] : {identificationRow['accuracy']:.4f}")
        return identificationRow, [{"Activity": activityName, **foldRow} for foldRow in heldOutFolds]
