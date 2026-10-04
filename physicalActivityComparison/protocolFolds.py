# This file is responsible for the two evaluation protocols: shuffled stratified folds, which is how the 10-fold cross validation of Attal et al. is read here, and leave one participant out
# %%
# Importing Libraries
import numpy as np
from sklearn.model_selection import StratifiedKFold


# %%
# Protocol Folds
class protocolFolds:
    def __init__(self, foldCount=10, innerFoldCount=3, randomSeed=42):
        """
        This class initializes the fold maker with the number of shuffled folds, the inner hold out share and the shuffling seed

        Arguments
        =========
        foldCount : Number of outer folds of the shuffled protocol ( default : 10 )
        innerFoldCount : Inverse share of the training rows held out once for tuning ( default : 3 )
        randomSeed : Seed of the shuffled folds ( default : 42 )

        Output
        ======
        None
        """
        # Store The Fold Settings
        self.foldCount = foldCount
        self.innerFoldCount = innerFoldCount
        self.randomSeed = randomSeed
        self.protocolNames = ["shuffledFolds", "leaveOneSubjectOut"]

    def outerFolds(self, protocolName, rowTable):
        """
        This method builds the outer folds of one protocol

        Arguments
        =========
        protocolName : shuffledFolds or leaveOneSubjectOut
        rowTable : Row table with subject and label columns

        Output
        ======
        List of ( training row positions , test row positions ) tuples
        """
        # Shuffle Rows Into Stratified Folds, The Reading Of Attal Et Al.'s 10-Fold Cross Validation Used Here
        rowPositions = np.arange(len(rowTable))
        if protocolName == "shuffledFolds":
            splitter = StratifiedKFold(n_splits=self.foldCount, shuffle=True, random_state=self.randomSeed)
            return [(trainRows, testRows) for trainRows, testRows in splitter.split(rowPositions, rowTable["label"].to_numpy())]

        # Hold Out Every Participant In Turn
        if protocolName == "leaveOneSubjectOut":
            subjects = rowTable["subject"].to_numpy()
            return [(rowPositions[subjects != subjectNumber], rowPositions[subjects == subjectNumber]) for subjectNumber in np.unique(subjects)]
        raise ValueError(f"No protocol named {protocolName}")

    def innerSplit(self, protocolName, rowTable, trainRows):
        """
        This method holds out part of the training rows for tuning in the same way the protocol holds out its test rows

        Arguments
        =========
        protocolName : shuffledFolds or leaveOneSubjectOut
        rowTable : Row table of all rows
        trainRows : Training row positions of the outer fold

        Output
        ======
        Tuple of the inner training and validation positions, both relative to trainRows
        """
        # Take The First Of The Shuffled Stratified Inner Folds
        trainTable = rowTable.iloc[trainRows].reset_index(drop=True)
        relativePositions = np.arange(len(trainRows))
        if protocolName == "shuffledFolds":
            splitter = StratifiedKFold(n_splits=self.innerFoldCount, shuffle=True, random_state=self.randomSeed)
            innerTrain, innerValid = next(splitter.split(relativePositions, trainTable["label"].to_numpy()))
            return innerTrain, innerValid

        # Hold Out Every Third Remaining Participant
        if protocolName == "leaveOneSubjectOut":
            remainingSubjects = np.unique(trainTable["subject"].to_numpy())
            validMask = np.isin(trainTable["subject"].to_numpy(), remainingSubjects[::self.innerFoldCount])
            return relativePositions[~validMask], relativePositions[validMask]
        raise ValueError(f"No protocol named {protocolName}")
