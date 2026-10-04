# This file is responsible for the work done on one fold in a worker process: feature selection, scaling, tuning, fitting and prediction
# %%
# Importing Libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


# %%
# Fold Workers
def runSupervisedFold(classifiers, featureMatrix, labels, trainRows, testRows, validSplit, selectFeatures, classifierNames):
    """
    This function selects features inside the training rows when asked, scales them with the training rows, then tunes, fits and applies every supervised classifier

    Arguments
    =========
    classifiers : attalClassifiers object
    featureMatrix : Features of all rows ( rows , features )
    labels : Label of every row
    trainRows : Training row positions
    testRows : Test row positions
    validSplit : Inner training and validation positions relative to trainRows
    selectFeatures : Whether the forest selection runs on the training rows
    classifierNames : Supervised classifiers to run

    Output
    ======
    Dictionary with the kept feature positions and, per classifier, the test predictions and the chosen settings
    """
    # Keep One Thread Per Worker
    with threadpool_limits(limits=1):
        keptColumns = np.arange(featureMatrix.shape[1])
        if selectFeatures:
            featureOrder, _, keptCount = classifiers.rankFeatures(featureMatrix[trainRows], labels[trainRows])
            keptColumns = featureOrder[:keptCount]

        # Scale With The Training Rows
        scaler = StandardScaler().fit(featureMatrix[np.ix_(trainRows, keptColumns)])
        trainFeatures = scaler.transform(featureMatrix[np.ix_(trainRows, keptColumns)])
        testFeatures = scaler.transform(featureMatrix[np.ix_(testRows, keptColumns)])

        # Tune, Fit And Predict Every Classifier
        foldResult = {"keptColumns": keptColumns}
        for classifierName in classifierNames:
            fittedModel, chosenSettings = classifiers.fitSupervised(classifierName, trainFeatures, labels[trainRows], validSplit)
            foldResult[classifierName] = {"predictions": fittedModel.predict(testFeatures), "settings": chosenSettings}
    return foldResult

def runUnsupervisedFold(classifiers, featureMatrix, labels, subjects, trainRows, testRows, selectFeatures, modelNames):
    """
    This function selects features inside the training rows when asked, fits every unsupervised model on the training rows without labels, maps clusters to labels on the training rows and labels the test rows

    Arguments
    =========
    classifiers : attalClassifiers object
    featureMatrix : Features of all rows ( rows , features ), rows in time order per participant
    labels : Label of every row
    subjects : Participant of every row
    trainRows : Training row positions in ascending order
    testRows : Test row positions in ascending order
    selectFeatures : Whether the forest selection runs on the training rows, with the same result as the supervised job of the same fold
    modelNames : Unsupervised models to run

    Output
    ======
    Dictionary with the kept feature positions and, per model, the test predictions and the fit details
    """
    # Keep One Thread Per Worker And Select Features Inside The Training Rows When Asked
    with threadpool_limits(limits=1):
        keptColumns = np.arange(featureMatrix.shape[1])
        if selectFeatures:
            featureOrder, _, keptCount = classifiers.rankFeatures(featureMatrix[trainRows], labels[trainRows])
            keptColumns = featureOrder[:keptCount]

        # Scale With The Training Rows
        scaler = StandardScaler().fit(featureMatrix[np.ix_(trainRows, keptColumns)])
        trainFeatures = scaler.transform(featureMatrix[np.ix_(trainRows, keptColumns)])
        testFeatures = scaler.transform(featureMatrix[np.ix_(testRows, keptColumns)])
        _, trainLengths = np.unique(subjects[trainRows], return_counts=True)
        _, testLengths = np.unique(subjects[testRows], return_counts=True)

        # Fit Each Model, Then Map Its Clusters To Labels With The Training Labels
        foldResult = {"keptColumns": keptColumns}
        for modelName in modelNames:
            trainClusters, testClusters, fitDetails = classifiers.fitUnsupervised(modelName, trainFeatures, trainLengths, testFeatures, testLengths)
            clusterLabels = classifiers.matchClusters(trainClusters, labels[trainRows])
            foldResult[modelName] = {"predictions": clusterLabels[testClusters], "settings": fitDetails}
    return foldResult
