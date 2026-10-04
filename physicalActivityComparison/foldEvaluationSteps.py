# This file is responsible for running every classifier under every protocol on the worker processes and turning the jobs into fold records
# %%
# Importing Libraries
import numpy as np
from joblib import Parallel, delayed

from physicalActivityComparison.foldWorkers import runSupervisedFold, runUnsupervisedFold


# %%
# Fold Evaluation Steps
class foldEvaluationSteps:
    def evaluateAll(self, variants):
        """
        This method runs every supervised classifier under every protocol and the unsupervised models under the shuffled folds, in one batch of worker processes that starts with the costliest job groups

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        List of fold records with the predictions, settings and row counts
        """
        # List The Job Groups In The Order Of Their Measured Cost Per Fold
        jobSpecs = self.listFoldJobs(variants)

        # Run Every Job On The Worker Processes, Keeping Job Order
        foldResults = self.runFoldJobs(variants, jobSpecs)

        # Check That Both Learning Kinds Kept The Same Features In The Same Fold
        self.checkKeptFeatures(jobSpecs, foldResults)

        # Turn Every Job Into One Record Per Classifier
        foldRecords = self.listFoldRecords(variants, jobSpecs, foldResults)
        print(f"[ COMPARISON : FOLD JOBS ] : {len(jobSpecs)}")
        return foldRecords

    def listFoldJobs(self, variants):
        """
        This method lists one job per fold for every group of classifiers, starting with the costliest groups

        Arguments
        =========
        variants : Output of prepareData

        Output
        ======
        List of job tuples: learning kind, variant, protocol, fold number, training rows, test rows, inner split, selection flag and classifier names
        """
        # List The Job Groups In The Order Of Their Measured Cost Per Fold
        jobSpecs = []
        for learningName, variantName, modelNames, selectFeatures in [
            ("supervised", "features80", self.supervisedNames, True), ("unsupervised", "rawFull", self.unsupervisedNames, False),
            ("unsupervised", "features80", self.unsupervisedNames, True), ("supervised", "rawSubsample", self.supervisedNames, False),
            ("supervised", "rawFull", ["kNearestNeighbour"], False),
        ]:
            variant = variants[variantName]

            # Add One Job Per Fold, The Unsupervised Models Under The Shuffled Folds Only
            for protocolName in self.folds.protocolNames if learningName == "supervised" else ["shuffledFolds"]:
                for foldNumber, (trainRows, testRows) in enumerate(self.folds.outerFolds(protocolName, variant["rows"]), start=1):
                    validSplit = self.folds.innerSplit(protocolName, variant["rows"], trainRows) if learningName == "supervised" else None
                    jobSpecs.append((learningName, variantName, protocolName, foldNumber, trainRows, testRows, validSplit, selectFeatures, modelNames))
        return jobSpecs

    def runFoldJobs(self, variants, jobSpecs):
        """
        This method runs every job on the worker processes of joblib and keeps the job order

        Arguments
        =========
        variants : Output of prepareData
        jobSpecs : Output of listFoldJobs

        Output
        ======
        List of fold results, one per job, in job order
        """
        # Run Every Job On The Worker Processes, Keeping Job Order
        foldResults = Parallel(n_jobs=self.nJobs)(
            delayed(runSupervisedFold)(
                self.classifiers, variants[variantName]["features"], variants[variantName]["rows"]["label"].to_numpy(),
                trainRows, testRows, validSplit, selectFeatures, classifierNames,
            ) if learningName == "supervised" else delayed(runUnsupervisedFold)(
                self.classifiers, variants[variantName]["features"], variants[variantName]["rows"]["label"].to_numpy(),
                variants[variantName]["rows"]["subject"].to_numpy(), trainRows, testRows, selectFeatures, classifierNames,
            )
            for learningName, variantName, _, _, trainRows, testRows, validSplit, selectFeatures, classifierNames in jobSpecs
        )
        return foldResults

    def checkKeptFeatures(self, jobSpecs, foldResults):
        """
        This method checks that the supervised and unsupervised jobs of the same fold kept the same features

        Arguments
        =========
        jobSpecs : Output of listFoldJobs
        foldResults : Output of runFoldJobs

        Output
        ======
        None
        """
        # Check That Both Learning Kinds Kept The Same Features In The Same Fold
        keptBySupervised = {
            (jobSpec[1], jobSpec[3]): foldResult["keptColumns"]
            for jobSpec, foldResult in zip(jobSpecs, foldResults) if jobSpec[0] == "supervised" and jobSpec[2] == "shuffledFolds"
        }
        for jobSpec, foldResult in zip(jobSpecs, foldResults):
            if jobSpec[0] == "unsupervised" and not np.array_equal(foldResult["keptColumns"], keptBySupervised[(jobSpec[1], jobSpec[3])]):
                raise RuntimeError(f"Feature selection differs between learning kinds in {jobSpec[1]} fold {jobSpec[3]}")

    def listFoldRecords(self, variants, jobSpecs, foldResults):
        """
        This method turns every job into one record per classifier with the predictions, the settings and the row counts

        Arguments
        =========
        variants : Output of prepareData
        jobSpecs : Output of listFoldJobs
        foldResults : Output of runFoldJobs

        Output
        ======
        List of fold records
        """
        # Turn Every Job Into One Record Per Classifier
        foldRecords = []
        for (learningName, variantName, protocolName, foldNumber, trainRows, testRows, _, selectFeatures, classifierNames), foldResult in zip(jobSpecs, foldResults):
            for classifierName in classifierNames:
                foldRecords.append({
                    "variant": variantName, "protocol": protocolName, "fold": foldNumber, "classifier": classifierName, "learning": learningName,
                    "trainRows": len(trainRows), "testRows": len(testRows), "testPositions": testRows,
                    "keptFeatures": len(foldResult["keptColumns"]) if selectFeatures else None,
                    "keptFeatureNames": ";".join(variants[variantName]["names"][featureIndex] for featureIndex in foldResult["keptColumns"]) if selectFeatures and learningName == "supervised" else "",
                    "predictions": foldResult[classifierName]["predictions"], **foldResult[classifierName]["settings"],
                })
        return foldRecords
