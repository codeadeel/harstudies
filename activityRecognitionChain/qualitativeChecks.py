# This file is responsible for checking the paper's statements on window size, feature types and raw frames against our values
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Qualitative Checks
class qualitativeChecks:
    def checkQualitativeClaims(self, featureTable, windowTable, classifierTable, selectionTable, referenceTable):
        """
        This method tests the paper's qualitative statements on window size, feature types, raw frames, the best classifier and the best feature count against our values

        Arguments
        =========
        featureTable : Section 5.2 averages
        windowTable : Section 5.3 averages
        classifierTable : Section 5.6 averages
        selectionTable : Section 5.7 averages
        referenceTable : Paper reference table with the statements

        Output
        ======
        Dataframe with one row per checked statement and case
        """
        # Read The Statements
        statementTable = referenceTable[referenceTable["referenceType"] == "statement"].set_index("referenceId")

        # Check The Statements Section By Section
        claimRows = self.checkWindowClaim(windowTable, statementTable)
        claimRows += self.checkFeatureClaims(featureTable, statementTable)
        claimRows += self.checkClassifierClaim(classifierTable, statementTable)
        claimRows += self.checkSelectionClaim(selectionTable, statementTable)
        claimTable = pd.DataFrame(claimRows)
        writeCsv(claimTable, self.resultsDirectory / "qualitativeChecks.csv")
        return claimTable

    def checkWindowClaim(self, windowTable, statementTable):
        """
        This method checks where precision peaks and whether recall never rises as the window grows

        Arguments
        =========
        windowTable : Section 5.3 averages
        statementTable : Paper statements indexed by reference id

        Output
        ======
        List of claim rows, one per step variant and scheme
        """
        claimRows = []
        # Check Where Precision Peaks And Whether Recall Never Rises As The Window Grows
        for (stepName, schemeName), sweepRows in windowTable.groupby(["stepVariant", "scheme"], sort=False):
            sweepRows = sweepRows.sort_values("windowSeconds")
            precisionValues = sweepRows["precisionAll"].to_numpy()
            precisionOrder = np.argsort(-precisionValues, kind="stable")
            peakWindow = sweepRows["windowSeconds"].iloc[precisionOrder[0]]
            runnerUpWindow = sweepRows["windowSeconds"].iloc[precisionOrder[1]]
            runnerUpGap = 100 * (precisionValues[precisionOrder[0]] - precisionValues[precisionOrder[1]])
            recallPeakWindow = sweepRows["windowSeconds"].iloc[int(np.argmax(sweepRows["recallAll"].to_numpy()))]
            recallNeverRises = bool(np.all(np.diff(sweepRows["recallAll"].to_numpy()) <= 0))
            claimRows.append({
                "referenceId": "5.3-statement", "section": "5.3", "page": statementTable.loc["5.3-statement", "page"], "scheme": schemeName, "variant": stepName,
                "paperStatement": statementTable.loc["5.3-statement", "paperSentence"],
                "ourFinding": f"precision peaks at Ws = {peakWindow:g} s, {runnerUpGap:.2f} points above the runner-up at Ws = {runnerUpWindow:g} s; "
                              f"recall peaks at Ws = {recallPeakWindow:g} s and {'never rises' if recallNeverRises else 'rises somewhere'} as Ws grows",
                "precisionChange": None, "recallChange": None,
                "firstPartHolds": bool(peakWindow == self.windowSeconds), "secondPartHolds": recallNeverRises,
                "holds": bool(peakWindow == self.windowSeconds and recallNeverRises),
            })
        return claimRows

    def checkFeatureClaims(self, featureTable, statementTable):
        """
        This method checks the paper's statements on the feature types and on the raw frames

        Arguments
        =========
        featureTable : Section 5.2 averages
        statementTable : Paper statements indexed by reference id

        Output
        ======
        List of claim rows, two per sensor set and scheme
        """
        claimRows = []

        # Check That Mean And Variance Beat The Paper's Feature Types And Beat Their Combination With The Others
        for (sensorSetName, schemeName), typeRows in featureTable.groupby(["sensorSet", "scheme"], sort=False):
            typeRows = typeRows.set_index("featureSet")
            claimRows.append(self.buildFeatureTypeRow(typeRows, sensorSetName, schemeName, statementTable))

            # Check That Raw Frames Trade Precision For Recall Against Mean And Variance
            claimRows.append(self.buildRawFrameRow(typeRows, sensorSetName, schemeName, statementTable))
        return claimRows

    def buildFeatureTypeRow(self, typeRows, sensorSetName, schemeName, statementTable):
        """
        This method checks that mean and variance beat the paper's feature types and beat their combination with the others

        Arguments
        =========
        typeRows : Section 5.2 rows of one sensor set and scheme, indexed by feature set
        sensorSetName : Name of the sensor set
        schemeName : Evaluation scheme
        statementTable : Paper statements indexed by reference id

        Output
        ======
        Claim row of the feature type statement
        """
        paperFeatureSets = [featureSetName for featureSetName, _, _ in self.featureSets if featureSetName != "RawWindow"]
        variantFeatureSets = [featureSetName for featureSetName, _, _ in self.featureSets if featureSetName not in paperFeatureSets]
        bestByPrecision = typeRows.loc[paperFeatureSets, "precisionAll"].idxmax()
        precisionChange = 100 * (typeRows.loc["All", "precisionAll"] - typeRows.loc["VerySimple", "precisionAll"])
        recallChange = 100 * (typeRows.loc["All", "recallAll"] - typeRows.loc["VerySimple", "recallAll"])
        return {
            "referenceId": "5.2-statement", "section": "5.2", "page": statementTable.loc["5.2-statement", "page"], "scheme": schemeName, "variant": sensorSetName,
            "paperStatement": statementTable.loc["5.2-statement", "paperSentence"],
            "ourFinding": f"best precision among the paper's feature types ({', '.join(paperFeatureSets)}; {', '.join(variantFeatureSets)} left out): "
                          f"{bestByPrecision}; All minus VerySimple: precision {precisionChange:+.1f} points, recall {recallChange:+.1f} points",
            "precisionChange": precisionChange, "recallChange": recallChange,
            "firstPartHolds": bool(bestByPrecision == "VerySimple"), "secondPartHolds": bool(precisionChange < 0),
            "holds": bool(bestByPrecision == "VerySimple" and precisionChange < 0),
        }

    def buildRawFrameRow(self, typeRows, sensorSetName, schemeName, statementTable):
        """
        This method checks that raw frames trade precision for recall against mean and variance

        Arguments
        =========
        typeRows : Section 5.2 rows of one sensor set and scheme, indexed by feature set
        sensorSetName : Name of the sensor set
        schemeName : Evaluation scheme
        statementTable : Paper statements indexed by reference id

        Output
        ======
        Claim row of the raw frame statement
        """
        rawPrecisionChange = 100 * (typeRows.loc["Raw", "precisionAll"] - typeRows.loc["VerySimple", "precisionAll"])
        rawRecallChange = 100 * (typeRows.loc["Raw", "recallAll"] - typeRows.loc["VerySimple", "recallAll"])
        return {
            "referenceId": "5.2-rawStatement", "section": "5.2", "page": statementTable.loc["5.2-rawStatement", "page"], "scheme": schemeName, "variant": sensorSetName,
            "paperStatement": statementTable.loc["5.2-rawStatement", "paperSentence"],
            "ourFinding": f"Raw per frame minus VerySimple: recall {rawRecallChange:+.1f} points, precision {rawPrecisionChange:+.1f} points",
            "precisionChange": rawPrecisionChange, "recallChange": rawRecallChange,
            "firstPartHolds": bool(rawRecallChange > 0), "secondPartHolds": bool(rawPrecisionChange < 0),
            "holds": bool(rawRecallChange > 0 and rawPrecisionChange < 0),
        }
