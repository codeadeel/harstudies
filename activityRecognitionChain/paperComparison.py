# This file is responsible for pairing every printed paper value with our values and marking repeated comparisons
# %%
# Importing Libraries
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Paper Comparison
class paperComparison:
    def comparePaper(self, referenceTable, sectionTables):
        """
        This method compares our values with every printed paper value under four class averaging conventions and both steps, adding the nearest class centroid as a second row in 5.1

        Arguments
        =========
        referenceTable : Paper reference table
        sectionTables : Dictionary from section label to its averaged table

        Output
        ======
        Dataframe of the comparison
        """
        # Pair Every Printed Value With Our Values
        comparisonRows = []
        for referenceRow in referenceTable[referenceTable["referenceType"] == "value"].itertuples(index=False):
            comparedClassifiers = [(referenceRow.classifier, "main")]
            if referenceRow.section == "5.1":
                comparedClassifiers.append(("nearestClassCentroid", "secondRow"))
                comparedClassifiers += [(classifierName, "sensitivityRow") for classifierName in self.basicSensitivityClassifiers]
            for classifierName, comparedAs in comparedClassifiers:
                for stepName in ["paperStep", "toolboxStep"]:
                    ourRow = self.findOurRow(referenceRow, sectionTables, stepName, classifierName)
                    if ourRow is None:
                        continue
                    comparisonRows.append(self.buildComparisonRow(referenceRow, ourRow, classifierName, comparedAs, stepName))
        comparisonTable = pd.DataFrame(comparisonRows)

        # Mark Comparisons That Repeat An Earlier One ( Same Printed Pair And Same Our Values )
        comparisonTable["duplicateOf"] = self.findRepeatedComparisons(comparisonTable)
        writeCsv(comparisonTable, self.resultsDirectory / "paperComparison.csv")

        # Plot Ours Against The Paper Per Distinct Printed Value, One Panel Per Metric, At The Paper Step
        self.plotPaperComparison(comparisonTable)
        return comparisonTable

    def findOurRow(self, referenceRow, sectionTables, stepName, classifierName):
        """
        This method finds our averaged row that matches one printed paper value

        Arguments
        =========
        referenceRow : Value row of the paper reference table
        sectionTables : Dictionary from section label to its averaged table
        stepName : paperStep or toolboxStep, used where the section has both
        classifierName : Classifier whose row is wanted

        Output
        ======
        Matching row as a series, or None when the section has no row for that step and classifier
        """
        # Filter The Section Table By Scheme And Step
        sectionTable = sectionTables[referenceRow.section]
        candidateRows = sectionTable[sectionTable["scheme"] == referenceRow.scheme]
        if "stepVariant" in candidateRows.columns:
            candidateRows = candidateRows[candidateRows["stepVariant"] == stepName]
        elif stepName != "paperStep":
            return None

        # Filter By The Configuration The Paper Names
        if referenceRow.section in ("5.1", "5.6"):
            candidateRows = candidateRows[(candidateRows["sensorSet"] == referenceRow.configuration) & (candidateRows["classifier"] == classifierName)]
        elif referenceRow.section == "5.4":
            candidateRows = candidateRows[candidateRows["placement"] == referenceRow.configuration]
        elif referenceRow.section == "5.5" and referenceRow.configuration == "bestGyroscopeOnly":
            gyroscopeRows = candidateRows[candidateRows["modality"] == "gyroscope"]
            candidateRows = gyroscopeRows.loc[[gyroscopeRows["precisionAll"].idxmax()]]
        elif referenceRow.section == "5.5":
            candidateRows = candidateRows[candidateRows["modalitySet"] == referenceRow.configuration]
        elif referenceRow.section == "5.7":
            candidateRows = candidateRows[candidateRows["selectedFeatures"] == int(referenceRow.selectedFeatures)]
        return candidateRows.iloc[0] if len(candidateRows) == 1 else None

    def findRepeatedComparisons(self, comparisonTable):
        """
        This method names, for every comparison, the earlier reference that has the same printed pair and the same values of ours

        Arguments
        =========
        comparisonTable : Comparison rows in the order of the reference table

        Output
        ======
        List with the reference id of the first identical comparison, or an empty text
        """
        # Find The First Reference With The Same Printed Pair And The Same Our Values
        repeatKeys = ["comparedClassifier", "scheme", "stepVariant", "paperPrecision", "paperRecall", "ourPrecision_all12ZeroPrecision", "ourRecall_all12ZeroPrecision"]
        firstReference = {}
        repeatedFrom = []
        for comparisonRow in comparisonTable.itertuples(index=False):
            repeatKey = tuple(getattr(comparisonRow, keyName) for keyName in repeatKeys)
            repeatedFrom.append(firstReference.get(repeatKey, ""))
            firstReference.setdefault(repeatKey, comparisonRow.referenceId)
        return repeatedFrom
