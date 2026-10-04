# This file is responsible for rerunning the crossing rate configurations with the zero line of the training participant
# %%
# Importing Libraries
import pandas as pd

from activityRecognitionChain.chainEvaluation import chainEvaluator
from common.csvWriters import writeCsv


# %%
# Zero Line Sensitivity
class zeroLineSensitivity:
    def testZeroLineSensitivity(self, featureTable, selectionTable):
        """
        This method reruns the person-independent rounds of every configuration with crossing rates, taking the zero line of both recordings from the training participant ( extra d )

        Arguments
        =========
        featureTable : Section 5.2 averages
        selectionTable : Section 5.7 averages

        Output
        ======
        Dataframe comparing the recording mean zero line with the training participant's zero line
        """
        # Pair Every Configuration With Crossing Rates With Its Published Rows
        comparedSets = self.pairCrossingConfigurations(featureTable, selectionTable)

        # Evaluate Each Round With The Channel Means Of Its Training Participant As The Zero Line
        roundRows = self.runCentredRounds(comparedSets)

        # Average The Rounds And Compare Them With The Published Person-Independent Rows
        sensitivityTable = pd.DataFrame(self.compareZeroLines(comparedSets, roundRows))
        sensitivityTable["selectedFeatures"] = sensitivityTable["selectedFeatures"].astype("Int64")
        writeCsv(sensitivityTable, self.resultsDirectory / "extraZeroLineSensitivity.csv")
        self.recordParameter("protocol", "zeroLineAlternative", "the training participant's channel means as the zero line of both recordings")
        return sensitivityTable

    def pairCrossingConfigurations(self, featureTable, selectionTable):
        """
        This method pairs every configuration with crossing rates with its published rows

        Arguments
        =========
        featureTable : Section 5.2 averages
        selectionTable : Section 5.7 averages

        Output
        ======
        List of ( configuration name , configuration , published rows ) tuples
        """
        # Pair Every Configuration With Crossing Rates With Its Published Rows
        comparedSets = [
            (f"{sensorSetName}.{featureSetName}", self.buildConfiguration(sensorKeys, featureType=featureSetName, schemeNames=["personIndependent"]),
             featureTable[(featureTable["sensorSet"] == sensorSetName) & (featureTable["featureSet"] == featureSetName)])
            for sensorSetName, sensorKeys in self.sensorSets.items() for featureSetName in self.crossingFeatureSets
        ]
        comparedSets.append((
            "allSensors.All.mRMR", self.buildConfiguration(self.allSensors, featureType="All", featureCounts=self.featureCounts, schemeNames=["personIndependent"]),
            selectionTable,
        ))
        return comparedSets

    def runCentredRounds(self, comparedSets):
        """
        This method runs the person-independent rounds of every configuration with the channel means of the training participant as the zero line

        Arguments
        =========
        comparedSets : Output of pairCrossingConfigurations

        Output
        ======
        Dictionary from configuration name to its round rows
        """
        # Evaluate Each Round With The Channel Means Of Its Training Participant As The Zero Line
        roundRows = {configurationName: [] for configurationName, _, _ in comparedSets}
        for trainSubject in self.evaluator.loader.subjects:
            centredEvaluator = chainEvaluator(self.dataRoot, randomSeed=self.randomSeed, nJobs=self.nJobs, zeroLineSubject=trainSubject)
            for configurationName, configuration, _ in comparedSets:
                subjectRows, _ = centredEvaluator.evaluateConfiguration(configuration)
                roundRows[configurationName] += [roundRow for roundRow in subjectRows if roundRow["trainSubject"] == trainSubject]
            print(f"[ CHAIN : zeroLineSensitivity training subject {trainSubject} ] : done")
        return roundRows

    def compareZeroLines(self, comparedSets, roundRows):
        """
        This method averages the centred rounds and compares them with the published person-independent rows

        Arguments
        =========
        comparedSets : Output of pairCrossingConfigurations
        roundRows : Output of runCentredRounds

        Output
        ======
        List with one comparison row per configuration and feature count
        """
        # Average The Rounds And Compare Them With The Published Person-Independent Rows
        sensitivityRows = []
        for configurationName, configuration, publishedRows in comparedSets:
            centredTable = self.averageRounds(self.labelRounds("zeroLineSensitivity", configurationName, configuration, roundRows[configurationName]))
            for centredRow in centredTable.to_dict("records"):
                matchingRows = publishedRows[publishedRows["scheme"] == "personIndependent"]
                if configuration.get("featureCounts"):
                    matchingRows = matchingRows[matchingRows["selectedFeatures"] == centredRow["variant"]]
                publishedRow = matchingRows.iloc[0]
                sensitivityRow = {
                    "configuration": configurationName, "selectedFeatures": centredRow["variant"] if configuration.get("featureCounts") else None,
                    "scheme": "personIndependent", "rounds": centredRow["rounds"],
                }
                for metricName in ["precisionAll", "recallAll", "precisionNonNull", "recallNonNull"]:
                    sensitivityRow[f"{metricName}Published"] = publishedRow[metricName]
                    sensitivityRow[f"{metricName}TrainingZeroLine"] = centredRow[metricName]
                    sensitivityRow[f"{metricName}Change"] = centredRow[metricName] - publishedRow[metricName]
                sensitivityRows.append(sensitivityRow)
        return sensitivityRows
