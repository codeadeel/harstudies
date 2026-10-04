# This file is responsible for running one configuration, labelling its rounds and averaging them for every section
# %%
# Importing Libraries
import pandas as pd


# %%
# Configuration Runner
class configurationRunner:
    def recordParameter(self, sectionName, parameterName, parameterValue):
        """
        This method stores one setting or check value for runParameters.csv

        Arguments
        =========
        sectionName : Section the value belongs to
        parameterName : Name of the value
        parameterValue : The value itself

        Output
        ======
        None
        """
        # Keep The Parameter Row
        self.parameterRows.append({"section": sectionName, "parameter": parameterName, "value": parameterValue})

    def runConfiguration(self, sectionName, configurationName, configuration, evaluator=None, keepRounds=True):
        """
        This method evaluates one configuration, keeps its round rows and returns its scheme averages

        Arguments
        =========
        sectionName : Section label such as 5.1
        configurationName : Short name of the configuration within the section
        configuration : Configuration dictionary for chainEvaluator.evaluateConfiguration
        evaluator : Evaluator to use, the main one when None ( default : None )
        keepRounds : Whether the round rows and pooled confusions are kept for the round table and the extras ( default : True )

        Output
        ======
        Dataframe with one row per scheme and variant, averaged over the rounds
        """
        # Run Every Round And Label The Rows
        roundRows, pooledConfusions = (evaluator or self.evaluator).evaluateConfiguration(configuration)
        roundTable = self.labelRounds(sectionName, configurationName, configuration, roundRows)

        # Keep The Rounds And Confusions Of The Main Evaluation
        if keepRounds:
            self.roundTables.append(roundTable)
            for (schemeName, variantName, testSubject), confusionCounts in pooledConfusions.items():
                self.pooledConfusions[(sectionName, configurationName, schemeName, variantName, testSubject)] = confusionCounts
        print(f"[ CHAIN : {sectionName} {configurationName} ] : done")
        return self.averageRounds(roundTable)

    def labelRounds(self, sectionName, configurationName, configuration, roundRows):
        """
        This method turns round rows into a table that starts with the section, configuration and setting columns

        Arguments
        =========
        sectionName : Section label such as 5.1
        configurationName : Short name of the configuration within the section
        configuration : Configuration dictionary the rounds were run with
        roundRows : Round rows of chainEvaluator.evaluateConfiguration

        Output
        ======
        Dataframe of the labelled round rows
        """
        # Put The Label Columns In Front Of The Round Columns
        roundTable = pd.DataFrame(roundRows)
        labelValues = [
            sectionName, configurationName, "+".join(configuration["sensorKeys"]),
            configuration["featureType"] if configuration["classifierName"] != "hiddenMarkovModel" else "rawSequence",
            configuration["classifierName"], configuration["windowSeconds"], configuration["stepSeconds"],
        ]
        for columnName, columnValue in reversed(list(zip(self.labelColumnNames, labelValues))):
            roundTable.insert(0, columnName, columnValue)
        return roundTable

    def averageRounds(self, roundTable):
        """
        This method averages the round metrics per scheme and variant, over all rounds and per test participant

        Arguments
        =========
        roundTable : Round rows of one configuration

        Output
        ======
        Dataframe with one row per scheme and variant
        """
        # Average Over All Cross Validation Rounds, As The Paper States
        groupColumns = self.labelColumnNames + ["scheme", "variant"]
        averageRows = []
        for groupValues, groupTable in roundTable.groupby(groupColumns, sort=False, dropna=False):
            averageRow = dict(zip(groupColumns, groupValues))
            averageRow["rounds"] = len(groupTable)
            averageRow["windowSamples"] = int(groupTable["windowSamples"].iloc[0])
            averageRow["stepSamples"] = int(groupTable["stepSamples"].iloc[0])
            for metricName in self.metricColumns:
                averageRow[metricName] = groupTable[metricName].mean()
            averageRow["detectedGestureSegmentShare"] = groupTable["detectedGestureSegments"].sum() / groupTable["gestureSegments"].sum()

            # Add The Per Participant Averages Of The Main Metrics
            for subjectNumber, subjectTable in groupTable.groupby("testSubject"):
                for metricName in ["precisionAll", "recallAll", "precisionNonNull", "recallNonNull"]:
                    averageRow[f"{metricName}Subject{subjectNumber}"] = subjectTable[metricName].mean()
            averageRows.append(averageRow)
        return pd.DataFrame(averageRows)

    def buildConfiguration(self, sensorKeys, classifierName="kNearestNeighbour", featureType="VerySimple", windowSeconds=None, stepSeconds=None, **extraSettings):
        """
        This method assembles a configuration dictionary with the paper's defaults

        Arguments
        =========
        sensorKeys : Sensors to use
        classifierName : Classifier name ( default : kNearestNeighbour )
        featureType : Feature type ( default : VerySimple )
        windowSeconds : Window length, the paper's window when None ( default : None )
        stepSeconds : Window step, the paper's step when None ( default : None )
        extraSettings : Further configuration entries such as featureCounts

        Output
        ======
        Configuration dictionary
        """
        # Fill In The Paper Defaults
        return {
            "sensorKeys": sensorKeys, "classifierName": classifierName, "featureType": featureType,
            "windowSeconds": self.windowSeconds if windowSeconds is None else windowSeconds,
            "stepSeconds": self.stepSeconds if stepSeconds is None else stepSeconds, **extraSettings,
        }
