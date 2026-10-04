# This file is responsible for sections 5.4 to 5.7 of the paper: placements, modalities, classifiers and feature selection
# %%
# Importing Libraries
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Sensor And Classifier Sections
class sensorAndClassifierSections:
    def comparePlacements(self):
        """
        This method compares sensor placements with accelerometer and gyroscope together ( section 5.4 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Every Placement Set
        sectionTables = []
        for placementName, sensorKeys in self.placementSets.items():
            sectionTable = self.runConfiguration("5.4", placementName, self.buildConfiguration(sensorKeys))
            sectionTable.insert(2, "placement", placementName)
            sectionTables.append(sectionTable)
        placementTable = pd.concat(sectionTables, ignore_index=True)
        writeCsv(placementTable, self.resultsDirectory / "section54Placements.csv")
        return placementTable

    def compareModalities(self):
        """
        This method compares accelerometers and gyroscopes per placement and combined ( section 5.5 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Every Modality Set
        sectionTables = []
        for modalityName, sensorKeys in self.modalitySets.items():
            sectionTable = self.runConfiguration("5.5", modalityName, self.buildConfiguration(sensorKeys))
            sectionTable.insert(2, "modalitySet", modalityName)
            sectionTable.insert(3, "modality", "accelerometer" if modalityName.startswith("acc") else "gyroscope")
            sectionTables.append(sectionTable)
        modalityTable = pd.concat(sectionTables, ignore_index=True)
        writeCsv(modalityTable, self.resultsDirectory / "section55Modalities.csv")
        return modalityTable

    def compareClassifiers(self):
        """
        This method compares LDA, naive Bayes, SVM, the left-right HMM, AdaBoost and 1-NN for both sensor sets at the paper's step, and the faster classifiers at the toolbox's step ( section 5.6 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Every Classifier At The Paper's Step And The Faster Ones At The Toolbox Step
        sectionTables = []
        svmTables = []
        for stepName, stepSeconds in [("paperStep", self.stepSeconds), ("toolboxStep", self.toolboxStepSeconds)]:
            for sensorSetName, sensorKeys in self.sensorSets.items():
                for classifierName in self.classifierOrder if stepName == "paperStep" else self.toolboxStepClassifiers:
                    configuration = self.buildConfiguration(sensorKeys, classifierName=classifierName, stepSeconds=stepSeconds)
                    sectionTable = self.runConfiguration("5.6", f"{sensorSetName}.{classifierName}.{stepName}", configuration)
                    sectionTable.insert(2, "sensorSet", sensorSetName)
                    sectionTable.insert(3, "stepVariant", stepName)
                    sectionTables.append(sectionTable)

                    # Keep The Parameters The SVM Search Chose In Every Round
                    if classifierName == "supportVectorMachine":
                        roundTable = self.roundTables[-1]
                        svmTables.append(roundTable[["sensors", "scheme", "round", "testSubject", "trainSubject", "heldOutRepetition", "svmCost", "svmGamma"]].assign(sensorSet=sensorSetName))
        classifierTable = pd.concat(sectionTables, ignore_index=True)
        svmTable = pd.concat(svmTables, ignore_index=True)
        writeCsv(classifierTable, self.resultsDirectory / "section56Classifiers.csv")
        writeCsv(svmTable, self.resultsDirectory / "section56SvmParameters.csv")

        # Count The Rounds Whose Chosen SVM Parameters Sit On An Edge Of The Grid
        factory = self.evaluator.factory
        for sensorSetName, sensorRows in svmTable.groupby("sensorSet", sort=False):
            onCostEdge = sensorRows["svmCost"].isin([min(factory.svmCostGrid), max(factory.svmCostGrid)])
            onGammaEdge = sensorRows["svmGamma"].isin([min(factory.svmGammaGrid), max(factory.svmGammaGrid)])
            self.recordParameter("classifiers", f"svmEdgeRounds {sensorSetName}", f"{int((onCostEdge | onGammaEdge).sum())} of {len(sensorRows)}")
        return classifierTable

    def selectFeatures(self):
        """
        This method ranks the All feature pool of every sensor with mRMR in every training round and evaluates 1-NN on each ranking prefix ( section 5.7 )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the section averages
        """
        # Evaluate Every Prefix Length Of The mRMR Ranking
        configuration = self.buildConfiguration(self.allSensors, featureType="All", featureCounts=self.featureCounts)
        selectionTable = self.runConfiguration("5.7", "allSensors.All.mRMR", configuration)
        selectionTable = selectionTable.rename(columns={"variant": "selectedFeatures"})
        writeCsv(selectionTable, self.resultsDirectory / "section57FeatureSelection.csv")

        # Keep The Top Features Of Every Round For Inspection
        roundTable = self.roundTables[-1]
        topTable = roundTable[roundTable["variant"] == max(self.featureCounts)][["scheme", "round", "testSubject", "heldOutRepetition", "topFeatures"]]
        writeCsv(topTable, self.resultsDirectory / "section57TopFeatures.csv")
        return selectionTable
