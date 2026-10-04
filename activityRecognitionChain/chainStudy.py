# This file is responsible for reproducing the case study of Bulling, Blanke and Schiele ( 2014 ) section by section and writing its tables and figures
# %%
# Importing Libraries
from activityRecognitionChain.barAndSweepPlots import barAndSweepPlots
from activityRecognitionChain.basicChainSections import basicChainSections
from activityRecognitionChain.basicConfusions import basicConfusions
from activityRecognitionChain.bestChoiceChecks import bestChoiceChecks
from activityRecognitionChain.chainEvaluation import chainEvaluator
from activityRecognitionChain.comparisonRows import comparisonRows
from activityRecognitionChain.configurationRunner import configurationRunner
from activityRecognitionChain.dataDescription import dataDescription
from activityRecognitionChain.eventAndNullLoss import eventAndNullLoss
from activityRecognitionChain.filledFrames import filledFrames
from activityRecognitionChain.paperComparison import paperComparison
from activityRecognitionChain.paperReference import paperReference
from activityRecognitionChain.qualitativeChecks import qualitativeChecks
from activityRecognitionChain.runControl import runControl
from activityRecognitionChain.runParameters import runParameters
from activityRecognitionChain.sectionFigures import sectionFigures
from activityRecognitionChain.sensorAndClassifierSections import sensorAndClassifierSections
from activityRecognitionChain.settingRows import settingRows
from activityRecognitionChain.zeroLineSensitivity import zeroLineSensitivity
from common.projectPaths import createPartDirectories


# %%
# Chain Study
class chainStudy(
    configurationRunner, runControl, dataDescription, filledFrames, basicChainSections, sensorAndClassifierSections,
    eventAndNullLoss, basicConfusions, zeroLineSensitivity, paperReference, paperComparison, comparisonRows,
    qualitativeChecks, bestChoiceChecks, barAndSweepPlots, sectionFigures, runParameters, settingRows,
):
    def __init__(self, dataRoot, resultsRoot, randomSeed=42, nJobs=4):
        """
        This class initializes the case study with its evaluator, output folders, sensor sets and section settings

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets
        resultsRoot : Root directory that receives every part's results
        randomSeed : Seed for every random operation ( default : 42 )
        nJobs : Thread and worker count for the native pools, the SVM grid search, the k-NN queries and the HMM class fits ( default : 4 )

        Output
        ======
        None
        """
        # Store The Run Settings
        self.dataRoot = dataRoot
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.evaluator = chainEvaluator(dataRoot, randomSeed=randomSeed, nJobs=nJobs)
        self.resultsDirectory, self.figuresDirectory = createPartDirectories(resultsRoot, "activityRecognitionChain")

        # Store The Sensor Sets ( IMU 1 Hand, 2 Lower Arm, 3 Upper Arm )
        self.allSensors = ["acc_1", "gyr_1", "acc_2", "gyr_2", "acc_3", "gyr_3"]
        self.sensorSets = {"handAccelerometer": ["acc_1"], "allSensors": self.allSensors}
        self.placementSets = {
            "hand": ["acc_1", "gyr_1"], "lowerArm": ["acc_2", "gyr_2"], "upperArm": ["acc_3", "gyr_3"],
            "handLowerArm": ["acc_1", "gyr_1", "acc_2", "gyr_2"], "handUpperArm": ["acc_1", "gyr_1", "acc_3", "gyr_3"],
            "lowerUpperArm": ["acc_2", "gyr_2", "acc_3", "gyr_3"], "allPlacements": self.allSensors,
        }
        self.modalitySets = {
            "accHand": ["acc_1"], "accLowerArm": ["acc_2"], "accUpperArm": ["acc_3"],
            "gyrHand": ["gyr_1"], "gyrLowerArm": ["gyr_2"], "gyrUpperArm": ["gyr_3"],
            "accHandLowerArm": ["acc_1", "acc_2"], "gyrHandLowerArm": ["gyr_1", "gyr_2"],
            "accHandUpperArm": ["acc_1", "acc_3"], "gyrHandUpperArm": ["gyr_1", "gyr_3"],
            "accLowerUpperArm": ["acc_2", "acc_3"], "gyrLowerUpperArm": ["gyr_2", "gyr_3"],
            "accAll": ["acc_1", "acc_2", "acc_3"], "gyrAll": ["gyr_1", "gyr_2", "gyr_3"],
        }

        # Store The Section Settings From The Paper ( Main ) And The Toolbox ( Sensitivity )
        self.windowSeconds = 1.0
        self.stepSeconds = 1.0
        self.toolboxStepSeconds = 0.1
        self.toolboxSweepStepSeconds = 0.05
        self.featureSets = [
            ("Raw", "Raw", True), ("RawWindow", "Raw", False), ("VerySimple", "VerySimple", False),
            ("Simple", "Simple", False), ("FFT", "FFT", False), ("All", "All", False),
        ]
        self.sweepWindowSeconds = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.classifierOrder = ["linearDiscriminant", "gaussianNaiveBayes", "supportVectorMachine", "hiddenMarkovModel", "adaBoost", "kNearestNeighbour"]
        self.toolboxStepClassifiers = ["linearDiscriminant", "gaussianNaiveBayes", "adaBoost", "kNearestNeighbour"]
        self.basicClassifiers = ["kNearestNeighbour", "nearestClassCentroid"]
        self.basicSensitivityClassifiers = ["toolboxNearestNeighbours"]
        self.featureCounts = [1, 5, 10, 20, 25, 50, 150, 200, 250, 300]
        self.paperActivitySeconds = (2.0, 8.0)
        self.fillReplacements = ["mean", "interpolate"]
        self.crossingFeatureSets = ["Simple", "All"]
        self.labelColumnNames = ["section", "configuration", "sensors", "featureType", "classifier", "windowSeconds", "stepSeconds"]
        self.metricColumns = [
            "precisionAll", "recallAll", "precisionNonNull", "recallNonNull", "precisionAllDefined", "precisionNonNullDefined",
            "classesNeverPredicted", "eventPrecisionAll", "eventRecallAll", "eventPrecisionNonNull", "eventRecallNonNull",
        ]
        self.averagingColumns = {
            "all12ZeroPrecision": ("precisionAll", "recallAll"), "nonNull11ZeroPrecision": ("precisionNonNull", "recallNonNull"),
            "all12DefinedPrecision": ("precisionAllDefined", "recallAll"), "nonNull11DefinedPrecision": ("precisionNonNullDefined", "recallNonNull"),
        }

        # Prepare The Collected Outputs
        self.parameterRows = []
        self.roundTables = []
        self.pooledConfusions = {}
