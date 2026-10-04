# This file is responsible for the window selection pilot on SONAR: the settings and the steps from the windows to the figures
# %%
# Importing Libraries
from common.plotDefaults import applyPlotDefaults
from common.projectPaths import createPartDirectories
from common.randomSeeds import limitThreads, seedEverything
from windowSelectionPilot.pilotFolds import pilotFolds
from windowSelectionPilot.pilotPlots import pilotPlots
from windowSelectionPilot.pilotScoring import pilotScoring
from windowSelectionPilot.pilotSelection import pilotSelection
from windowSelectionPilot.pilotTables import pilotTables
from windowSelectionPilot.pilotWindows import pilotWindows
from windowSelectionPilot.sonarWindows import sonarWindowBuilder


# %%
# Selection Pilot
class selectionPilot(pilotWindows, pilotFolds, pilotSelection, pilotScoring, pilotTables, pilotPlots):
    def __init__(self, dataRoot, resultsRoot, randomSeed=42, nJobs=4, foldCount=5, forestTrees=100):
        """
        This class initializes the pilot with the SONAR window class, the output folders, the folds, the forest size and the selection settings

        Arguments
        =========
        dataRoot : Root directory of the fetched datasets
        resultsRoot : Root directory that receives every part's results
        randomSeed : Seed of the forest and of the random selection ( default : 42 )
        nJobs : Worker processes for the windows and threads for the forest ( default : 4 )
        foldCount : Number of participant grouped folds ( default : 5 )
        forestTrees : Trees of the random forest recognizer ( default : 100 )

        Output
        ======
        None
        """
        # Store The Run Settings
        self.randomSeed = randomSeed
        self.nJobs = nJobs
        self.foldCount = foldCount
        self.forestTrees = forestTrees
        self.builder = sonarWindowBuilder(dataRoot)
        self.resultsDirectory, self.figuresDirectory = createPartDirectories(resultsRoot, "windowSelectionPilot")

        # Store The Selection Settings
        self.budgetPercents = [5, 10, 20, 50, 100]
        self.reportPercent = 20
        self.randomSeedCount = 3
        self.methodNames = ["random", "uniformStride", "changeTrigger", *self.builder.scoreNames, "meanRank"]
        self.changeColumn = self.builder.scoreNames.index("windowChange")

        # Prepare The State Filled While The Pilot Runs
        self.parameterRows = []
        self.windowTable = None
        self.featureMatrix = None
        self.scoreMatrix = None
        self.streamRanges = None
        self.cpuSeconds = None
        self.fullPredictions = None
        self.foldNumbers = None
        self.triggerThresholds = {}
        self.forestPredictionSeconds = 0.0

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

    def prepareRun(self):
        """
        This method fixes the seeds, the thread pools and the plot style before the first step

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Fix The Seeds, Thread Pools And Plot Style
        seedEverything(self.randomSeed)
        limitThreads(self.nJobs)
        applyPlotDefaults()
