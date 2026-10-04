# This file is responsible for writing the class distribution, the descriptive statistics and the correlation counts of the UCI HAR features
# %%
# Importing Libraries
import numpy as np
import pandas as pd

from common.csvWriters import writeCsv


# %%
# Statistics Sections
class statisticsSections:
    def describeStatistics(self):
        """
        This method writes the class distribution, descriptive statistics, skew and correlation counts, and draws the histograms, box plot and heatmap ( chapters Understanding Data with Statistics and with Visualization )

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the descriptive statistics of the plotted features
        """
        # Write The Class Distribution And The Feature Statistics
        self.writeClassDistribution()
        statisticsTable = self.writeFeatureStatistics()

        # Count Skewed Features And Strongly Correlated Pairs Over All Features
        self.recordCorrelationCounts()

        # Write The Correlations Of The Heatmap Features
        heatmapCorrelation = self.writeHeatmapCorrelation()

        # Plot The Histograms, The Box Plot And The Correlation Heatmap
        self.plotHistograms()
        self.plotBoxPlot()
        self.plotCorrelationHeatmap(heatmapCorrelation)
        return statisticsTable

    def writeClassDistribution(self):
        """
        This method counts the windows of every activity in the training and test windows and writes classDistribution.csv

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Count The Windows Per Activity And Split
        distributionRows = []
        for activityName in self.activityOrder:
            trainCount = int(np.sum(self.trainLabels == activityName))
            testCount = int(np.sum(self.testLabels == activityName))
            distributionRows.append({
                "activity": activityName,
                "trainWindows": trainCount,
                "trainShare": trainCount / len(self.trainLabels),
                "testWindows": testCount,
                "testShare": testCount / len(self.testLabels),
            })
        writeCsv(pd.DataFrame(distributionRows), self.resultsDirectory / "classDistribution.csv")

    def writeFeatureStatistics(self):
        """
        This method describes the plotted features over all windows with their skew and writes featureStatistics.csv

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the descriptive statistics of the plotted features
        """
        # Describe The Plotted Features Over All Windows
        statisticsTable = self.featureFrame[self.histogramFeatures].describe().T
        statisticsTable["skew"] = self.featureFrame[self.histogramFeatures].skew()
        statisticsTable = statisticsTable.rename(columns={"25%": "quartile1", "50%": "median", "75%": "quartile3"}).reset_index(names="feature")
        writeCsv(statisticsTable, self.resultsDirectory / "featureStatistics.csv")
        return statisticsTable

    def recordCorrelationCounts(self):
        """
        This method counts the skewed features and the strongly correlated feature pairs and records the counts for runParameters.csv

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Count Skewed Features And Strongly Correlated Pairs Over All Features
        allFeatures = self.featureFrame[self.featureNames]
        correlationMatrix = np.corrcoef(allFeatures.to_numpy(), rowvar=False)
        pairCorrelations = correlationMatrix[np.triu_indices(len(self.featureNames), k=1)]
        self.recordParameter("statistics", "skewThreshold", self.skewThreshold)
        self.recordParameter("statistics", "skewedFeatures", int((allFeatures.skew().abs() > self.skewThreshold).sum()))
        self.recordParameter("statistics", "correlationThreshold", self.correlationThreshold)
        self.recordParameter("statistics", "featurePairs", len(pairCorrelations))
        self.recordParameter("statistics", "correlatedPairs", int(np.sum(np.abs(pairCorrelations) > self.correlationThreshold)))
        self.recordParameter("statistics", "undefinedCorrelations", int(np.sum(np.isnan(pairCorrelations))))

    def writeHeatmapCorrelation(self):
        """
        This method writes the correlations of the features shown in the heatmap

        Arguments
        =========
        None

        Output
        ======
        Dataframe of the correlations of the heatmap features
        """
        # Write The Correlations Of The Heatmap Features
        heatmapCorrelation = self.featureFrame[self.heatmapFeatures].corr()
        writeCsv(heatmapCorrelation.reset_index(names="feature"), self.resultsDirectory / "featureCorrelation.csv")
        return heatmapCorrelation
