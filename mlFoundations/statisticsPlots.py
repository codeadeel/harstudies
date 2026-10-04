# This file is responsible for drawing the histograms, the box plot and the correlation heatmap of the UCI HAR features
# %%
# Importing Libraries
import matplotlib.pyplot as plt

from common.plotDefaults import categoricalColours, saveFigure


# %%
# Statistics Plots
class statisticsPlots:
    def plotHistograms(self):
        """
        This method draws the histograms of the plotted features over all windows

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Plot The Histograms
        barColour = categoricalColours(1)[0]
        figure, axes = plt.subplots(2, 3, figsize=(12, 6))
        for axis, featureName in zip(axes.ravel(), self.histogramFeatures):
            axis.hist(self.featureFrame[featureName], bins=self.histogramBins, color=barColour)
            axis.set_title(featureName, fontsize=9)
            axis.set_xlabel("Feature value")
            axis.set_ylabel("Windows")
        figure.suptitle(f"Histograms of {len(self.histogramFeatures)} UCI HAR features over all {len(self.featureFrame)} windows")
        figure.tight_layout()
        saveFigure(figure, self.figuresDirectory / "histograms.png")

    def plotBoxPlot(self):
        """
        This method draws the box plot of one feature for every activity

        Arguments
        =========
        None

        Output
        ======
        None
        """
        # Plot The Box Plot By Activity
        figure, axis = plt.subplots(figsize=(9, 5))
        axis.boxplot(
            [self.featureFrame.loc[self.featureFrame["Activity"] == activityName, self.boxFeature].to_numpy() for activityName in self.activityOrder],
            tick_labels=self.activityOrder,
        )
        axis.set_ylabel(self.boxFeature)
        axis.set_title(f"{self.boxFeature} by activity, all windows")
        axis.tick_params(axis="x", labelrotation=20)
        saveFigure(figure, self.figuresDirectory / "boxPlot.png")

    def plotCorrelationHeatmap(self, heatmapCorrelation):
        """
        This method draws the correlation heatmap of the heatmap features

        Arguments
        =========
        heatmapCorrelation : Dataframe of the correlations of the heatmap features

        Output
        ======
        None
        """
        # Plot The Correlation Heatmap
        figure, axis = plt.subplots(figsize=(9, 8))
        heatmapImage = axis.imshow(heatmapCorrelation.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axis.set_xticks(range(len(self.heatmapFeatures)), labels=self.heatmapFeatures, rotation=90, fontsize=8)
        axis.set_yticks(range(len(self.heatmapFeatures)), labels=self.heatmapFeatures, fontsize=8)
        for rowIndex in range(len(self.heatmapFeatures)):
            for columnIndex in range(len(self.heatmapFeatures)):
                axis.text(columnIndex, rowIndex, f"{heatmapCorrelation.iat[rowIndex, columnIndex]:.2f}", ha="center", va="center", fontsize=6)
        axis.grid(False)
        figure.colorbar(heatmapImage, ax=axis, label="Pearson correlation")
        axis.set_title(f"Correlation of {len(self.heatmapFeatures)} UCI HAR features over all windows")
        saveFigure(figure, self.figuresDirectory / "correlationHeatmap.png")
