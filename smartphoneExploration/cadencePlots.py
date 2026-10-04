# This file is responsible for plotting the cadence estimates of every participant and the pooled raw spectrum of one participant
# %%
# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np

from common.plotDefaults import categoricalColours, saveFigure


# %%
# Cadence Plots
class cadencePlots:
    def plotCadenceComparison(self, cadenceTable):
        """
        This method draws the raw and the SSA cadence estimates of every participant

        Arguments
        =========
        cadenceTable : Dataframe comparing the raw and the SSA step frequencies per participant

        Output
        ======
        None
        """
        estimator = self.cadenceEstimator

        # Plot The Raw And SSA Estimates Per Participant
        figure, axis = plt.subplots(figsize=(13, 6))
        methodColours = categoricalColours(5)
        participantPositions = np.arange(len(cadenceTable))
        axis.axhspan(estimator.lowFrequency, estimator.highFrequency, color="0.93", zorder=0, label=f"cadence band {estimator.lowFrequency:g}-{estimator.highFrequency:g} Hz")
        axis.errorbar(
            participantPositions - 0.2, cadenceTable["fftStepsPerSecond"],
            yerr=[cadenceTable["fftStepsPerSecond"] - cadenceTable["fftBoutMin"], cadenceTable["fftBoutMax"] - cadenceTable["fftStepsPerSecond"]],
            fmt="o", color=methodColours[0], label="raw FFT, median over bouts ( bar: bout range )",
        )
        axis.plot(participantPositions, cadenceTable["acfStepsPerSecond"], "s", color=methodColours[1], label="raw autocorrelation, median over bouts")
        axis.plot(participantPositions + 0.2, cadenceTable["verticalStepsPerSecond"], "D", color=methodColours[4], markersize=4, label="raw vertical acceleration peak ( no doubling ), median over bouts")
        axis.plot(participantPositions, cadenceTable["dominantPeakHz"], "x", color=methodColours[2], label="tallest raw magnitude peak in the band ( naive )")
        axis.plot(participantPositions, cadenceTable["ssaStepsPerSecond"], "^", color=methodColours[3], label="SSA sine fit on window means")
        axis.axhline(cadenceTable["ssaNyquistHz"].iloc[0], color="black", linestyle="--", linewidth=1, label=f"Nyquist of the {self.loader.windowStepSeconds} s spacing")
        axis.set_xticks(participantPositions, cadenceTable["subject"])
        axis.set_xlabel("Participant")
        axis.set_ylabel("Steps per second ( Hz )")
        axis.set_title(f"Walking cadence: raw {estimator.samplingRate:g} Hz signal versus SSA on window means")

        # Order The Legend And Save The Figure
        self.placeCadenceLegend(axis)
        saveFigure(figure, self.figuresDirectory / "cadenceComparison.png")

    def placeCadenceLegend(self, axis):
        """
        This method orders the legend from the raw estimates to the reference lines and places it beside the plot

        Arguments
        =========
        axis : Matplotlib axis of the cadence comparison

        Output
        ======
        None
        """
        # Order The Legend From The Raw Estimates To The Reference Lines
        legendHandles, legendLabels = axis.get_legend_handles_labels()
        labelPrefixes = ["raw FFT", "raw autocorrelation", "raw vertical", "tallest raw", "SSA sine fit", "Nyquist", "cadence band"]
        legendOrder = [
            next(labelIndex for labelIndex, legendLabel in enumerate(legendLabels) if legendLabel.startswith(labelPrefix))
            for labelPrefix in labelPrefixes
        ]
        axis.legend([legendHandles[orderIndex] for orderIndex in legendOrder], [legendLabels[orderIndex] for orderIndex in legendOrder],
                    loc="center left", bbox_to_anchor=(1.01, 0.5))

    def plotRawSpectrum(self, participantSpectrum):
        """
        This method draws the pooled raw spectrum and the cadence estimates of one participant

        Arguments
        =========
        participantSpectrum : Frequencies, pooled density, tallest peak, pooled cadence and cadence row of the participant

        Output
        ======
        None
        """
        estimator = self.cadenceEstimator
        methodColours = categoricalColours(5)

        # Plot The Notebook Participant's Pooled Raw Spectrum
        frequencies, pooledDensity, dominantPeak, pooledCadence, participantRow = participantSpectrum
        shownMask = frequencies <= 2 * estimator.highFrequency
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.axvspan(estimator.lowFrequency, estimator.highFrequency, color="0.93", zorder=0, label="cadence band")
        axis.axvspan(participantRow["fftBoutMin"], participantRow["fftBoutMax"], color=methodColours[1], alpha=0.25, zorder=0,
                     label=f"per-bout step estimates {participantRow['fftBoutMin']:.3f}-{participantRow['fftBoutMax']:.3f} ( median {participantRow['fftStepsPerSecond']:.3f} )")
        axis.plot(frequencies[shownMask], pooledDensity[shownMask], color=methodColours[0], label="Welch PSD pooled over the walking bouts")
        axis.axvline(pooledCadence / 2, color=methodColours[2], linestyle=":", label=f"harmonic-sum stride estimate ( half the step estimate ) {pooledCadence / 2:.3f} Hz")
        axis.axvline(pooledCadence, color=methodColours[1], linestyle="--", label=f"harmonic-sum step estimate of the pooled PSD {pooledCadence:.3f} Hz")
        axis.plot([dominantPeak], [np.interp(dominantPeak, frequencies, pooledDensity)], "x", color="black", markersize=10, label=f"tallest peak in the band {dominantPeak:.3f} Hz")
        axis.set_xlabel("Frequency ( Hz )")
        axis.set_ylabel("Power spectral density")
        axis.set_title(f"Participant {self.ssaParticipant}: raw body acceleration magnitude while walking")
        axis.legend()
        saveFigure(figure, self.figuresDirectory / f"rawSpectrumParticipant{self.ssaParticipant}.png")
