# This file is responsible for drawing one figure per section of the paper with the paper values as markers
# %%
# Importing Libraries


# %%
# Section Figures
class sectionFigures:
    def plotSections(self, sectionTables, referenceTable):
        """
        This method draws one figure per section in the paper's style with the paper values as markers

        Arguments
        =========
        sectionTables : Dictionary from section label to its averaged table
        referenceTable : Paper reference table

        Output
        ======
        None
        """
        valueTable = referenceTable[referenceTable["referenceType"] == "value"]

        # Draw Sections 5.1 To 5.3
        self.plotBasicFigures(sectionTables, valueTable)

        # Draw Sections 5.4 And 5.5
        self.plotSensorFigures(sectionTables, valueTable)

        # Draw Sections 5.6 And 5.7
        self.plotClassifierFigures(sectionTables, valueTable)

    def plotBasicFigures(self, sectionTables, valueTable):
        """
        This method draws the figures of the basic chain, the feature types and the window sizes

        Arguments
        =========
        sectionTables : Dictionary from section label to its averaged table
        valueTable : Printed paper values of the paper reference table

        Output
        ======
        None
        """
        # Section 5.1: Sensor Sets And Classifiers At The Paper's Step
        basicRows = sectionTables["5.1"][sectionTables["5.1"]["stepVariant"] == "paperStep"].copy()
        basicRows["setup"] = basicRows["sensorSet"] + " " + basicRows["classifier"]
        basicPaper = valueTable[valueTable["section"] == "5.1"].assign(setup=lambda table: table["configuration"] + " " + table["classifier"])
        self.plotPrecisionRecallBars(
            basicRows, "setup", ["scheme"],
            f"5.1 Basic ARC ( {self.windowSeconds:g} s windows, {self.stepSeconds:g} s step, mean and variance )", "section51BasicChain.png", basicPaper,
        )

        # Section 5.2: Feature Sets Per Sensor Set
        self.plotPrecisionRecallBars(sectionTables["5.2"], "featureSet", ["scheme", "sensorSet"], "5.2 Feature types ( 1-NN; Raw is per frame, RawWindow a variant )", "section52FeatureTypes.png")

        # Section 5.3: Window Size Sweep
        self.plotSweep(
            sectionTables["5.3"], "windowSeconds", "Window size Ws ( s )",
            f"5.3 Window size, all sensors ( solid: paper step rule, dotted: toolbox {self.toolboxSweepStepSeconds:g} s step )", "section53WindowSizes.png",
        )

    def plotSensorFigures(self, sectionTables, valueTable):
        """
        This method draws the figures of the sensor placements and the sensor modalities

        Arguments
        =========
        sectionTables : Dictionary from section label to its averaged table
        valueTable : Printed paper values of the paper reference table

        Output
        ======
        None
        """
        # Section 5.4: Placements
        placementPaper = valueTable[valueTable["section"] == "5.4"].rename(columns={"configuration": "placement"})
        self.plotPrecisionRecallBars(sectionTables["5.4"], "placement", ["scheme"], "5.4 Sensor placement ( accelerometer and gyroscope, 1-NN )", "section54Placements.png", placementPaper)

        # Section 5.5: Modalities, The Best Gyroscope Value Marked On Our Best Gyroscope Set
        modalityTable = sectionTables["5.5"]
        modalityPaper = valueTable[valueTable["section"] == "5.5"].rename(columns={"configuration": "modalitySet"}).copy()
        dependentGyroscope = modalityTable[(modalityTable["scheme"] == "personDependent") & (modalityTable["modality"] == "gyroscope")]
        modalityPaper.loc[modalityPaper["modalitySet"] == "bestGyroscopeOnly", "modalitySet"] = dependentGyroscope.loc[dependentGyroscope["precisionAll"].idxmax(), "modalitySet"]
        self.plotPrecisionRecallBars(modalityTable, "modalitySet", ["scheme"], "5.5 Sensor modality ( 1-NN )", "section55Modalities.png", modalityPaper)

    def plotClassifierFigures(self, sectionTables, valueTable):
        """
        This method draws the figures of the classifiers and the mRMR feature selection

        Arguments
        =========
        sectionTables : Dictionary from section label to its averaged table
        valueTable : Printed paper values of the paper reference table

        Output
        ======
        None
        """
        # Section 5.6: Classifiers At The Paper's Step
        classifierRows = sectionTables["5.6"][sectionTables["5.6"]["stepVariant"] == "paperStep"]
        classifierPaper = valueTable[valueTable["section"] == "5.6"].rename(columns={"configuration": "sensorSet"})
        self.plotPrecisionRecallBars(classifierRows, "classifier", ["scheme", "sensorSet"], "5.6 Classifiers ( AdaBoost replaces Joint Boosting )", "section56Classifiers.png", classifierPaper)

        # Section 5.7: mRMR Sweep Against Mean And Variance Only
        basicAll = sectionTables["5.1"]
        baselineRows = basicAll[(basicAll["stepVariant"] == "paperStep") & (basicAll["sensorSet"] == "allSensors") & (basicAll["classifier"] == "kNearestNeighbour")]
        baselineValues = {schemeRow.scheme: (schemeRow.precisionAll, schemeRow.recallAll) for schemeRow in baselineRows.itertuples(index=False)}
        selectionPaper = valueTable[valueTable["section"] == "5.7"].assign(xValue=lambda table: table["selectedFeatures"].astype(float))
        self.plotSweep(sectionTables["5.7"], "selectedFeatures", "Selected features S", "5.7 mRMR feature selection from the All pool, 1-NN", "section57FeatureSelection.png", baselineValues, selectionPaper)
