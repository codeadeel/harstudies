# This file is responsible for checking the paper's statements on the best classifier and the best feature count
# %%
# Importing Libraries


# %%
# Best Choice Checks
class bestChoiceChecks:
    def checkClassifierClaim(self, classifierTable, statementTable):
        """
        This method checks that the stated classifier gives the best precision and recall with the stated sensors at the paper step

        Arguments
        =========
        classifierTable : Section 5.6 averages
        statementTable : Paper statements indexed by reference id

        Output
        ======
        List of claim rows, one per scheme
        """
        claimRows = []
        # Check That The Stated Classifier Gives The Best Precision And Recall With The Stated Sensors At The Paper Step
        classifierStatement = statementTable.loc["5.6-statement"]
        statedClassifier = classifierStatement["classifier"]
        paperStepRows = classifierTable[(classifierTable["stepVariant"] == "paperStep") & (classifierTable["sensorSet"] == classifierStatement["configuration"])]
        for schemeName, schemeRows in paperStepRows.groupby("scheme", sort=False):
            schemeRows = schemeRows.set_index("classifier")
            otherRows = schemeRows.drop(index=statedClassifier)
            precisionChange = 100 * (schemeRows.loc[statedClassifier, "precisionAll"] - otherRows["precisionAll"].max())
            recallChange = 100 * (schemeRows.loc[statedClassifier, "recallAll"] - otherRows["recallAll"].max())
            claimRows.append({
                "referenceId": "5.6-statement", "section": "5.6", "page": classifierStatement["page"], "scheme": schemeName, "variant": classifierStatement["configuration"],
                "paperStatement": classifierStatement["paperSentence"],
                "ourFinding": f"best precision {schemeRows['precisionAll'].idxmax()}, best recall {schemeRows['recallAll'].idxmax()}; {statedClassifier} minus the best "
                              f"other classifier: precision {precisionChange:+.1f} points, recall {recallChange:+.1f} points",
                "precisionChange": precisionChange, "recallChange": recallChange,
                "firstPartHolds": bool(precisionChange > 0), "secondPartHolds": bool(recallChange > 0),
                "holds": bool(precisionChange > 0 and recallChange > 0),
            })
        return claimRows

    def checkSelectionClaim(self, selectionTable, statementTable):
        """
        This method checks that the stated feature count gives the best precision and recall in both schemes

        Arguments
        =========
        selectionTable : Section 5.7 averages
        statementTable : Paper statements indexed by reference id

        Output
        ======
        List of claim rows, one per scheme
        """
        claimRows = []
        # Check That The Stated Feature Count Gives The Best Precision And Recall In Both Schemes
        selectionStatement = statementTable.loc["5.7-statement"]
        statedCount = int(selectionStatement["selectedFeatures"])
        for schemeName, schemeRows in selectionTable.groupby("scheme", sort=False):
            schemeRows = schemeRows.set_index("selectedFeatures")
            otherRows = schemeRows.drop(index=statedCount)
            precisionChange = 100 * (schemeRows.loc[statedCount, "precisionAll"] - otherRows["precisionAll"].max())
            recallChange = 100 * (schemeRows.loc[statedCount, "recallAll"] - otherRows["recallAll"].max())
            claimRows.append({
                "referenceId": "5.7-statement", "section": "5.7", "page": selectionStatement["page"], "scheme": schemeName, "variant": selectionStatement["configuration"],
                "paperStatement": selectionStatement["paperSentence"],
                "ourFinding": f"precision peaks at S = {schemeRows['precisionAll'].idxmax()} and recall at S = {schemeRows['recallAll'].idxmax()}; S = {statedCount} minus "
                              f"the best other S: precision {precisionChange:+.1f} points, recall {recallChange:+.1f} points",
                "precisionChange": precisionChange, "recallChange": recallChange,
                "firstPartHolds": bool(precisionChange > 0), "secondPartHolds": bool(recallChange > 0),
                "holds": bool(precisionChange > 0 and recallChange > 0),
            })
        return claimRows
