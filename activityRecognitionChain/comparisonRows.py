# This file is responsible for measuring one printed paper value against our values and finding the closest convention
# %%
# Importing Libraries


# %%
# Comparison Rows
class comparisonRows:
    def buildComparisonRow(self, referenceRow, ourRow, classifierName, comparedAs, stepName):
        """
        This method measures our precision and recall against one printed paper value under every averaging convention

        Arguments
        =========
        referenceRow : Value row of the paper reference table
        ourRow : Our averaged row that matches the printed value
        classifierName : Classifier of our row
        comparedAs : main, secondRow or sensitivityRow
        stepName : paperStep or toolboxStep

        Output
        ======
        Dictionary with the paper values, our values, their differences and the gaps
        """
        comparisonRow = {
            "referenceId": referenceRow.referenceId, "comparedAs": comparedAs, "comparedClassifier": classifierName,
            "section": referenceRow.section, "page": referenceRow.page, "scheme": referenceRow.scheme,
            "paperConfiguration": referenceRow.configuration, "ourConfiguration": ourRow["configuration"], "stepVariant": stepName,
            "paperPrecision": referenceRow.paperPrecision, "paperRecall": referenceRow.paperRecall,
        }

        # Measure The Distance Under Every Averaging Convention
        for averagingName, (precisionColumn, recallColumn) in self.averagingColumns.items():
            ourPrecision = 100 * ourRow[precisionColumn]
            ourRecall = 100 * ourRow[recallColumn]
            comparisonRow[f"ourPrecision_{averagingName}"] = ourPrecision
            comparisonRow[f"ourRecall_{averagingName}"] = ourRecall
            comparisonRow[f"precisionDifference_{averagingName}"] = ourPrecision - referenceRow.paperPrecision
            comparisonRow[f"recallDifference_{averagingName}"] = ourRecall - referenceRow.paperRecall
            comparisonRow[f"absoluteGap_{averagingName}"] = abs(ourPrecision - referenceRow.paperPrecision) + abs(ourRecall - referenceRow.paperRecall)

        # Pick The Closest Class Set And Precision Convention
        return self.addClosestConvention(comparisonRow)

    def addClosestConvention(self, comparisonRow):
        """
        This method adds the closest class set, its margin and the closest precision convention to a comparison row

        Arguments
        =========
        comparisonRow : Dictionary with the gaps of every averaging convention

        Output
        ======
        The comparison row with the added entries
        """
        # Decide Between All Classes And Gesture Classes First, Then Between The Precision Conventions
        allClassesGaps = (comparisonRow["absoluteGap_all12ZeroPrecision"], comparisonRow["absoluteGap_all12DefinedPrecision"])
        gestureClassesGaps = (comparisonRow["absoluteGap_nonNull11ZeroPrecision"], comparisonRow["absoluteGap_nonNull11DefinedPrecision"])
        comparisonRow["bestGapAllClasses"] = min(allClassesGaps)
        comparisonRow["bestGapGestureClasses"] = min(gestureClassesGaps)
        comparisonRow["closerClassSet"] = (
            "all classes" if min(allClassesGaps) < min(gestureClassesGaps)
            else "gesture classes" if min(gestureClassesGaps) < min(allClassesGaps) else "tie"
        )
        comparisonRow["classSetMargin"] = abs(min(allClassesGaps) - min(gestureClassesGaps))
        closerGaps = gestureClassesGaps if comparisonRow["closerClassSet"] == "gesture classes" else allClassesGaps
        comparisonRow["precisionConvention"] = (
            "same, no class went unpredicted" if closerGaps[0] == closerGaps[1]
            else "zero precision closer" if closerGaps[0] < closerGaps[1] else "defined precision closer"
        )
        return comparisonRow
