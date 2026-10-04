# This file is responsible for resolving the data and results directories shared by every part
# %%
# Importing Libraries
from pathlib import Path


# %%
# Path Helpers
def resolveDirectory(rootDirectory, *subDirectories, createDirectory=False):
    """
    This function joins a root directory with its sub directories and optionally creates the result

    Arguments
    =========
    rootDirectory : Root directory as a string or path
    subDirectories : Directory names appended below the root directory
    createDirectory : Whether a missing directory is created ( default : False )

    Output
    ======
    Resolved directory path
    """
    # Join The Directory Parts
    directoryPath = Path(rootDirectory).joinpath(*subDirectories)

    # Create The Directory When Requested
    if createDirectory:
        directoryPath.mkdir(parents=True, exist_ok=True)
    return directoryPath


def createPartDirectories(resultsRoot, partName):
    """
    This function creates the results directory of one analysis part and its figures directory

    Arguments
    =========
    resultsRoot : Root directory that holds every part's results
    partName : Folder name of the analysis part

    Output
    ======
    Tuple of the part results directory and its figures directory
    """
    # Create The Part Directories
    resultsDirectory = resolveDirectory(resultsRoot, partName, createDirectory=True)
    figuresDirectory = resolveDirectory(resultsDirectory, "figures", createDirectory=True)
    return resultsDirectory, figuresDirectory
