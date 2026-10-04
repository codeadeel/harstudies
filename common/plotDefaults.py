# This file is responsible for the shared matplotlib style, categorical colours and figure saving
# %%
# Importing Libraries
import matplotlib.pyplot as plt


# %%
# Plotting Helpers
def applyPlotDefaults():
    """
    This function applies the shared matplotlib style used by every figure

    Arguments
    =========
    None

    Output
    ======
    None
    """
    # Update The Matplotlib Parameters
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
    })


def categoricalColours(colourCount):
    """
    This function returns distinct categorical colours for up to sixty categories

    Arguments
    =========
    colourCount : Number of colours needed

    Output
    ======
    List of RGB colours in a fixed order
    """
    # Collect The Qualitative Palettes
    paletteNames = ["tab10"] if colourCount <= 10 else ["tab20", "tab20b", "tab20c"]
    paletteColours = [colour for paletteName in paletteNames for colour in plt.get_cmap(paletteName).colors]

    # Check The Palette Size
    if colourCount > len(paletteColours):
        raise ValueError(f"Only {len(paletteColours)} categorical colours are available, {colourCount} requested")
    return paletteColours[:colourCount]


def saveFigure(figure, figurePath):
    """
    This function saves a figure as png and closes it to release memory

    Arguments
    =========
    figure : Matplotlib figure to save
    figurePath : Destination png path

    Output
    ======
    Path of the saved figure
    """
    # Save And Close The Figure
    figure.savefig(figurePath, bbox_inches="tight")
    plt.close(figure)
    return figurePath
