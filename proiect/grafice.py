import matplotlib.pyplot as plt
import pandas as pd
from seaborn import kdeplot, scatterplot
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from scikitplot.metrics import plot_silhouette
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from matplotlib.colors import rgb2hex


def generare_rampa(denumire, nr_clusteri):
    cmap = cm.get_cmap(denumire, nr_clusteri)
    culori = []
    for i in range(cmap.N):
        rgba = cmap(i)
        culori.append(rgb2hex(rgba))
    return culori


def plot_ierarhie(h, threshold, titlu, k, etichete,culori=None):
    fig = plt.figure(titlu, figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    if culori is not None:
        set_link_color_palette(culori)
    dendrogram(h, ax=ax, color_threshold=threshold,
               labels=etichete)


def show():
    plt.show()
