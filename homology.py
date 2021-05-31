import os
import numpy as np
from scipy.sparse import coo_matrix
from ripser import Rips
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence, FlagserPersistence
from gtda.diagrams import Filtering

MAXDIM = 4


def coomat2vr(coo_m, distance_matrix):
    rips = VietorisRipsPersistence(metric='precomputed', homology_dimensions=list(range(MAXDIM)))
    diagrams = rips.fit_transform([coo_m])
    rips.plot(diagrams)

    diagrams = rips.fit_transform([distance_matrix])
    rips.plot(diagrams)


def graphig2vr(G, path=None):
    rips = VietorisRipsPersistence(metric='precomputed', homology_dimensions=list(range(MAXDIM)))
    diagrams = rips.fit_transform_plot([G.get_adjacency_sparse(attribute='weight').tocoo()])
    if path:
        rips.plot(diagrams, plotly_params={'filename': path})
    else:
        rips.plot(diagrams)
    return diagrams


def graphigs2vrs(graphs):
    flags = FlagserPersistence(directed=True, homology_dimensions=list(range(MAXDIM)))
    diagrams = flags.fit_transform([G.get_adjacency_sparse(attribute='weight') for G in graphs])
    return diagrams


def graphigs2vrs_clean(graphs, n_jobs=2):
    flags = FlagserPersistence(directed=True, homology_dimensions=list(range(MAXDIM)), max_edge_weight=1.0,
                               n_jobs=n_jobs)
    diagrams = flags.fit_transform([G.get_adjacency_sparse(attribute='weight') for G in graphs])

    print("Before filtering", diagrams.shape)
    diagrams = Filtering(epsilon=0.6).fit_transform(diagrams)
    print("After filtering", diagrams.shape)

    # Replace
    # diagrams[diagrams == np.Inf] = 1.0
    return diagrams


''' networkx
def graph2vr(G):
    rips = Rips(maxdim=4)
    diagrams = rips.fit_transform(nx.adjacency_matrix(G), distance_matrix=True)
    rips.plot(diagrams, legend=True)
    plt.show()
'''
