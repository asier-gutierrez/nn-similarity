from homology import graphigs2vrs_clean
from gtda.plotting import plot_diagram
import igraph

if __name__ == '__main__':
    graph1 = igraph.load("./output/learning_EpochEvolutionCallback/CIFAR100CNN/DROPOUT_0.0/0/14.pickle")
    dgms = graphigs2vrs_clean([graph1])
    fig = plot_diagram(dgms[0], homology_dimensions=list(range(4)), plotly_params={'font-family': 'CMU Serif', 'font-size': 16})
    fig.write_image("diagram.png")

