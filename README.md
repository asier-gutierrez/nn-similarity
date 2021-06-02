# Characterizing and Measuring the Similarity of Neural Networks with Persistent Homology
This repository contains the code for our paper: "Characterizing and Measuring the Similarity of Neural Networks with Persistent Homology". This code transforms a trained Neural Network (NN) into a directed graph, performs the Persistent Homology computations and calculates distances across the different trained NNs. This process is done 5 times to validate the result and reduce the variation.

## Computational Requirements
This code requires large Random Access Memory (RAM). We used a machine of 1.5TB of RAM and 128 physical processor cores.

We suggest using either a Cloud Computing Machine (Amazon Web Services, Microsoft Azure, Google Cloud, IBM Cloud or similar) or reducing the network capacity.

## Execution
Install the requirements. We suggest using conda.
```
conda create --name <env> --file requirements.txt
```

Configuration of experiments can be changed in basic_properties/conf/*

To execute experiments regarding architecture comparison:
```
python basic_properties/(cifar_mlp.py|mnist_fashion_mlp.py|reuters_mlp.py|mnist_mlp.py|language_identification_mlp.py)
```

To execute experiments additional control experiment on input order over different datasets
`python basic_properties/input_order.py`

For visualization of results try `draw_analyze.py` and `draw_input_order.py` in the `basic_properties` folder.

## Paper
See our pre-print on arXiv: https://arxiv.org/abs/2101.07752

Cite our paper:
```
@misc{pérezfernández2021characterizing,
      title={Characterizing and Measuring the Similarity of Neural Networks with Persistent Homology}, 
      author={David Pérez-Fernández and Asier Gutiérrez-Fandiño and Jordi Armengol-Estapé and Marta Villegas},
      year={2021},
      eprint={2101.07752},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
