# Road Graph Generator

This repository contains the source code for the paper:  
[Road Graph Generator: Mapping roads at construction sites from GPS data](https://arxiv.org/abs/2402.09919).

## Introduction

The Road Graph Generator is a tool designed to map roads at construction sites using GPS data. It provides a solution for accurately generating road graphs based on the collected GPS data.

## Installation

To use the Road Graph Generator, follow these steps:

1. Clone the repository: 
```bash
git clone https://github.com/katarzynamichalowska/road-graph-generator.git
```

We recommend creating a new environment. If you are using Conda, please run:

```bash
conda create -n road_generator
conda activate road_generator
conda install pip
```

3. Install the required dependencies: 
```bash
pip install -r requirements.txt
```

## Usage

To generate road graphs from GPS data, run the following command:

```bash
python generate_graph.py
```

## Contributors

We want to thank Helga Holmestad for her invaluable contribution to this project:  
[GitHub: Helga Holmestad](https://github.com/helgaholmestadsintef).


## Citing

If you use Road Graph Generator in an academic paper, please cite as:

```bash
@article{michalowska2024road,
  title={{Road Graph Generator}: {M}apping roads at construction sites from {GPS} data},
  author={Micha{\l}owska, Katarzyna and Holmestad, Helga Margrete Bodahl and Riemer-S{\o}rensen, Signe},
  journal={arXiv preprint arXiv:2402.09919},
  year={2024}
}
```

