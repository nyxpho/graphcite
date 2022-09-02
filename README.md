# GraphCite
The code and data for citation intent prediction used in the paper:

```GraphCite: Citation Intent Classification in Scientific Publications via Graph Embeddings```, 
accepted at the workshop Sci-k 2022, in conjunction with TheWebConf 2022:
https://sci-k.github.io/2022/



We provide models for doing citation intent prediction on ACL-ARC and SciCite datasets. To run the code please install the libraries specified in requirements.txt

The code can be run as follows:

usage: graphcite.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [-s SEED] [-lr LEARNING_RATE] [-a AUTHORS] [-v VENUES] dataset model

The following arguments are required: dataset, model

The default values are:
- batch size 16
- epochs 10
- seed 0
- learning rate 2r-5
- adding authors and venues to citation graphs is by default set to false


The parameter dataset can take two values: acl-arc or scicite



The parameter model can take the following values: 

CitationBERT - fine tuning BERT on citation prediction

CitationGAT - training GAT on citation prediction

CitationBERTGAT - our GraphCite model

CitationMLP - a simple MLP layer on top of the titles embeddings

To cite this work:

>@inproceedings{10.1145/3487553.3524657,
author = {Berrebbi, Dan and Huynh, Nicolas and Balalau, Oana},
title = {GraphCite: Citation Intent Classification in Scientific Publications via Graph Embeddings},
year = {2022},
isbn = {9781450391306},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3487553.3524657},
doi = {10.1145/3487553.3524657},
abstract = {Citations are crucial in scientific works as they help position a new publication. Each citation carries a particular intent, for example, to highlight the importance of a problem or to compare against results provided by another method. The authors’ intent when making a new citation has been studied to understand the evolution of a field over time or to make recommendations for further citations. In this work, we address the task of citation intent prediction from a new perspective. In addition to textual clues present in the citation phrase, we also consider the citation graph, leveraging high-level information of citation patterns. In this novel setting, we perform a thorough experimental evaluation of graph-based models for intent prediction. We show that our model, GraphCite, improves significantly upon models that take into consideration only the citation phrase. Our code is available online1.},
booktitle = {Companion Proceedings of the Web Conference 2022},
pages = {779–783},
numpages = {5},
keywords = {citation intent classification, graph neural network},
location = {Virtual Event, Lyon, France},
series = {WWW '22}
}
