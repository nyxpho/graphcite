# GraphCite
The code and data for citation intent prediction.
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
