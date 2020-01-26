NN and Doc2Vec exercise
=======================
*Jenia Kim*

This repo contains an example of how to use the simple neural network from Tariq Rashid's
[Make Your Own Neural Network](https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G/)
book for an NLP task (instead of the MNIST database example used in the book).

## Data
The movies dataset (plots and genres) used in this example is downloadable from
[here](https://github.com/RaRe-Technologies/movie-plots-by-genre/tree/master/data).

## Step-by-step
- requirements are listed in [**environment.yml**](environment.yml)
- run [**preprocess.py**](preprocess.py) to pre-process the raw data
- run [**doc2vec.py**](doc2vec.py) to train and save a doc2vec model
- run [**experiment.py**](experiment.py) to run a classification experiment

## Usage example
For all scripts, a variety of parameters can be used; check the scripts for details.
Example:

```
python preprocess.py --inpath ./data/tagged_plots_movielens.csv --outpath ./data/all_data.pkl
```

## Doc2Vec
We use Le and Mikolov's (2014) Doc2Vec to create representations for the movie plots.

> Le, Q., & Mikolov, T. (2014, January). Distributed representations of sentences and documents.
> In International conference on machine learning (pp. 1188-1196). url: [https://arxiv.org/pdf/1405.4053.pdf](https://arxiv.org/pdf/1405.4053.pdf)

For a tutorial about the gensim implementation of Doc2Vec (which was used here), see [here](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-download-auto-examples-tutorials-run-doc2vec-lee-py).
