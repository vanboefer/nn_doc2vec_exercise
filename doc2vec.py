"""
Train and save a Doc2Vec model.

More info about the gensim Doc2Vec:
https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec
"""

from pathlib import Path
import argparse
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def doc2vec_model(
    in_path,
    out_path,
    dm=0,
    vector_size=50,
    min_count=2,
    epochs=40,
):
    """
    Train and save a Doc2Vec model.

    Parameters
    ----------
    in_path: str
        path to pkl file with a DataFrame containing a `tokens` column
    out_path: str
        path to save the doc2vec model
    dm: {0,1}
        defines the training algorithm:
            `distributed memory` (PV-DM) if dm=1,
            `distributed bag of words` (PV-DBOW) if dm=0
    vector_size: int
        number of dimensions of the feature vectors (default: 50)
    min_count: int
        model ignores tokens with total frequency lower than this (default: 2)
    epochs: int
        number of iterations over the corpus (default: 40)

    Returns
    -------
    None
    """
    in_path = Path(in_path)

    # out_path = Path(out_path)
    # commented out since was giving trouble in Windows

    df = pd.read_pickle(in_path)
    print(f"Data loaded: {df.shape[0]} rows.")

    # tag docs (lists of tokens) with their index
    print("Tagging documents...", flush=True, end=' ')
    tagged = df.apply(
        lambda x: TaggedDocument(words=x['tokens'], tags=[x.name]), axis=1)
    print("done!")

    # instantiate a Doc2Vec model
    model = gensim.models.doc2vec.Doc2Vec(
        dm=dm,
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs
    )

    # build a vocabulary of all the unique words in the data and their counts
    print("Building vocabulary...", flush=True, end=' ')
    model.build_vocab(tagged)
    print(f"Vocabulary of {len(model.wv.vocab)} words is created!")

    # train the Doc2Vec model
    print(f"Training Doc2Vec model ({epochs} epochs)...", flush=True, end=' ')
    model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)
    print("done!")

    # save the model
    model.save(out_path)
    print(f"The model is saved: {out_path}")

    return None


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--inpath', default='./data/all_data.pkl')
    argparser.add_argument('--outpath', default='./doc2vec_models/model_50_dm0.d2v')
    argparser.add_argument('--dm', default=0)
    argparser.add_argument('--vecsize', default=50)
    argparser.add_argument('--mincount', default=2)
    argparser.add_argument('--epochs', default=40)
    args = argparser.parse_args()

    doc2vec_model(
        in_path=args.inpath,
        out_path=args.outpath,
        dm=int(args.dm),
        vector_size=int(args.vecsize),
        min_count=int(args.mincount),
        epochs=int(args.epochs),
    )
