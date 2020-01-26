"""
Preprocess a csv file of movie plots and genres.(*)
Save the resulting dataframe in a pkl file.

(*) The csv file can be downloaded from here:
    https://github.com/RaRe-Technologies/movie-plots-by-genre/tree/master/data

To set paths to in and out files, use the parameters '--inpath' and '--outpath'.
"""

from pathlib import Path
import argparse
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords


def tokenize_text(text):
    """
    Preprocess text:
    - lowercase
    - tokenize with nltk
    - remove stopwords
    - remove punctuation

    Parameters
    ----------
    text: str

    Returns
    -------
    tokens: list of str
    """
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            word = word.lower()
            if word in stopwords.words('english') + ["'s", "n't"]:
                continue
            if word in list(string.punctuation) + ['...', '“', '”', '..', '…', '``', "''", '--']:
                continue
            tokens.append(word)
    return tokens


def preprocess_movieplots(in_path, out_path):
    """
    Preprocess a csv file of movie plots and genres.
    Pickle the data as a pandas DataFrame with the following columns:
        - index (unnamed column)
        - movieId
        - plot
        - tag
        - tokens

    Parameters
    ----------
    in_path: str
        path to csv file
    out_path: str
        path to output pkl file

    Returns
    -------
    None
    """
    in_path = Path(in_path)
    out_path = Path (out_path)

    df = pd.read_csv(in_path, index_col='Unnamed: 0')
    # remove the name of the index column
    df.index.name = None
    # drop rows that have NA values (in at least one column)
    df = df.dropna()
    print(f"Data loaded: {df.shape[0]} rows.")

    print("Tokenizing...", flush=True, end=' ')
    df['tokens'] = df['plot'].apply(lambda x: tokenize_text(x))
    print("done!")

    df.to_pickle(out_path)
    print(f"Data is pickled: {out_path}")

    return None


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--inpath', default='./data/tagged_plots_movielens.csv')
    argparser.add_argument('--outpath', default='./data/all_data.pkl')
    args = argparser.parse_args()

    preprocess_movieplots(
        in_path=args.inpath,
        out_path=args.outpath,
    )
