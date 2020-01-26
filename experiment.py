"""
Run a classification experiment with a simple feed-forward neural network with one hidden layer.
"""

from nn import neuralNetwork
from utils import tag2vec, vec2tag, plot_confusion_matrix, classif_report

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split


def run_experiment(
    data_path,
    model_path,
    experiment_title,
    results_dir,
    report_format='latex',
    test_size=0.1,
    reinfer=False,
    input_nodes=50,
    hidden_nodes=30,
    lr=0.005,
    epochs=40,
):
    """
    Run a classification experiment.
    Requires a pre-trained doc2vec model trained on the same data.

    Parameters
    ----------
    data_path: str
        path to a pkl file with the data
    model_path: str
        path to a file with a pre-trained doc2vec model
    experiment_title: str
        unique name for the experiment (to be used for storing results)
    results_dir: str
        path to the directory where results are to be stored
    report_format: {'latex', 'json'}
        format in which the classification report is to be stored
    test_size: float
        percentage of data to be saved for testing (default: 0.1)
    reinfer: bool
        if True, vectors will be re-inferred; if False, the vectors that are already in the model will be used (default: False)
    input_nodes: int
        number of neurons in the input layer; should match the input vector size (default: 50)
    hidden_nodes: int
        number of neurons in the hidden layer (default: 30)
    lr: float
        learning rate of the neural network (default: 0.005)
    epochs: int
        number of iterations over the training data (default: 40)

    Returns:
    -------
    None
    """
    ############## PATHS ##############
    data_path = Path(data_path)
    results_dir = Path(results_dir)

    # model_path = Path(model_path)
    # commented out since was giving trouble in Windows

    ############## PREPARE DATA ##############
    df = pd.read_pickle(data_path)
    print(f"Data loaded: {df.shape[0]} rows.")

    # convert the genre tags to vectors, where
    # each vector has 6 dimensions (6 genres), where
    # all values are 0.01 except for the gold class which is 0.99
    print("Converting tags to arrays...", flush=True, end=' ')
    tags_index = {
        'action': 0,
        'animation': 1,
        'comedy': 2,
        'fantasy': 3,
        'romance': 4,
        'sci-fi': 5,
    }
    df['out_vec'] = df['tag'].apply(lambda x: tag2vec(x, convert_dict=tags_index))
    print("done!")

    # split
    print("Splitting to train / test...", flush=True, end=' ')
    train_data, test_data = train_test_split(
        df,
        test_size=test_size,
        random_state=19,
    )
    print("done!")

    ############## PREPARE VECTORS ##############
    # load doc2vec model
    print(f"Loading doc2vec model: {model_path} ...", flush=True, end=' ')
    model = Doc2Vec.load(model_path)
    print("done!")

    # tag docs (lists of tokens) with their index
    print("Tagging documents...", flush=True, end=' ')
    train_tagged = train_data.apply(lambda x: TaggedDocument(words=x['tokens'], tags=[x.name]), axis=1)
    test_tagged = test_data.apply(lambda x: TaggedDocument(words=x['tokens'], tags=[x.name]), axis=1)
    print("done!")

    # prep input vectors
    print(f"Preparing input vectors (reinfer={reinfer})...", flush=True, end=' ')
    if reinfer:
        X_train = [model.infer_vector(tagged_doc.words) for tagged_doc in train_tagged]
        X_test = [model.infer_vector(tagged_doc.words) for tagged_doc in test_tagged]
    else:
        X_train = [model.docvecs[tagged_doc.tags[0]] for tagged_doc in train_tagged]
        X_test = [model.docvecs[tagged_doc.tags[0]] for tagged_doc in test_tagged]
    print("done!")

    # prep output vectors (gold labels):
    print("Preparing output vectors...", flush=True, end=' ')
    y_train = train_data['out_vec'].to_list()
    y_test = test_data['out_vec'].to_list()
    print("done!")

    ############## TRAIN NEURAL NETWORK ##############
    # create an instance of neural network
    n = neuralNetwork(
        inputnodes=input_nodes,
        hiddennodes=hidden_nodes,
        outputnodes=6,
        learningrate=lr)

    # train the neural network
    print(f"Training the neural network ({epochs} epochs)...")
    for e in range(epochs):
        print(f"epoch {e}...")
        for idx, record in enumerate(X_train):
            n.train(record, y_train[idx])
    print("done!")

    ############## TEST NEURAL NETWORK ##############
    print("Testing the neural network...", flush=True, end=' ')
    y_pred = list()
    for record in X_test:
        pred_label = n.query(record)
        y_pred.append(pred_label)
    print("done!")

    ############## EVALUATE ##############
    # save classification report to file
    print("Generating classification report...", flush=True, end=' ')
    classif_report(
        y_test,
        y_pred,
        convert_dict=tags_index,
        out_dir=results_dir,
        title=f"report_{experiment_title}",
        format=report_format,
    )

    # save confusion matrix plot to file
    print("Generating confusion matrix plot...", flush=True, end=' ')
    plot_confusion_matrix(
        y_test,
        y_pred,
        convert_dict=tags_index,
        out_dir=results_dir,
        title=f"cm_plot_{experiment_title}",
    )

    return None


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', default='./data/all_data.pkl')
    argparser.add_argument('--model_path', default='./doc2vec_models/model_50_dm0.d2v')
    argparser.add_argument('--experiment_title', default='model_50_dm0_default_nn')
    argparser.add_argument('--results_dir', default='./results/')
    argparser.add_argument('--report_format', default='latex')
    argparser.add_argument('--test_size', default=0.1)
    argparser.add_argument('--reinfer', default='False')
    argparser.add_argument('--input_nodes', default=50)
    argparser.add_argument('--hidden_nodes', default=30)
    argparser.add_argument('--lr', default=0.005)
    argparser.add_argument('--epochs', default=40)
    args = argparser.parse_args()

    BOOL_STATES = {
        'true': True,
        'false': False,
        '1': True,
        '0': False,
    }

    run_experiment(
        data_path=args.data_path,
        model_path=args.model_path,
        experiment_title= args.experiment_title,
        results_dir=args.results_dir,
        report_format=args.report_format,
        test_size=float(args.test_size),
        reinfer=BOOL_STATES[args.reinfer.lower()],
        input_nodes=int(args.input_nodes),
        hidden_nodes=int(args.hidden_nodes),
        lr=float(args.lr),
        epochs=int(args.epochs),
    )
