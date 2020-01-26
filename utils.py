"""
Helper functions for running the experiment.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def tag2vec(tag, convert_dict):
    """
    Convert a genre tag (e.g. 'comedy') to a vector of 6 dimensions, where all values are 0.01 except for the gold class which is 0.99.

    Parameters:
    ----------
    tag: str
    convert_dict: dict
        dictionary with genre tags as keys and indices as values

    Returns:
    -------
    out_vec: np.ndarray
    """
    converted_tag = convert_dict[tag]
    out_vec = np.zeros(6) + 0.01
    out_vec[converted_tag] = 0.99
    return out_vec


def vec2tag(out_vec, convert_dict):
    """
    Back-convert a vector of 6 dimensions to the corresponding genre tag.
    The tag is determined based on the vector index with the max value.

    Parameters:
    ----------
    out_vec: np.ndarray
    convert_dict: dict
        dictionary with genre tags as keys and indices as values

    Returns:
    -------
    tag: str
    """
    reverse_dict = {v:k for k, v in convert_dict.items()}
    tag = reverse_dict[np.argmax(out_vec)]
    return tag


def plot_confusion_matrix(
    y_test,
    y_pred,
    convert_dict,
    out_dir,
    title='Confusion matrix',
    cmap=plt.cm.Blues
):
    """
    Save a confusion matrix plot in `out_dir`.

    Parameters:
    ----------
    y_test: list
        list of 6-dimensional vectors
    y_pred: list
        list of 6-dimensional vectors
    convert_dict: dict
        dictionary with genre tags as keys and indices as values
    out_dir: str
        path to the directory where plot is to be stored
    title: str
        title of the plot
    cmap: matplotlib Colormap
        colormap for the plot (default: plt.cm.Blues)

    Returns:
    -------
    None
    """
    # convert to labels
    y_test_convert = list()
    y_pred_convert = list()
    for item in y_test:
        tag = vec2tag(item, convert_dict)
        y_test_convert.append(tag)
    for item in y_pred:
        tag = vec2tag(item, convert_dict)
        y_pred_convert.append(tag)

    # confusion matrix
    cm = confusion_matrix(y_test_convert, y_pred_convert)

    # plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    target_names = ['action', 'animation', 'comedy', 'fantasy', 'romance', 'sci-fi']
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.ylim((5.5, -0.5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    saving_path = Path(f"{out_dir}/{title}.png")
    plt.savefig(saving_path, dpi=150)
    print(f"Plot saved: {out_dir}/{title}.png")
    plt.close()

    return None


def classif_report(
    y_test,
    y_pred,
    convert_dict,
    out_dir,
    title='Classification report',
    format='latex'
):
    """
    Save a classification report in `out_dir`.

    Parameters:
    ----------
    y_test: list
        list of 6-dimensional vectors
    y_pred: list
        list of 6-dimensional vectors
    convert_dict: dict
        dictionary with genre tags as keys and indices as values
    out_dir: str
        path to the directory where report is to be stored
    title: str
        title of the report
    format: {'latex', 'json'}
        store report in a json file or as a latex string in a txt file

    Returns:
    -------
    None
    """
    # convert to labels
    y_test_convert = list()
    y_pred_convert = list()
    for item in y_test:
        tag = vec2tag(item, convert_dict)
        y_test_convert.append(tag)
    for item in y_pred:
        tag = vec2tag(item, convert_dict)
        y_pred_convert.append(tag)

    # classification report
    report = classification_report(
        y_test_convert,
        y_pred_convert,
        output_dict=True,
    )

    # save
    if format == 'json':
        filepath = Path(f"{out_dir}/{title}.json")
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Report saved: {filepath}")
    elif format == 'latex':
        filepath = Path(f"{out_dir}/{title}.txt")
        # fix the report dict
        report['accuracy'] = {
            'precision': None,
            'recall': None,
            'f1-score': report['accuracy'],
            'support': report['macro avg']['support']
        }
        df = pd.DataFrame(report).transpose()
        df = df.round(3).astype({'support': int})
        latex = df.to_latex(na_rep='')
        with open(filepath, 'w') as f:
            f.write(latex)
        print(f"Report saved: {filepath}")
    else:
        print("Report not saved! `format` parameter should be either 'json' or 'latex'!")

    return None
