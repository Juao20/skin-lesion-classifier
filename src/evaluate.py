"""Métriques et matrice de confusion."""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def report(y_true, y_pred, class_names):
    return classification_report(y_true, y_pred, target_names=class_names)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path=None):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, xticks_rotation=45
    )
    if out_path:
        disp.figure_.savefig(out_path, bbox_inches="tight")
    return disp


# TODO: CLI d'évaluation sur le test set
