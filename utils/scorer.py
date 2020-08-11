import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class Scorer:
    def __init__(self):
        self.binary_scorers = {'f1_micro': (lambda x, y: f1_score(x, y, average='micro')),
                               'f1_macro': (lambda x, y: f1_score(x, y, average='macro')),
                               'precision_micro': (lambda x, y: precision_score(x, y, average='micro')),
                               'precision_macro': (lambda x, y: precision_score(x, y, average='macro')),
                               'recall_micro': (lambda x, y: recall_score(x, y, average='micro')),
                               'recall_macro': (lambda x, y: recall_score(x, y, average='macro')),
                               'accuracy': accuracy_score}
        self.non_binary_scorers = {'roc_auc_micro': (lambda x, y: roc_auc_score(x, y, average='micro')),
                                   'roc_auc_macro': (lambda x, y: roc_auc_score(x, y, average='macro'))}

    def __call__(self, pred, pred_proba, labels, round=4):
        n_classes = pred_proba.shape[1]
        ohe_labels = np.zeros((labels.shape[0], n_classes))
        for i, label in enumerate(labels):
            ohe_labels[i, label] = 1
        history = defaultdict(list)
        for scorer_name, scorer in self.binary_scorers.items():
            history[scorer_name] = np.round(scorer(labels, pred), round)
        for scorer_name, scorer in self.non_binary_scorers.items():
            if 'roc_auc' in scorer_name:
                if scorer_name is 'roc_auc_macro':
                    try:
                        history[scorer_name] = np.round(scorer(ohe_labels, pred_proba), round)
                    except ValueError:
                        history[scorer_name] = 0
                else:
                    history[scorer_name] = np.round(scorer(ohe_labels, pred_proba), round)
            else:
                history[scorer_name] = np.round(scorer(labels, pred_proba), round)
        return history