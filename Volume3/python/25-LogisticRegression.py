
def best_tau(predicted_labels, true_labels, n_tau=100, plot=True):
    """
    Parameters
    ----------
    predicted_labels : ndarray of shape (n,)
        The predicted labels for the data
    true_labels : ndarray of shape (n,)
        The actual labels for the data
    n_tau : int
        The number of values to try for tau
    plot : boolean
        Whether or not to plot the roc curve

    Returns
    -------
    best_tau : float
        The optimal value for tau for the data.
    """
    pass

def auc_scores(unchanged_logreg, changed_logreg):
    """
    Parameters
    ----------
    unchanged_logreg : float in (0,1)
        The value to use for C in the unchanged model
    changed_logreg : float in (0,1)
        The value to use for C in the changed model

    Returns
    -------
    unchanged_auc : float
        The auc for the unchanged model
    changed_auc : float
        The auc for the changed model
    """
    pass

def find_best_parameters(choices):
    """
    Parameters
    ----------
    choices : int
        The number of values to try for C and alpha

    Returns
    -------
    best : list of length 4
        The best values for C for the unchanged and changed logistic
         regression models, and the best values for alpha for the
         unchanged and changed Naive Bayes models, respectively.
    """
    pass
