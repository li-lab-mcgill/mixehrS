import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc


def precision_recall_curve_metric(y, p, plot=False):
    precision, recall, threshold = precision_recall_curve(y, p)
    avg_pr = average_precision_score(y, p)
    if plot:
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(avg_pr))
        plt.show()
    return precision, recall, avg_pr


def roc_curve_metric(y, p, plot=False):
    fpr, tpr, threshold = roc_curve(y, p)
    roc_auc_rf = auc(fpr, tpr)

    if plot:
        plt.plot(fpr, tpr, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF', roc_auc_rf))
        plt.xlim([-0.01, 1.00])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.show()

    return roc_auc_rf
