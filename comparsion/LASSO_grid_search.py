import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from metrics import precision_recall_curve_metric, roc_curve_metric
from sklearn.model_selection import GridSearchCV
import numpy as np

alphas = np.linspace(0.0, 0.2, num=6)
print(alphas)

df = pd.read_csv("../sample_store/processed_corpus_full.csv", index_col=0)
with open(r"../sample_store/label_full.pkl", "rb") as input_file:
   label = pickle.load(input_file)
print(df)
print(label)

# model = Lasso()
# grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
# grid.fit(df, label)
# print(grid.best_estimator_.alpha)
# print(grid.best_score_)
# best_alpha = grid.best_estimator_.alpha

alpha = 0.005
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20)
clf = Lasso(alpha=alpha)
clf.fit(X_train, y_train)
predicted_result = clf.predict(X_test)
precision, recall, avg_pr = precision_recall_curve_metric(y_test, predicted_result, plot=True)
fpr, tpr, roc_auc_rf = roc_curve_metric(y_test, predicted_result, plot=True)
print(avg_pr, roc_auc_rf)


# save parameter and curve
# pickle.dump((y_test, predicted_result), open(os.path.join("", '../plot_result/LASSO_PRC.pkl'), 'wb'))
# pickle.dump((y_test, predicted_result), open(os.path.join("", '../plot_result/LASSO_ROC.pkl'), 'wb'))


