import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from metrics import precision_recall_curve_metric, roc_curve_metric

df = pd.read_csv("../sample_store/processed_corpus_full.csv", index_col=0)
with open(r"../sample_store/label_full.pkl", "rb") as input_file:
   label = pickle.load(input_file)

X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.30, random_state=42)
clf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=0)
clf.fit(X_train, y_train)
predicted_result = clf.predict_proba(X_test)
predicted_result = predicted_result[:, 1]

precision, recall, avg_pr = precision_recall_curve_metric(y_test, predicted_result, plot=True)
fpr, tpr, roc_auc_rf = roc_curve_metric(y_test, predicted_result, plot=True)
print(avg_pr, roc_auc_rf)
# pickle.dump((y_test, predicted_result), open(os.path.join("", '../plot_result/sLDA_PRC.pkl'), 'wb'))
# pickle.dump((y_test, predicted_result), open(os.path.join("", '../plot_result/sLDA_ROC.pkl'), 'wb'))
