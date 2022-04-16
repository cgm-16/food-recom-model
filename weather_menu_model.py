import matplotlib
import pandas as pd
import numpy as np
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib import font_manager

fm = font_manager.FontManager()
font_manager.FontManager().addfont("./NanumGothic.ttf")
fontp = font_manager.FontProperties(fname="./NanumGothic.ttf")

dataset = pd.DataFrame(pd.read_csv("./results/model_dataset.csv"))
X_data = pd.concat([dataset.loc[:,"avg_temp":"highest_temp"], dataset.loc[:,"is_rain":"is_holiday"]], axis=1)
X_data = pd.concat([dataset.loc[:,"precipitation":"highest_temp"], dataset.loc[:,"is_holiday"]], axis=1)
y_data = dataset.loc[:,"meal_type"]
class_names = list()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
for n in y_test:
    if n not in class_names:
        class_names.append(n)
X_data.head(10)

clf_boosting_base = GradientBoostingClassifier(random_state=10).fit(X_train, y_train)
clf_boosting_base.score(X_test, y_test)

clf_boosting = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=5, random_state=10, max_features="sqrt", subsample=0.8).fit(X_train, y_train)
clf_boosting.score(X_test, y_test)
y_pred = clf_boosting.predict(X_test)
y_test
cm = metrics.confusion_matrix(y_test, y_pred)
metrics.accuracy_score(y_test, y_pred)
metrics.recall_score(y_test, y_pred, average="weighted")
metrics.f1_score(y_test, y_pred, average="micro")

cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
plt.figure(figsize=(5,4))
sns.heatmap(cm_df,annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.xticks(fontproperties=fontp)
plt.yticks(fontproperties=fontp)
plt.show()

cm_df_norm = cm_df.applymap(lambda x : x/cm_df.max().max())
plt.figure(figsize=(5,4))
sns.heatmap(cm_df_norm,annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.xticks(fontproperties=fontp)
plt.yticks(fontproperties=fontp)
plt.show()

sklearn.tree.plot_tree(clf_boosting.estimators_[999, 0])
plt.show()

clf_rf_base = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0).fit(X_train, y_train)
clf_rf_base.score(X_test, y_test)