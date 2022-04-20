import pandas as pd
import sklearn.tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

## 한글이 처리 안되는 문제 위한 한글 폰트
fm = font_manager.FontManager()
font_manager.FontManager().addfont("./NanumGothic.ttf")
fontp = font_manager.FontProperties(fname="./NanumGothic.ttf")

## 데이터 입력
dataset = pd.DataFrame(pd.read_csv("./results/model_dataset.csv"))
X_data = pd.concat([dataset.loc[:,"avg_temp":"highest_temp"], dataset.loc[:,"is_rain":"is_holiday"]], axis=1)
y_data = dataset.loc[:,"meal_type"]

## 데이터 요약
X_data.info()
y_data.info()
X_data.describe()
X_data.describe(include=["bool"], exclude=["int"])
y_data.describe()
y_data_counts = pd.DataFrame(data=y_data.value_counts())
y_data_counts.describe()

# 요약 시각화
plt.figure(figsize=(16, 10))
plt.title("y_data plot")
sns.barplot(x=y_data_counts.index, y="meal_type", data=y_data_counts)
plt.xticks(fontproperties=fontp)
plt.show()

## 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)

class_names = list()
for n in y_test:
    if n not in class_names:
        class_names.append(n)

X_data.head(10)
y_data.head(10)
X_train.head(10)
X_test.head(10)
y_train.head(10)
y_test.head(10)


## 모델 생성
clf_boosting_base = HistGradientBoostingClassifier(random_state=10).fit(X_train, y_train)
clf_boosting_base.score(X_test, y_test)

clf_boosting = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.02, random_state=10).fit(X_train, y_train)
clf_boosting.score(X_test, y_test)
y_pred = clf_boosting.predict(X_test)
y_prob = clf_boosting.predict_proba(X_test)
y_test


## 시각화
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=class_names)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report")
print(classification_report(y_test, y_pred))

# Confusion Matrix 시각화
cm_df = pd.DataFrame(cm, columns=class_names, index=class_names)
plt.figure(figsize=(16, 10))
sns.heatmap(cm_df,annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.xticks(fontproperties=fontp)
plt.yticks(fontproperties=fontp)
plt.show()

cm_df_norm = cm_df.applymap(lambda x : x/cm_df.max().max())
plt.figure(figsize=(16, 10))
sns.heatmap(cm_df_norm,annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.xticks(fontproperties=fontp)
plt.yticks(fontproperties=fontp)
plt.show()


# ROC 커브
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(clf_boosting, X_test, y_test, n_classes=len(class_names), figsize=(16, 10))