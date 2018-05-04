from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

import numpy as np
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore')

file = open("wvc2010_features.dat", "r")
features = []
labels = []
for line in file:
    temp = []
    line = line[:-1]
    line_arr = line.split(',')
    for i in range(1,len(line_arr)-1):
        temp.append(float(line_arr[i]))
    features.append(temp)
    labels.append(int(line[-1]))

train_features, test_features, train_labels, test_labels = train_test_split(features,labels,test_size=0.3)

#random forest

#finding optimal "n_estimators" hyper parameter
count = 10
max_score = 0
fcount = 10
while(count<=100):
    rf_model = RandomForestClassifier(n_estimators=count, random_state=32)
    rf_model.fit(train_features,train_labels)
    score=accuracy_score(rf_model.predict(test_features),test_labels)
    if(max_score < score):
        max_score = score
        fcount = count
    count = count + 20

#defining RF classifier with the n_estimators found
random_forest = RandomForestClassifier(n_estimators=fcount, random_state=32)
random_forest.fit(train_features, train_labels)
forest_predicted_labels = random_forest.predict(test_features)
forest_score=accuracy_score(forest_predicted_labels,test_labels)

#getting confusion matrix to calculate accuracy, f_Score, precision, recall
tn, fp, fn, tp = confusion_matrix(forest_predicted_labels,test_labels).ravel()
n = tn+fp+fn+tp
forest_accuracy = (tp+tn)/n
forest_f_score = 2*tp/(2*tp+fp+fn)
forest_precision = tp/(tp+fp)
forest_recall = tp/(tp+fn)
print()
print("random forest classification")
print("accuracy: ", forest_accuracy)
print("recall: ", forest_recall)
print("precision: ", forest_precision)
print("f_score: ", forest_f_score)

#fetching predicted labels probabilities for plotting precision recall curve
forest_predicted_labels_prob = random_forest.predict_proba(test_features)
forest_predicted_labels_prob = np.asarray(forest_predicted_labels_prob)
forest_precision, forest_recall, _ = precision_recall_curve(test_labels,forest_predicted_labels_prob[:, 1])
#getting AUC for PR
auc_rf = metrics.auc(forest_precision, forest_recall,reorder = True)
print("AUC of PR: ",auc_rf)

#decision tree

#finding optimal hyper parameter values

depth = 2
max_score = 0
fdepth = 10
while(depth<=10):
    decision_tree = tree.DecisionTreeClassifier(max_depth=depth)
    decision_tree.fit(train_features, train_labels)
    score=accuracy_score(decision_tree.predict(test_features),test_labels)
    if(max_score < score):
        max_score = score
        fdepth = depth
    depth = depth + 2

#defining decision tree classifier with the hyper parameters found

decision_tree = tree.DecisionTreeClassifier( min_samples_split=8, min_samples_leaf=15)
decision_tree.fit(train_features, train_labels)
decision_score=accuracy_score(decision_tree.predict(test_features),test_labels)

#getting confusion matrix to calculate accuracy, f_Score, precision, recall
tn, fp, fn, tp = confusion_matrix(decision_tree.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
decision_accuracy = (tp+tn)/n
decision_f_score = 2*tp/(2*tp+fp+fn)
decision_precision = tp/(tp+fp)
decision_recall = tp/(tp+fn)
print()
print("decision tree classification")
print("accuracy: ", decision_accuracy)
print("recall: ", decision_recall)
print("precision: ", decision_precision)
print("f_score: ", decision_f_score)

#fetching predicted labels probabilities for plotting precision recall curve
dt_predicted_labels_prob = decision_tree.predict_proba(test_features)
dt_predicted_labels_prob = np.asarray(dt_predicted_labels_prob)
dt_precision, dt_recall, _ = precision_recall_curve(test_labels,dt_predicted_labels_prob[:, 1])
#getting AUC for PR
auc_dt = metrics.auc(dt_precision, dt_recall,reorder = True)
print("AUC of PR: ",auc_dt)

# ada boost

#finding optimal hyper parameter values
estimators = 10
max_score = 0
festimators = 0

while(estimators<=100):
    ada_boost=AdaBoostClassifier(n_estimators=estimators)
    ada_boost.fit(train_features, train_labels)
    score=accuracy_score(ada_boost.predict(test_features),test_labels)
    if(max_score < score):
        max_score = score
        festimators = estimators
    estimators = estimators + 10
#defining adaboost classifier with the hyper parameters found

ada_boost=AdaBoostClassifier(n_estimators=estimators)
ada_boost.fit(train_features, train_labels)
adaboost_score=accuracy_score(ada_boost.predict(test_features),test_labels)

#getting confusion matrix to calculate accuracy, f_Score, precision, recall
tn, fp, fn, tp = confusion_matrix(ada_boost.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
adaboost_accuracy = (tp+tn)/n
adaboost_f_score = 2*tp/(2*tp+fp+fn)
adaboost_precision = tp/(tp+fp)
adaboost_recall = tp/(tp+fn)
print()
print("ada boost classification")
print("accuracy: ", adaboost_accuracy)
print("recall: ", adaboost_recall)
print("precision: ", adaboost_precision)
print("f_score: ", adaboost_f_score)

#fetching predicted labels probabilities for plotting precision recall curve
ada_predicted_labels_prob = ada_boost.predict_proba(test_features)
ada_predicted_labels_prob = np.asarray(ada_predicted_labels_prob)
ada_precision, ada_recall, _ = precision_recall_curve(test_labels,ada_predicted_labels_prob[:, 1])
#getting AUC for PR
auc_ada = metrics.auc(ada_precision, ada_recall,reorder = True)
print("AUC of PR: ",auc_ada)

# Logistic
#finding optimal hyper parameter values
Cp = 0.1
max_score = 0
fC = 0.1

while(fC<=1):
    logistic = LogisticRegression(C=Cp)
    logistic.fit(train_features, train_labels)
    score=accuracy_score(logistic.predict(test_features),test_labels)
    if(max_score < score):
        max_score = score
        fC = Cp
    Cp = Cp + 0.1
#defining logistic regression classifier with the hyper parameters found
logistic = LogisticRegression(C=fC)
logistic.fit(train_features, train_labels)
logistic_score=accuracy_score(logistic.predict(test_features),test_labels)

#getting confusion matrix to calculate accuracy, f_Score, precision, recall
tn, fp, fn, tp = confusion_matrix(logistic.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
logistic_accuracy = (tp+tn)/n
logistic_f_score = 2*tp/(2*tp+fp+fn)
logistic_precision = tp/(tp+fp)
logistic_recall = tp/(tp+fn)
print()
print("Logistic classification")
print("accuracy: ", logistic_accuracy)
print("recall: ", logistic_recall)
print("precision: ", logistic_precision)
print("f_score: ", logistic_f_score)

#fetching predicted labels probabilities for plotting precision recall curve
logistic_predicted_labels_prob = logistic.predict_proba(test_features)
logistic_predicted_labels_prob = np.asarray(logistic_predicted_labels_prob)
logistic_precision, logistic_recall, _ = precision_recall_curve(test_labels,logistic_predicted_labels_prob[:, 1])
#getting AUC for PR
auc_log = metrics.auc(logistic_precision, logistic_recall,reorder = True)
print("AUC of PR: ",auc_log)
# Naive bayes
nb = GaussianNB()
nb.fit(train_features, train_labels)
bayes_score=accuracy_score(nb.predict(test_features),test_labels)

#getting confusion matrix to calculate accuracy, f_Score, precision, recall
tn, fp, fn, tp = confusion_matrix(nb.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
nb_accuracy = (tp+tn)/n
nb_f_score = 2*tp/(2*tp+fp+fn)
nb_precision = tp/(tp+fp)
nb_recall = tp/(tp+fn)
print()
print("Naive Bayes")
print("accuracy: ", nb_accuracy)
print("recall: ", nb_recall)
print("precision: ", nb_precision)
print("f_score: ", nb_f_score)

#fetching predicted labels probabilities for plotting precision recall curve
nb_predicted_labels_prob = nb.predict_proba(test_features)
nb_predicted_labels_prob = np.asarray(nb_predicted_labels_prob)
nb_precision, nb_recall, _ = precision_recall_curve(test_labels,nb_predicted_labels_prob[:, 1])

#getting AUC for PR
auc_nb = metrics.auc(nb_precision, nb_recall,reorder = True)
print("AUC of PR: ",auc_nb)

plt.step(forest_recall, forest_precision, alpha=0.8)
plt.step(dt_recall,dt_precision,alpha=0.8)
plt.step(ada_recall, ada_precision, alpha = 0.8)
plt.step(logistic_recall,logistic_precision, alpha=0.8)
plt.step(nb_recall,nb_precision,alpha=0.8)
plt.legend(['Random Forest','Decision Tree', 'ADA Boost', 'Logistic Regression','Naive Bayes'], loc='upper right')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs recall of classifiers')
plt.savefig('pr_curve_classifiers.png')
plt.show()
plt.close()

#We found that random forest is giving best performance so we are trying to see the contributions of feature categories
metadata_features = [features[i][0:4] for i in range(0,len(features))]
language_features = [features[i][16:28] for i in range(0,len(features))]
text_features = [features[i][5:15] for i in range(0,len(features))]

metadata_features=np.asarray(metadata_features)
language_features=np.asarray(language_features)
text_features = np.asarray(text_features)
meta_lang_features = np.concatenate((metadata_features, language_features), axis=1)
meta_text_features = np.concatenate((metadata_features, text_features), axis = 1)
lang_text_features = np.concatenate((language_features, text_features), axis = 1)

#fitting metadata features
train_features, test_features, train_labels, test_labels = train_test_split(metadata_features,labels,test_size=0.3,random_state=32)
rf_classifier = RandomForestClassifier(n_estimators=fcount, random_state=32)
rf_classifier.fit(train_features, train_labels)
tn, fp, fn, tp = confusion_matrix(rf_classifier.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
meta_accuracy = (tp+tn)/n
meta_f_score = 2*tp / (2*tp+fp+fn)
meta_precision = tp/(tp+fp)
meta_recall = tp/(tp+fn)

meta_predicted_labels_prob = rf_classifier.predict_proba(test_features)
meta_predicted_labels_prob = np.asarray(meta_predicted_labels_prob)
meta_precision, meta_recall, _ = precision_recall_curve(test_labels,meta_predicted_labels_prob[:, 1])



#fitting language features
train_features, test_features, train_labels, test_labels = train_test_split(language_features,labels,test_size=0.3,random_state=32)
rf_classifier = RandomForestClassifier(n_estimators=fcount, random_state=32)
rf_classifier.fit(train_features, train_labels)
tn, fp, fn, tp = confusion_matrix(rf_classifier.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
lang_accuracy = (tp+tn)/n
lang_f_score = 2*tp / (2*tp+fp+fn)
lang_precision = tp/(tp+fp)
lang_recall = tp/(tp+fn)

lang_predicted_labels_prob = rf_classifier.predict_proba(test_features)
lang_predicted_labels_prob = np.asarray(lang_predicted_labels_prob)
lang_precision, lang_recall, _ = precision_recall_curve(test_labels,lang_predicted_labels_prob[:, 1])


#fitting text features
train_features, test_features, train_labels, test_labels = train_test_split(text_features,labels,test_size=0.3,random_state=32)
rf_classifier = RandomForestClassifier(n_estimators=fcount, random_state=32)
rf_classifier.fit(train_features, train_labels)
tn, fp, fn, tp = confusion_matrix(rf_classifier.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
text_accuracy = (tp+tn)/n
text_f_score = 2*tp / (2*tp+fp+fn)
text_precision = tp/(tp+fp)
text_recall = tp/(tp+fn)

text_predicted_labels_prob = rf_classifier.predict_proba(test_features)
text_predicted_labels_prob = np.asarray(text_predicted_labels_prob)
text_precision, text_recall, _ = precision_recall_curve(test_labels,text_predicted_labels_prob[:, 1])


#fitting metadata and language features
train_features, test_features, train_labels, test_labels = train_test_split(meta_lang_features,labels,test_size=0.3,random_state=32)
rf_classifier = RandomForestClassifier(n_estimators=fcount, random_state=32)
rf_classifier.fit(train_features, train_labels)
tn, fp, fn, tp = confusion_matrix(rf_classifier.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
meta_lang_accuracy = (tp+tn)/n
meta_lang_f_score = 2*tp / (2*tp+fp+fn)
meta_lang_precision = tp/(tp+fp)
meta_lang_recall = tp/(tp+fn)

meta_lang_predicted_labels_prob = rf_classifier.predict_proba(test_features)
meta_lang_predicted_labels_prob = np.asarray(meta_lang_predicted_labels_prob)
meta_lang_precision, meta_lang_recall, _ = precision_recall_curve(test_labels,meta_lang_predicted_labels_prob[:, 1])

#fitting metadata and text features
train_features, test_features, train_labels, test_labels = train_test_split(meta_text_features,labels,test_size=0.3,random_state=32)
rf_classifier = RandomForestClassifier(n_estimators=fcount, random_state=32)
rf_classifier.fit(train_features, train_labels)
tn, fp, fn, tp = confusion_matrix(rf_classifier.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
meta_text_accuracy = (tp+tn)/n
meta_text_f_score = 2*tp / (2*tp+fp+fn)
meta_text_precision = tp/(tp+fp)
meta_text_recall = tp/(tp+fn)

meta_text_predicted_labels_prob = rf_classifier.predict_proba(test_features)
meta_text_predicted_labels_prob = np.asarray(meta_text_predicted_labels_prob)
meta_text_precision, meta_text_recall, _ = precision_recall_curve(test_labels,meta_text_predicted_labels_prob[:, 1])


#fitting language and text features
train_features, test_features, train_labels, test_labels = train_test_split(lang_text_features,labels,test_size=0.3,random_state=32)
rf_classifier = RandomForestClassifier(n_estimators=fcount, random_state=32)
rf_classifier.fit(train_features, train_labels)
tn, fp, fn, tp = confusion_matrix(rf_classifier.predict(test_features),test_labels).ravel()
n = tn+fp+fn+tp
lang_text_accuracy = (tp+tn)/n
lang_text_f_score = 2*tp / (2*tp+fp+fn)
lang_text_precision = tp/(tp+fp)
lang_text_recall = tp/(tp+fn)

lang_text_predicted_labels_prob = rf_classifier.predict_proba(test_features)
lang_text_predicted_labels_prob = np.asarray(lang_text_predicted_labels_prob)
lang_text_precision, lang_text_recall, _ = precision_recall_curve(test_labels,lang_text_predicted_labels_prob[:, 1])


#print("meta features accuracy : ", meta_accuracy)
#print("accuracy : ", lang_accuracy)
#print("accuracy : ", text_accuracy)
#print("f_score : ", meta_f_score)
#print("f_score : ", lang_f_score)
#print("f_score : ", text_f_score)
#print("precision: ", meta_precision)
#print("precision: ", lang_precision)
#print("precision: ", text_precision)
#print("recall: ", meta_recall)
#print("recall: ", lang_recall)
#print("recall: ", text_recall)

plt.step( meta_recall, meta_precision, alpha = 0.8)
plt.step( lang_recall, lang_precision, alpha = 0.8)
plt.step( text_recall, text_precision, alpha = 0.8)
plt.step(meta_lang_recall, meta_lang_precision, alpha = 0.8)
plt.step(meta_text_recall, meta_text_precision, alpha = 0.8)
plt.step(lang_text_recall, lang_text_precision, alpha = 0.8)
plt.legend(['Metadata','Language','Text','Metadata and Language','Metadata and Text', 'Language and text'], loc='upper right')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall for features')
plt.savefig('pr_curve_features.png')
plt.show()
plt.close()
