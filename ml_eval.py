from dataset import *
from feature_extraction import *
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

training_instances = [(assignment.name, student_id)
                      for assignment in training for student_id in assignment.ids]
testing_instances = [(assignment.name, student_id)
                     for assignment in testing for student_id in assignment.ids]

training_plagiarized = {(assignment.name, student_id)
                        for assignment in training for student_id in assignment.plagiarisms}
testing_plagiarized = {(assignment.name, student_id)
                       for assignment in testing for student_id in assignment.plagiarisms}


training_features = {(assignment_id, student_id): get_student_features(assignment_id,
                                                                       student_id) for (assignment_id, student_id) in training_instances}
testing_features = {(assignment_id, student_id): get_student_features(assignment_id,
                                                                      student_id) for (assignment_id, student_id) in testing_instances}

training_features = {key: val for key,
                     val in training_features.items() if val != None}
testing_features = {key: val for key,
                    val in testing_features.items() if val != None}

train_X = []
train_y = []
for key in training_features:
    train_X.append(assignmentToFeatureVector(training_features[key]))
    train_y.append(key in training_plagiarized)
train_X = np.array(train_X, dtype="float32")
train_X[(train_X != train_X)] = 0

test_X = []
test_y = []
for key in testing_features:
    test_X.append(assignmentToFeatureVector(testing_features[key]))
    test_y.append(key in testing_plagiarized)
test_X = np.array(test_X, dtype="float32")
test_X[(test_X != test_X)] = 0

print("---- RandomForestClassifier ----")
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('classifier', RandomForestClassifier(class_weight="balanced"))])
pipeline.fit(train_X, train_y)
prediction = pipeline.predict(test_X)
print(metrics.classification_report(test_y, prediction))
print(metrics.confusion_matrix(test_y, prediction))

print("---- LogisticRegression ----")
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('classifier', LogisticRegression(class_weight="balanced"))])
pipeline.fit(train_X, train_y)
prediction = pipeline.predict(test_X)
print(metrics.classification_report(test_y, prediction))
print(metrics.confusion_matrix(test_y, prediction))

print("---- SVC ----")
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('classifier', SVC(class_weight="balanced"))])
pipeline.fit(train_X, train_y)
prediction = pipeline.predict(test_X)
print(metrics.classification_report(test_y, prediction))
print(metrics.confusion_matrix(test_y, prediction))

print("---- LinearSVC ----")
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('classifier', LinearSVC(class_weight="balanced"))])
pipeline.fit(train_X, train_y)
prediction = pipeline.predict(test_X)
print(metrics.classification_report(test_y, prediction))
print(metrics.confusion_matrix(test_y, prediction))

print("---- DecisionTreeClassifier ----")
pipeline = Pipeline([('scaler', StandardScaler()),
                     ('classifier', DecisionTreeClassifier(class_weight="balanced"))])
pipeline.fit(train_X, train_y)
prediction = pipeline.predict(test_X)
print(metrics.classification_report(test_y, prediction))
print(metrics.confusion_matrix(test_y, prediction))
