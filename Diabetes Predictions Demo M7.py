'''
Title: BUAD 5802 (AA) Module 7 Group Project Coding Demo for Group 4
Author: Jason Street
Collaborators: Anand Persaud, Andrew Hetzner
Created: 08-Oct-2024
Description:
    To demonstrate the ability for AI to be able to accurately diagnose the
    likelihood for a patient having diabetes even if patient records are
    missing important information through the use of imputation methods. The
    model compares results when all data is present, then randomly drops data
    and utilizes imputation methods from scikit-learn:
    Simple (mean, median, and most frequent), KNN, and Iterative (experimental)
    Comparing the results of the various methods to the full data results.

Dataset: Diabetes Prediction Dataset; Mohammed Mustafa (Owner)
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset

'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer

# global random seed for reproducibility
np.random.seed(1693)

# ADJUSTABLE PARAMETERS
# percent of data values to simulate missing
percent_missing = 0.25
# n-nearest neighbor for KNN imputer model
num_neighbor = 5
# % test size for splitting into training vs test
test_set_size = 0.4
# maximum number of iterations for model to run
iterations = 20

# load dataset and drop 'smoking_history' column
data = pd.read_csv('diabetes_prediction_dataset.csv')
data = data.drop(columns=['smoking_history'])

# reassign 'gender' column to numerical values: Female = 1, Male = 2, Other = 3
data['gender'] = data['gender'].map({'Female': 1, 'Male': 2, 'Other': 3})

# keep copy of data for a control evaluation (with all data present)
full_data = data.copy()

# separate copy of data to simulate missing patient information to allow for imputation
sim_missing_data = data.copy()

# establishing which columns can be simulated with missing data (all X columns)
columns_to_modify = ['gender', 'age', 'hypertension', 'heart_disease',
                     'bmi', 'HbA1c_level', 'blood_glucose_level']

# randomly select records in which values will be dropped across different
# columns while keeping track to compare original values to the imputed values
missing_info = []
num_missing = int(percent_missing * len(sim_missing_data))
missing_indices = np.random.choice(sim_missing_data.index,
                                   size=num_missing, replace=False)
for idx in missing_indices:
    col = np.random.choice(columns_to_modify)
    missing_info.append({'index': idx, 'column': col,
                         'original_value': sim_missing_data.loc[idx, col]})
    sim_missing_data.loc[idx, col] = np.nan

# IMPUTE - using KNNImputer
knn_imputer = KNNImputer(n_neighbors=num_neighbor)
data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(sim_missing_data),
                                columns=sim_missing_data.columns)

# IMPUTE - using IterativeImputer
iterative_imputer = IterativeImputer(random_state=1693)
data_iter_imputed = pd.DataFrame(iterative_imputer.fit_transform(sim_missing_data),
                                 columns=sim_missing_data.columns)

# IMPUTE - using SimpleImputer with mean
simple_imputer_mean = SimpleImputer(strategy='mean')
data_simple_imputed_mean = pd.DataFrame(simple_imputer_mean.fit_transform(sim_missing_data),
                                        columns=sim_missing_data.columns)

# IMPUTE - using SimpleImputer with median
simple_imputer_median = SimpleImputer(strategy='median')
data_simple_imputed_median = pd.DataFrame(simple_imputer_median.fit_transform(sim_missing_data),
                                          columns=sim_missing_data.columns)

# IMPUTE - using SimpleImputer with most_frequent
simple_imputer_most_frequent = SimpleImputer(strategy='most_frequent')
data_simple_imputed_most_frequent = pd.DataFrame(simple_imputer_most_frequent.fit_transform(sim_missing_data),
                                                 columns=sim_missing_data.columns)

# print entries from missing_info with original and imputed values
print("\nComparison of the original and imputed values \nfor the first 20 of the randomly dropped data points:")
for entry in missing_info[:20]:
    idx = entry['index']
    col = entry['column']
    original_value = entry['original_value']
    knn_imputed_value = data_knn_imputed.loc[idx, col]
    iterative_imputed_value = data_iter_imputed.loc[idx, col]
    simple_mean_value = data_simple_imputed_mean.loc[idx, col]
    simple_median_value = data_simple_imputed_median.loc[idx, col]
    simple_most_frequent_value = data_simple_imputed_most_frequent.loc[idx, col]

    print(f"\nIndex: {idx}, Column: {col}")
    print(f"Original Value: {original_value}")
    print(f"KNN Imputed Value: {knn_imputed_value}")
    print(f"Iterative Imputed Value: {iterative_imputed_value}")
    print(f"Simple Imputer (Mean) Imputed Value: {simple_mean_value}")
    print(f"Simple Imputer (Median) Imputed Value: {simple_median_value}")
    print(f"Simple Imputer (Most Frequent) Imputed Value: {simple_most_frequent_value}")

# establish X variables and y-target for full data
X_full = full_data.drop(columns=['diabetes']).values
y_full = full_data['diabetes'].values

# split training and test sets
X_train_full, X_test_full, \
    y_train_full, y_test_full = train_test_split(X_full, y_full,
                                                 test_size=test_set_size,
                                                 random_state=1693)

# scale X variables for the full data
scaler_full = StandardScaler()
X_train_full = scaler_full.fit_transform(X_train_full)
X_test_full = scaler_full.transform(X_test_full)

# train/evaluate using logistic regression (binary y-target) on the full data
classifier_full = LogisticRegression(random_state=1693, max_iter=iterations)
classifier_full.fit(X_train_full, y_train_full)
train_accuracy_full = accuracy_score(y_train_full, classifier_full.predict(X_train_full))
test_accuracy_full = accuracy_score(y_test_full, classifier_full.predict(X_test_full))

# establish X variables and y-target from the simulated data ("missing" values)
X_knn = data_knn_imputed.drop(columns = ['diabetes']).values
y_knn = data_knn_imputed['diabetes'].values

X_iter = data_iter_imputed.drop(columns = ['diabetes']).values
y_iter = data_iter_imputed['diabetes'].values

X_simple_mean = data_simple_imputed_mean.drop(columns = ['diabetes']).values
y_simple_mean = data_simple_imputed_mean['diabetes'].values

X_simple_median = data_simple_imputed_median.drop(columns = ['diabetes']).values
y_simple_median = data_simple_imputed_median['diabetes'].values

X_simple_most_frequent = data_simple_imputed_most_frequent.drop(columns = ['diabetes']).values
y_simple_most_frequent = data_simple_imputed_most_frequent['diabetes'].values

# FUNCTION - split into train and test then scale X variables
def split_and_scale(X, y):
    X_train, X_test, \
        y_train, y_test = train_test_split(X, y,
                                           test_size = test_set_size,
                                           random_state=1693)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# split and scale function on each imputation method
X_train_knn, X_test_knn, \
    y_train_knn, y_test_knn = split_and_scale(X_knn, y_knn)
X_train_iter, X_test_iter, \
    y_train_iter, y_test_iter = split_and_scale(X_iter, y_iter)
X_train_simple_mean, X_test_simple_mean, \
    y_train_simple_mean, y_test_simple_mean = split_and_scale(X_simple_mean,
                                                              y_simple_mean)
X_train_simple_median, X_test_simple_median, \
    y_train_simple_median, y_test_simple_median = split_and_scale(X_simple_median,
                                                                  y_simple_median)
X_train_simple_most_frequent, \
    X_test_simple_most_frequent, \
        y_train_simple_most_frequent, \
            y_test_simple_most_frequent = split_and_scale(X_simple_most_frequent,
                                                          y_simple_most_frequent)

# FUNCTION - train and evaluate using logistic regression, includes detailed metrics
def train_and_evaluate_with_metrics(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(random_state=1693, max_iter=iterations)
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    # Calculate accuracy for training and test sets
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Get classification report for the test set
    test_report = classification_report(y_test, y_test_pred, target_names=['No Diabetes (0)', 'Diabetes (1)'])

    return train_accuracy, test_accuracy, test_report

# METRICS - train/test accuracy, report for F1/recall/precision each imputer
train_accuracy_knn, test_accuracy_knn, \
    report_knn = train_and_evaluate_with_metrics(X_train_knn, X_test_knn, y_train_knn, y_test_knn)
train_accuracy_iter, test_accuracy_iter, \
    report_iter = train_and_evaluate_with_metrics(X_train_iter, X_test_iter,
                                                  y_train_iter, y_test_iter)
train_accuracy_simple_mean, test_accuracy_simple_mean, \
    report_simple_mean = train_and_evaluate_with_metrics(X_train_simple_mean, X_test_simple_mean,
                                                         y_train_simple_mean, y_test_simple_mean)
train_accuracy_simple_median, test_accuracy_simple_median, \
    report_simple_median = train_and_evaluate_with_metrics(X_train_simple_median, X_test_simple_median,
                                                           y_train_simple_median, y_test_simple_median)
train_accuracy_simple_most_frequent, test_accuracy_simple_most_frequent, \
    report_simple_most_frequent = train_and_evaluate_with_metrics(X_train_simple_most_frequent, X_test_simple_most_frequent,
                                                                  y_train_simple_most_frequent, y_test_simple_most_frequent)

# print train/test accuracy and reports for comparison
print("\nAll models ran with\n\tTraining Size: {:.2f}% \n\tTest Size: {:.2f}% \n\tMaximum Number of Iterations: {}\n".format((1 - test_set_size) * 100, test_set_size * 100, iterations))

print("Full Data (no missing values)")
print(f"\tTraining Accuracy: {train_accuracy_full * 100:.2f}%")
print(f"\tTest Accuracy: {test_accuracy_full * 100:.2f}%")
print("Classification Report for Full Data:\n", classification_report(y_test_full, classifier_full.predict(X_test_full), target_names=['No Diabetes (0)', 'Diabetes (1)']))

print(f"\nKNN Imputer (n-nearest neighbor = {num_neighbor})")
print(f"\tTraining Accuracy: {train_accuracy_knn * 100:.2f}%")
print(f"\tTest Accuracy: {test_accuracy_knn * 100:.2f}%")
print("Classification Report for KNN Imputer:\n", report_knn)

print("\nIterative Imputer")
print(f"\tTraining Accuracy: {train_accuracy_iter * 100:.2f}%")
print(f"\tTest Accuracy: {test_accuracy_iter * 100:.2f}%")
print("Classification Report for Iterative Imputer:\n", report_iter)

print("\nSimple Imputer (Mean)")
print(f"\tTraining Accuracy: {train_accuracy_simple_mean * 100:.2f}%")
print(f"\tTest Accuracy: {test_accuracy_simple_mean * 100:.2f}%")
print("Classification Report for Simple Imputer (Mean):\n", report_simple_mean)

print("\nSimple Imputer (Median)")
print(f"\tTraining Accuracy: {train_accuracy_simple_median * 100:.2f}%")
print(f"\tTest Accuracy: {test_accuracy_simple_median * 100:.2f}%")
print("Classification Report for Simple Imputer (Median):\n", report_simple_median)

print("\nSimple Imputer (Most Frequent)")
print(f"\tTraining Accuracy: {train_accuracy_simple_most_frequent * 100:.2f}%")
print(f"\tTest Accuracy: {test_accuracy_simple_most_frequent * 100:.2f}%")
print("Classification Report for Simple Imputer (Most Frequent):\n", report_simple_most_frequent)
