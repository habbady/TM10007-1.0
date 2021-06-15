'''
This is the main of the ML method.
'''
# Run this to use from colab environment
!pip install -q --upgrade git+https://github.com/karinvangarderen/tm10007_project.git

# General packages
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from sklearn.model_selection import RepeatedStratifiedKFold as k_fold_strat
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
# Classifiers and kernels
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
 
# Data loading function
from brats.load_data import load_data

# Some functions we will use here

# Preprocessing - Handling missing data 
def drop_features(check, drop1, drop2, thresh):
  '''
  Drop features that contain too many missing values
    
  Input:
      check: result of the check wether imputation is needed
      drop1: resulting values after dropping features
      drop2: resulting values after dropping features
      thresh: threshold for amount of allowed missing values
  '''
  dropped = []
  for idx, value in enumerate(check.isna().sum()):
      if value > (len(check.index) * thresh)//1:
          dropped.append(idx)

  drop1.drop(drop1.columns[dropped], axis = 1, inplace = True)
  drop2.drop(drop2.columns[dropped], axis = 1, inplace = True)

  return drop1, drop2

def drop_samples(dataframe, labels, thresh):
  '''
  Drop samples that contain too many missing values
    
  Input:
      dataframe: dataframe consisting of all the sample values
      labels: LGG/GBM labels
      thres: threshold for amount of allowed missing values
  '''
  dropped = []
  for idx, value in enumerate(dataframe.isna().sum(axis = 1)):
      if value > (len(dataframe.columns) * thresh)//1:
          dropped.append(idx)

  dataframe.drop(dataframe.index[dropped], axis = 0, inplace = True)
  labels.drop(labels.index[dropped], axis = 0, inplace = True)

  return dataframe, labels

def replace_missing_data(df, labels):
  '''
  Replaces missing data from x_train with KNN
  
  Input:
      df: datafame containing all the values
      labels: LGG/GBM labels
  '''
  nan = np.nan
  df_no_obj = df.select_dtypes(exclude = 'object')
  mask = np.isinf(df_no_obj)
  df_no_obj[mask] = np.nan

  LGG_data = df_no_obj.loc[labels == 'LGG']
  GBM_data = df_no_obj.loc[labels == 'GBM']

  # Get features untill TGM
  LGG_data = LGG_data.iloc[:,0:696]
  GBM_data = GBM_data.iloc[:,0:696]

  # Drop features
  df_no_obj, GBM_data = drop_features(LGG_data, df_no_obj, GBM_data, 0.3)
  df_no_obj, LGG_data = drop_features(GBM_data, df_no_obj, LGG_data, 0.3)

  # Drop samples
  df_no_obj, labels = drop_samples(df_no_obj, labels, 0.3)

  # Check if imputation is needed
  df_no_TGM = df_no_obj.iloc[:,0:474]
  sum_series = df_no_TGM.isna().sum()
  tot_sum = sum_series.sum()

  # Missing data in TGM features are imputed with zeros
  mask = np.isnan(df_no_obj)
  df_no_obj[mask] = 0
  clean_df = df_no_obj

  # Transforming a string label to a binary label
  labels = pd.Series([1 if label == 'GBM' else 0 for label in labels])
  return clean_df, labels

#Preprocessing - scaling the data 
def pre_processing(x_train, x_test):
  '''
  Scales the training data with a Min-Max scaler
  Input:
      x_train: training data
      x_test: test data
  '''
  scaler = preprocessing.MinMaxScaler()
  x_train_scaled = scaler.fit_transform(x_train)
  x_test_scaled = scaler.transform(x_test)
  return x_train_scaled, x_test_scaled

# Feature selection - PCA 
def get_pcs_train_test(x_train, x_test, exp_var_thresh = 0.95):
  '''
  Perform PCA to reduce the number of features
  Input:
      x_train: training data
      x_test: test data
      exp_var_thres: the threshold for explained variance (set to 0.95)
  '''
  pca = PCA()
  pca.fit(x_train)
  explained_variance = pca.explained_variance_ratio_

  n_components = 0
  tot_var = 0
  for variance in explained_variance:
      tot_var += variance
      n_components += 1
      if tot_var > exp_var_thresh:
          break

  pca_95 = PCA(n_components = n_components)
  pcs_train = pca_95.fit_transform(x_train)
  pcs_test = pca_95.transform(x_test)
  return pcs_train, pcs_test, n_components

# Confusion matrix 
def confusion_matrix_visualized(conf_mat, labels_conf, title):
  '''
  Gives a figure of (normalized) confusion matrix of 2 class classification performance , accompanied by a colormap.
  Input:
      conf_mat: confusion matrix of (size 2 by 2) --> [[AA  AB][BA BB]] for labels A and B with the first and second lettre representing the true and predicted class, respectively.
      labels_conf: the labels of the respective classes, list with size 2
      title: string to supply in title 
  '''
  fig = plt.figure()
  ax = fig.add_subplot(111)

  conf_mat_norm = conf_mat / conf_mat.sum(axis=1)[:,None]
  for i in [0,1]:
    for j in [0,1]:
      if i == j:
        ax.text(i,j,f'{np.round(conf_mat_norm[j][i],2)}', color = 'w', va = 'center', ha = 'center')
      else:
        ax.text(i,j,f'{np.round(conf_mat_norm[j][i],2)}', color = 'k', va = 'center', ha = 'center')

  cax = ax.matshow(conf_mat_norm, cmap=plt.cm.Blues)
  cax.set_clim(0, 1)

  fig.colorbar(cax)
  ax.set_xticklabels([''] + labels_conf)
  ax.set_yticklabels([''] + labels_conf)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title(title)
  plt.show()

# AUC curve 
def AUC_visualized(fpr,tpr,auc_score,title):
  '''
  Gives a figure of the AUC curve.
  Input:
      fpr: array of false positive rates.
      tpr: array of true positive rates.
      auc_score: AUC score to supply the figure with.   
      title: sting to supply in title.
  '''
  
  fig = plt.figure()
  ax = fig.add_subplot(111)

  ax.plot(fpr, tpr, marker='o', color = 'b', label = f'AUC = {auc_score}')
  ax.plot(fpr, fpr, 'k--')
  ax.fill_between(fpr, tpr, fpr, color = 'b', alpha = 0.25)
  ax.legend(loc = 'lower right')
  plt.title(title)
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.show()

# function for calculating the average n_components and slack 
def param_means(accuracies, C_vals, n_vals):
  '''
  Calculate the mean for N_components and C_parameter for every unique value in the accuracy that is presented.
  Input:
      accuracies: array with accuracies over the different iterations.
      C_vals: array with slack parameters chosen for each iteration.
      n_vals: array with number of principal components chosen for each iteration.

  '''
  
  acc_unique = np.unique(accuracies)
  C_means = np.empty(acc_unique.shape)
  C_std = np.empty(acc_unique.shape)
  n_comp_means = np.empty(acc_unique.shape)
  n_comp_std = np.empty(acc_unique.shape)

  for num, val in enumerate(acc_unique):
      C_means[num] = np.mean([C_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
      C_std[num] = np.std([C_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
      n_comp_means[num] = np.mean([n_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
      n_comp_std[num] = np.std([n_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])

  return acc_unique, C_means, C_std, n_comp_means, n_comp_std
  
def param_means2(accuracies, E_vals, n_vals):
    '''
    Calculate the mean for N_components and E_parameter for every unique value in the accuracy that is presented.
    Input:
    accuracies: array with accuracies over the different iterations.
    E_vals: array with estimator parameters chosen for each iteration (number of trees).
    n_vals: array with number of principal components chosen for each iteration.
    '''

    acc_unique = np.unique(accuracies) 
    E_means = np.empty(acc_unique.shape) 
    E_std = np.empty(acc_unique.shape)
    n_comp_means = np.empty(acc_unique.shape)
    n_comp_std = np.empty(acc_unique.shape)
    
    for num, val in enumerate(acc_unique):
      E_means[num] = np.mean([E_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
      E_std[num] = np.std([E_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
      n_comp_means[num] = np.mean([n_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
      n_comp_std[num] = np.std([n_vals[i] for i in range(len(accuracies)) if accuracies[i] == val])
    return acc_unique, E_means, E_std, n_comp_means, n_comp_std

def evaluate(y_test, y_test_pred, y_test_proba, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_AUC, conf_mat_total):
    '''
    A function that calculates metric scores of a model based on the true label, prediction and probability of a test set. 
    input:
    y_test = true label
    y_test_pred = the predicted label of the test set
    y_test_proba = the probability that the sample has label 1
    mean_accuracy = list that saves the mean accuracy per repeat
    mean_precision = list that saves the mean precision per repeat
    mean_recall = list that saves the mean recall per repeat
    AUC = list that saves the mean AUC per repeat
    conf_mat_total = confusion matrix that sums the confusion matrices of all previous repeats
    '''

    print(f'The test accuracy is {accuracy_score(y_test, y_test_pred) * 100} %')
    print(f'The test AUC score is {roc_auc_score(y_test, y_test_proba)}')
    
    # Add the accuracy for each round to the mean accuracy list
    accuracy_round = np.round(accuracy_score(y_test, y_test_pred),3)
    mean_accuracy.append(accuracy_round)
    mean_precision.append(precision_score(y_test, y_test_pred))
    mean_recall.append(recall_score(y_test, y_test_pred))
    mean_f1.append(f1_score(y_test, y_test_pred))
    mean_AUC.append(roc_auc_score(y_test, y_test_proba))

    # Confusion matrix total
    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
    conf_mat_total = conf_mat_total + conf_mat

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)

    AUC = np.round(roc_auc_score(y_test, y_test_proba),2)
    return mean_accuracy, mean_precision, mean_recall, mean_f1, mean_AUC, conf_mat, conf_mat_total, fpr, tpr, AUC

def appending_splits(y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat,  y_test, y_test_pred, y_test_proba, best_score):
    '''
    A function that adds the values of the current split to the values of the previous splits within the same repeat.
    y_test = true label per split
    y_test_pred = the predicted label of the test set per split
    y_test_proba = the probability that the sample has label 1 per split
    best_score = the accuracy of the classifier on the validation set per split
    y_test_repeat = true label per repeat
    y_test_pred_repeat = the predicted label of the test set per repeat
    y_test_proba_repeat = the probability that the sample has label 1  per repeat
    val_accuracy_repeat = the accuracy of the classifier on the validation set per repeat
    '''

    y_test_repeat.extend(y_test)
    y_test_pred_repeat.extend(y_test_pred)
    y_test_proba_repeat.extend(y_test_proba)
    val_accuracy_repeat.append(np.round(best_score,2))
    return y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat

def update_roc(count_repeat,accuracy_best, accuracy_worst,mean_accuracy, conf_mat_best, conf_mat_worst, fpr_best, tpr_best, AUC_best, fpr_worst, tpr_worst, AUC_worst, conf_mat, fpr, tpr, AUC):

      '''
      A function that updates the best and worst ROC metric scores based on accuracy.
      '''
      
      if count_repeat == 1:
        accuracy_best = mean_accuracy[count_repeat-1]
        accuracy_worst = mean_accuracy[count_repeat-1]
        conf_mat_best = conf_mat
        conf_mat_worst = conf_mat
        fpr_best, tpr_best, AUC_best = fpr, tpr, AUC
        fpr_worst, tpr_worst, AUC_worst = fpr, tpr, AUC

      else:
          if accuracy_best < mean_accuracy[count_repeat-1]:
              conf_mat_best = conf_mat
              accuracy_best = mean_accuracy[count_repeat-1]
              fpr_best, tpr_best, AUC_best = fpr, tpr, AUC

          elif accuracy_worst >= mean_accuracy[count_repeat-1]:
              conf_mat_worst = conf_mat
              accuracy_worst = mean_accuracy[count_repeat-1]
              fpr_worst, tpr_worst, AUC_worst = fpr, tpr, AUC
      return accuracy_best, accuracy_worst, conf_mat_best, conf_mat_worst, fpr_best, tpr_best, AUC_best, fpr_worst, tpr_worst, AUC_worst

data = load_data() 

# Handle missing data
labels = data.pop('label')
clean_df, labels = replace_missing_data(data, labels)

# Defining k-fold parameters
n_split = 5
n_repeat = 10
rskf = k_fold_strat(n_splits= n_split, n_repeats= n_repeat, random_state=None)
floored_val_len = len(clean_df)//n_split

# Initializations before the loop SVM
count = accuracy_best = accuracy_worst = AUC_best = AUC_worst = 0
mean_accuracy, mean_val_accuracy, diff_val_sets , mean_precision, mean_recall, mean_f1, mean_AUC = [], [], [], [], [], [], []
n_comp_overall, C_overall = [], []
tpr, fpr = [], []
conf_mat_worst, conf_mat_best = [], []
conf_mat_total = [[0]*2]*2
fpr_best = tpr_best = fpr_worst = tpr_worst = 0
labels_conf = ['LGG','GBM']
y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat = [], [], [], []
y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF, val_accuracy_repeat_RF = [], [], [], []
count_repeat = 0

# Initializations before the loop RF
count_RF = accuracy_best_RF = accuracy_worst_RF = AUC_best_RF = AUC_worst_RF = 0
mean_accuracy_RF, mean_val_accuracy_RF, diff_val_sets_RF, mean_precision_RF, mean_recall_RF, mean_f1_RF, mean_AUC_RF = [], [], [], [], [], [], []
n_comp_overall_RF, E_overall = [], []
tpr_RF, fpr_RF = [], []
fpr_best_RF = tpr_best_RF = fpr_worst_RF = tpr_worst_RF = 0
conf_mat_worst_RF, conf_mat_best_RF = [], []
conf_mat_total_RF = [[0]*2]*2
labels_conf_RF = ['LGG','GBM']


# Loop to perform a cross-validation split SVM
for train_index, test_index in rskf.split(clean_df, labels):
    count += 1
    
    # Add indices of test set to check for unique sets at the end
    diff_val_sets.append(test_index[0:floored_val_len-1])
    diff_val_sets_RF.append(test_index[0:floored_val_len-1])

    # Get x_train and y_train
    x_train = clean_df.iloc[train_index]
    y_train = labels.iloc[train_index]

    # Get x_test and y_test
    x_test = clean_df.iloc[test_index]
    y_test = labels.iloc[test_index]

    # Do preprocessing
    x_train_scaled, x_test_scaled = pre_processing(x_train, x_test)

    # Initializing the time frame
    start_time = time.time()

    # Perform PCA
    pcs_train, pcs_test, n_components = get_pcs_train_test(x_train_scaled, x_test_scaled, 0.95)

    # Construct SVM classifiers
    degrees = np.linspace(1, 4, 4)
    slacks = np.logspace(-1, 1, 50)

    # Randomized grid search SVM
    parameters = {'kernel': ['poly'], 'C': slacks, 'degree': degrees, 'gamma': ['scale']}
    random_grid_search = GridSearchCV(SVC(), parameters, return_train_score=bool)
    accuracy_grid = random_grid_search.fit(pcs_train, y_train)

    # Calculate the accuracy on the validation set with the best parameters from the grid search SVM
    best_params = random_grid_search.best_params_
    best_score = random_grid_search.best_score_

    # Construct RF classifiers
    n_estimators = [int(x) for x in np.linspace(10, 400, num = 10)]
    bootstrap = ['True']

    # Randomized grid search RF
    parameters_RF = {'n_estimators': n_estimators, 'bootstrap': bootstrap}
    random_grid_search_RF = GridSearchCV(RandomForestClassifier(), parameters_RF, return_train_score=bool)
    accuracy_grid_RF = random_grid_search_RF.fit(pcs_train, y_train)

    # Calculate the accuracy on the validation set with the best parameters from the grid search RF
    best_params_RF = random_grid_search_RF.best_params_
    best_score_RF = random_grid_search_RF.best_score_
    
    # Fit classifier with the best parameter settings RF
    clf_RF = RandomForestClassifier(**best_params_RF)
    clf_RF.fit(pcs_train, y_train) 

    # Fit classifier with the best parameter settings
    clf_SVM = SVC(**best_params, probability = True)
    clf_SVM.fit(pcs_train, y_train)
    
    # Calculate the accuracy on the test set
    y_test_pred = clf_SVM.predict(pcs_test) # Binary classification
    y_test_proba = clf_SVM.predict_proba(pcs_test)[:,1] # Probability classification

    # Calculate the accuracy on the test set
    y_test_pred_RF = clf_RF.predict(pcs_test) # Binary classification
    y_test_proba_RF = clf_RF.predict_proba(pcs_test)[:,1] # Probability classification

    # Calculating the performance per repeat 
    if count % n_split == 0: # checking in which repeat of the cross validation the model is 
      count_repeat += 1

      # Calculating test predictions per repeat for SVM and RF 
      y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat = appending_splits(y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat,  y_test, y_test_pred, y_test_proba, best_score)
      y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF, val_accuracy_repeat_RF = appending_splits(y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF, val_accuracy_repeat_RF,  y_test, y_test_pred_RF, y_test_proba_RF, best_score_RF)

      # Calculating performance metrics per repeat for SVM 
      print(f'SVM: repeat number {count_repeat}')
      print(f'The validation accuracy is {np.mean(val_accuracy_repeat) * 100} %')
      mean_accuracy, mean_precision, mean_recall, mean_f1, mean_AUC, conf_mat, conf_mat_total, fpr, tpr, AUC = evaluate(y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_AUC, conf_mat_total)
      y_test_repeat, y_test_pred_repeat, y_test_proba_repeat = [], [], []

      # Calculating performance metrics per repeat for RF 
      print(f'RF: repeat number {count_repeat}')
      print(f'The validation accuracy is {np.mean(val_accuracy_repeat_RF) * 100} %')
      mean_accuracy_RF, mean_precision_RF, mean_recall_RF, mean_f1_RF, mean_AUC_RF, conf_mat_RF, conf_mat_total_RF, fpr_RF, tpr_RF, AUC_RF = evaluate(y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF, mean_accuracy_RF, mean_precision_RF, mean_recall_RF, mean_f1_RF, mean_AUC_RF, conf_mat_total_RF)
      y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF = [], [], []

      print("--- %s seconds ---" % (time.time() - start_time))

      # Calculate the mean validation accuracy 
      mean_val_accuracy.append(np.round(np.mean(val_accuracy_repeat),3))
      mean_val_accuracy_RF.append(np.round(np.mean(val_accuracy_repeat_RF),3))

      #Needs fixing
      n_comp_overall.append(n_components)
      C_overall.append(best_params.get('C'))
      E_overall.append(best_params_RF.get('n_estimators'))

      # Update best performance (accuracy based)
      accuracy_best, accuracy_worst, conf_mat_best, conf_mat_worst, fpr_best, tpr_best, AUC_best, fpr_worst, tpr_worst, AUC_worst = update_roc(count_repeat,accuracy_best, accuracy_worst,mean_accuracy, conf_mat_best, conf_mat_worst, fpr_best, tpr_best, AUC_best, fpr_worst, tpr_worst, AUC_worst, conf_mat, fpr, tpr, AUC)
      accuracy_best_RF, accuracy_worst_RF, conf_mat_best_RF, conf_mat_worst_RF, fpr_best_RF, tpr_best_RF, AUC_best_RF, fpr_worst_RF, tpr_worst_RF, AUC_worst_RF = update_roc(count_repeat,accuracy_best_RF, accuracy_worst_RF, mean_accuracy_RF, conf_mat_best_RF, conf_mat_worst_RF, fpr_best_RF, tpr_best_RF, AUC_best_RF, fpr_worst_RF, tpr_worst_RF, AUC_worst_RF, conf_mat_RF, fpr_RF, tpr_RF, AUC_RF)

    else:

      y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat = appending_splits(y_test_repeat, y_test_pred_repeat, y_test_proba_repeat, val_accuracy_repeat,  y_test, y_test_pred, y_test_proba, best_score)
      y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF, val_accuracy_repeat_RF = appending_splits(y_test_repeat_RF, y_test_pred_repeat_RF, y_test_proba_repeat_RF, val_accuracy_repeat_RF,  y_test, y_test_pred_RF, y_test_proba_RF, best_score_RF)

      #Needs fixing
      n_comp_overall.append(n_components)
      C_overall.append(best_params.get('C'))
      E_overall.append(best_params_RF.get('n_estimators'))

# Create confusion matrix of worst score, best score and average based on all iterations SVM
conf_mat_average = conf_mat_total/count_repeat
for matrix, title in zip([conf_mat_average, conf_mat_best, conf_mat_worst],['Mean','Best','Worst']):
    confusion_matrix_visualized(matrix ,labels_conf, title) 

# Create AUC plot for best and worst accuracy score SVM
for fpr, tpr, AUC, title in zip([fpr_best,fpr_worst],[tpr_best,tpr_worst],[AUC_best,AUC_worst],['Best','Worst']):
    AUC_visualized(fpr, tpr, AUC, title)

# Creating a list with multiple identical accuracies for the same splits
mean_accuracy_mult = [[ele] * n_split for _,ele in enumerate(mean_accuracy)]
mean_accuracy_mult = [ele for sub in mean_accuracy_mult for ele in sub]
mean_val_accuracy_mult = [[ele] * n_split for _,ele in enumerate(mean_val_accuracy)]
mean_val_accuracy_mult = [ele for sub in mean_val_accuracy_mult for ele in sub]

# Create arrays with mean and std of variable parameters for the test and validation data SVM
acc_unique_test, C_means_test, C_std_test, n_comp_means_test, n_comp_std_test =  param_means(mean_accuracy_mult, C_overall, n_comp_overall)
acc_unique_val, C_means_val, C_std_val, n_comp_means_val, n_comp_std_val =  param_means(mean_val_accuracy_mult, C_overall, n_comp_overall)

# Create overview figure of iterations and chosen hyperparameters SVM
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize = (16,8), gridspec_kw={'width_ratios': [6, 2, 1]})
ax1.plot(mean_accuracy,'b-', label = 'Test data')
ax1.plot(mean_val_accuracy,'r-', label = 'Validation data')
ax1.set_xlabel("Repeat")
ax1.set_ylabel("Accuracy [%]")
ax1.set_title("SVM: Accuracy per repeat for training and validation set")
ax1.grid(axis = 'y', alpha = 0.5)
ax1.legend()

ax2.plot(C_means_test, acc_unique_test,'o-', color = 'b')
ax2.plot(C_means_val, acc_unique_val, 'o-', color = 'r')
ax2.fill_betweenx(acc_unique_test, C_means_test - C_std_test, C_means_test + C_std_test, alpha = 0.1, color = 'b')
ax2.fill_betweenx(acc_unique_val, C_means_val - C_std_val, C_means_val + C_std_val, alpha = 0.1, color= 'r')
ax2.set_xlabel("C")
ax2.set_title("SVM: C parameter")
ax2.grid(alpha = 0.5)

ax3.plot(n_comp_means_test, acc_unique_test, 'o-', color = 'b')
ax3.plot(n_comp_means_val, acc_unique_val, 'o-', color = 'r')
ax3.fill_betweenx(acc_unique_test, n_comp_means_test - n_comp_std_test, n_comp_means_test + n_comp_std_test, alpha = 0.1, color = 'b')
ax3.fill_betweenx(acc_unique_val, n_comp_means_val - n_comp_std_val, n_comp_means_val + n_comp_std_val, alpha = 0.1, color= 'r')
ax3.set_xlabel("n")
ax3.set_title("SVM: N components")
ax3.grid(alpha = 0.5)

# Print mean (and std) scores across all iterations SVM
print(f'SVM: Mean test accuracy across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_accuracy)*100} % ({np.std(mean_accuracy)*100})')
print(f'SVM: Mean test precision across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_precision)*100} % ({np.std(mean_precision)*100})')
print(f'SVM: Mean test recall across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_recall)*100} % ({np.std(mean_recall)*100})')
print(f'SVM: Mean test F1-score across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_f1)*100} ({np.std(mean_f1)*100})')
print(f'SVM: Mean test AUC across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_AUC)} ({np.std(mean_AUC)})')
print(f'SVM: Mean accuracy of best models on inner loop validations =  {np.mean(mean_val_accuracy)*100} % ({np.std(mean_val_accuracy)*100})')

# Create confusion matrix of worst score, best score and average based on all iterations RF
conf_mat_average_RF = conf_mat_total_RF/count_repeat
for matrix2, title2 in zip([conf_mat_average_RF, conf_mat_best_RF, conf_mat_worst_RF],['Mean_RF','Best_RF','Worst_RF']):
  confusion_matrix_visualized(matrix2 ,labels_conf_RF, title2)

# Create AUC plot for best and worst accuracy score SVM
for fpr2, tpr2, AUC2, title2 in zip([fpr_best_RF,fpr_worst_RF],[tpr_best_RF,tpr_worst_RF],[ AUC_best_RF,AUC_worst_RF],['Best','Worst']):
    AUC_visualized(fpr2, tpr2, AUC2, title2)

#creating a list with multiple identical accuracies for the same splits
mean_accuracy_RF_mult = [[ele] * n_split for _,ele in enumerate(mean_accuracy_RF)]
mean_accuracy_RF_mult = [ele for sub in mean_accuracy_RF_mult for ele in sub]
mean_val_accuracy_RF_mult = [[ele] * n_split for _,ele in enumerate(mean_val_accuracy_RF)]
mean_val_accuracy_RF_mult = [ele for sub in mean_val_accuracy_RF_mult for ele in sub]

# Create arrays with mean and std of variable parameters for the test and validation data RF
acc_unique_test_RF, E_means_test_RF, E_std_test_RF, n_comp_means_test_RF, n_comp_std_test_RF =  param_means2(mean_accuracy_RF_mult, E_overall, n_comp_overall)
acc_unique_val_RF, E_means_val_RF, E_std_val_RF, n_comp_means_val_RF, n_comp_std_val_RF =  param_means2(mean_val_accuracy_RF_mult, E_overall, n_comp_overall)

# Create overview figure of iterations and chosen hyperparameters RF
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharey=True, figsize = (16,8), gridspec_kw={'width_ratios': [6, 2, 1]})
ax1.plot(mean_accuracy_RF,'b-', label = 'Test data')
ax1.plot(mean_val_accuracy_RF,'r-', label = 'Validation data')
ax1.set_xlabel("Repeat")
ax1.set_ylabel("Accuracy [%]")
ax1.set_title("Accuracy per repeat for training and validation set RF")
ax1.grid(axis = 'y', alpha = 0.5)
ax1.legend()

ax2.plot(E_means_test_RF, acc_unique_test_RF,'o-', color = 'b')
ax2.plot(E_means_val_RF, acc_unique_val_RF, 'o-', color = 'r')
ax2.fill_betweenx(acc_unique_test_RF, E_means_test_RF - E_std_test_RF, E_means_test_RF + E_std_test_RF, alpha = 0.1, color = 'b')
ax2.fill_betweenx(acc_unique_val_RF, E_means_val_RF - E_std_val_RF, E_means_val_RF + E_std_val_RF, alpha = 0.1, color= 'r')
ax2.set_xlabel("E")
ax2.set_title("RF: E parameter")
ax2.grid(alpha = 0.5)

ax3.plot(n_comp_means_test_RF, acc_unique_test_RF, 'o-', color = 'b')
ax3.plot(n_comp_means_val_RF, acc_unique_val_RF, 'o-', color = 'r')
ax3.fill_betweenx(acc_unique_test_RF, n_comp_means_test_RF - n_comp_std_test_RF, n_comp_means_test_RF + n_comp_std_test_RF, alpha = 0.1, color = 'b')
ax3.fill_betweenx(acc_unique_val_RF, n_comp_means_val_RF - n_comp_std_val_RF, n_comp_means_val_RF + n_comp_std_val_RF, alpha = 0.1, color= 'r')
ax3.set_xlabel("n")
ax3.set_title("RF: N components")
ax3.grid(alpha = 0.5)

# Print mean (and std) scores across all iterations RF
print(f'RF: Mean test accuracy across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_accuracy_RF)*100} % ({np.std(mean_accuracy_RF)*100})')
print(f'RF: Mean test precision across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_precision_RF)*100} % ({np.std(mean_precision_RF)*100})')
print(f'RF: Mean test recall across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_recall_RF)*100} % ({np.std(mean_recall_RF)*100})')
print(f'RF: Mean test F1-score across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_f1_RF)*100} % ({np.std(mean_f1_RF)*100})')
print(f'RF: Mean AUC across {count_repeat} repeats with {n_split} splits =  {np.mean(mean_AUC_RF)} ({np.std(mean_AUC_RF)})')
print(f'RF: Mean accuracy of best model on inner loop validations =  {np.mean(mean_val_accuracy_RF)*100} % ({np.std(mean_val_accuracy_RF)*100})')