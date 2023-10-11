# The Challenge

# # The sinking of the Titanic is one of the most infamous shipwrecks in history.
# # On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg.
# # Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.
# # While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
# # In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” 
# # using passenger data (ie name, age, gender, socio-economic class, etc).

################################################################################################################################################################################

# Dataset Description

# Overview

# The data has been split into two groups:

# training set (train.csv)
# test set (test.csv)
# The training set should be used to build your machine learning models. 
# For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. 
# Your model will be based on “features” like passengers’ gender and class. 
# You can also use feature engineering to create new features.

# The test set should be used to see how well your model performs on unseen data. 
# For the test set, we do not provide the ground truth for each passenger. 
# It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

# We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.

################################################################################################################################################################################

# Data Dictionary

# Variable	     Definition	                                     Key
# survival	     Survival	                                     0 = No, 1 = Yes
# pclass	     Ticket class	                                 1 = 1st, 2 = 2nd, 3 = 3rd
# sex	         Sex	
# Age	         Age in years	
# sibsp	         # of siblings / spouses aboard the Titanic	
# parch	         # of parents / children aboard the Titanic	
# ticket         Ticket number	
# fare	         Passenger fare	
# cabin	         Cabin number	
# embarked	     Port of Embarkation	                         C = Cherbourg, Q = Queenstown, S = Southampton

################################################################################################################################################################################

# Variable Notes

# pclass: A proxy for socio-economic status (SES)
# # 1st = Upper
# # 2nd = Middle
# # 3rd = Lower

# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

# sibsp: The dataset defines family relations in this way...
# # Sibling = brother, sister, stepbrother, stepsister
# # Spouse = husband, wife (mistresses and fiancés were ignored)

# parch: The dataset defines family relations in this way...
# # Parent = mother, father
# # Child = daughter, son, stepdaughter, stepson
# # Some children travelled only with a nanny, therefore parch=0 for them.

################################################################################################################################################################################
################################################################################################################################################################################

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score 
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

################################################################################################################################################################################

def data_info(train, test, df):
    print('\n' + '-'*50)

    print('Info about train:')
    print(train.info())
    
    print('\nInfo about test:')
    print(test.info())

    print('\nInfo about train + test:')
    print(df.info())

    print('\n'+ '-'*50 + '\n')

    print('\nNull values by columns in train:')
    print(train.isnull().sum())
    
    print('\nNull values by columns in test:')
    print(test.isnull().sum())

    print('\nNull values by columns in complexity:')
    print(df.isnull().sum())

    print('-'*50 + '\n')

def data_visualization(df):
    sns.countplot(df, x='Pclass', hue='Survived').tick_params(bottom=False)
    plt.show()
    
    sns.countplot(df, x='Sex', hue='Survived').tick_params(bottom=False)
    plt.show()
    
    fig, axs = plt.subplots(ncols=2)
    sns.histplot(df, x='Age', kde=True, hue='Survived', ax=axs[0]).tick_params(bottom=False)
    sns.boxplot(df, x='Age', ax=axs[1]).tick_params(bottom=False)
    plt.show()
    
    fig, axs = plt.subplots(ncols=2)
    sns.histplot(df, x='SibSp', binwidth=1, hue='Survived', ax=axs[0]).tick_params(bottom=False)
    sns.boxplot(df, x='SibSp', ax=axs[1]).tick_params(bottom=False)
    plt.show()
    
    fig, axs = plt.subplots(ncols=2)
    sns.histplot(df, x='Parch', binwidth=1, hue='Survived', ax=axs[0]).tick_params(bottom=False)
    sns.boxplot(df, x='Parch', ax=axs[1]).tick_params(bottom=False)
    plt.show()
    
    fig, axs = plt.subplots(ncols=2)
    sns.histplot(df, x='Fare', kde=True, hue='Survived', ax=axs[0]).tick_params(bottom=False)
    sns.boxplot(df, x='Fare', ax=axs[1]).tick_params(bottom=False)
    plt.show()
    
    sns.countplot(df, x='Embarked', hue='Survived').tick_params(bottom=False)
    plt.show()
        
    sns.countplot(df, x='InCabin', hue='Survived').tick_params(bottom=False)
    plt.show()
    
    sns.countplot(df, x='Deck', hue='Survived').tick_params(bottom=False)
    plt.show()

    sns.countplot(df, x='Title', hue='Survived').tick_params(bottom=False)
    plt.show()

    sns.countplot(df, x='Children', hue='Survived').tick_params(bottom=False)
    plt.show()

    sns.countplot(df, x='Aged', hue='Survived').tick_params(bottom=False)
    plt.show()

    sns.countplot(df, x='Alone', hue='Survived').tick_params(bottom=False)
    plt.show()

def fill_null_values(df1, col_category):
    df_1 = df1.copy()

    idx_list = df_1.index
    col_list = df_1.columns.to_list()

    le_col = []
    for col in col_category:
        le = LabelEncoder()
        df_1[col] = le.fit_transform(df_1[col])
        df_1[col] = df_1[col].astype('int64')
        le_col.append([le,col])

    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=8, weights="distance")
    df_1 = imputer.fit_transform(df_1.to_numpy())

    df_1 = pd.DataFrame(df_1, columns=col_list, index=idx_list)

    for lecol in le_col:
        le, col = lecol[0], lecol[1]
        df_1[col] = df_1[col].astype('int64')
        df_1[col] = le.inverse_transform(df_1[col])
        df_1[col] = df_1[col].astype('object')

    return df_1

def drop_outliers(df1, df2, columns_outliers):
    df_1 = df1.copy()

    # create a index list for the outliers
    idx_outlier = []
    
    for col in columns_outliers:  
        # 1st quartile (25%)
        q25 = np.percentile(df_1[col], 25)
        # 3rd quartile (75%)
        q75 = np.percentile(df_1[col],75)
        # interquartile
        iq = q75 - q25

        # lower bound
        v_min = q25-(1.5*iq)
        # upper bound
        v_max = q75+(1.5*iq)
        
        # determining the index values of outliers
        idx_outlier_col = df_1[(df_1[col] < v_min) | (df_1[col] > v_max)].index
        
        # appending the list of outliers 
        idx_outlier.extend(idx_outlier_col)

    # remove the duplicates from the list
    idx_outlier = list(dict.fromkeys(idx_outlier))
    
    # remove outliers of dataset
    df_1 = df_1.drop(idx_outlier, axis = 0)
    df_2 = df2.drop(idx_outlier, axis = 0) 
    return df_1, df_2

def segregate_category(df1, col_category):
    # create columns to separate the columns for a classification
    # i used OneHotEncoder because i dont want the bias for category types
    onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), col_category)], remainder='passthrough')
    df_1 = onehotencoder.fit_transform(df1)

    # return to dataframe
    df_1 = pd.DataFrame(df_1.toarray(), columns=onehotencoder.get_feature_names_out(), index=df1.index)
    name_col = []
    for col in df_1.columns:
        if col.startswith('OneHot'):
            name_col.append(col[8:])
        elif col.startswith('remainder'):
            name_col.append(col[11:])
    df_1.columns = name_col
    return df_1

# Doesnt used because i didnt see best results
# def segregate_folders(model, X_train, X_valid, y_train, y_valid, seed_value, splits_n=15):
    
#     nkf = KFold(n_splits=splits_n, shuffle=True, random_state=seed_value)

#     acc_score = 0
#     mf = model
#     for train_idx, test_idx in nkf.split(X_train, y_train):
#         m = model
#         m.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
#         m_acc_score = accuracy_score(m.predict(X_valid), y_valid)
#         if m_acc_score > acc_score:
#             acc_score = m_acc_score
#             mf = m
    
#     return mf

def Logistic_Regression(X_train, X_valid, y_train, y_valid, seed_value):

    # define the model to use and training the model
    clf = LogisticRegression(random_state=seed_value)
    clf.fit(X_train, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid, y_valid)
    print("Logistic Regression Free Accuracy: {0:.2f}%".format(acc_free*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf.predict(X_valid))}\n')
    print('-'*50)

    # Tunning model

    # define grid 
    grid = { 
        'penalty':['l2', 'l1'],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': range(10,1000, 10),
        'random_state': [seed_value]
    }

    best_acc_tunned = 0
    best_scoring = ''
    best_cv = 0
    for s in ['roc_auc', 'accuracy']:
        for i in range(2,5):
            grid_search = GridSearchCV(clf,
                                       grid, 
                                       cv=i,
                                       scoring=s)
            
            grid_search.fit(X_train, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = LogisticRegression(**best_params)
            clf_tunned.fit(X_train, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('\nBest Params for Logistic Regression')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = LogisticRegression(**best_params_all)
    clf_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid, y_valid)
    print("\nLogistic Regression Tunned Accuracy: {0:.2f}%".format(acc_tunned*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf_tunned.predict(X_valid))}\n')
    return clf_tunned

def cost_pruning_tree(X_train, X_valid, y_train, y_valid, seed_value):

    # found the cost for pruning the tree
    ccp_alphas = DecisionTreeClassifier(random_state=seed_value)\
                 .cost_complexity_pruning_path(X_train, y_train).ccp_alphas
    
    # create the condition of with Decision Tree
    cond_dtr = []
    for ccp_alpha in ccp_alphas:
        dtr = DecisionTreeClassifier(random_state=seed_value, ccp_alpha=ccp_alpha)
        dtr.fit(X_train, y_train)
        cond_dtr.append(dtr)
    
    # found the score for with condition
    test_scores = [dtr.score(X_valid, y_valid) for dtr in cond_dtr]
    
    # define the best alpha for the lower cost
    best_ccp_alpha = ccp_alphas[test_scores.index(max(test_scores))]
    return best_ccp_alpha

def Decision_Tree(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value):

    # define the model to use and training the model
    clf = DecisionTreeClassifier(random_state=seed_value)
    clf.fit(X_train, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid, y_valid)
    print("Decision Tree Free Accuracy: {0:.2f}%".format(acc_free*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf.predict(X_valid))}\n')
    print('-'*50)

    # see the tree
    # tree.plot_tree(clf)
    
    # Tunning the Tree

    # define grid 
    grid = { 
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': [None, 'sqrt','log2'],
        'max_depth': [None, range(5,12)],
        'random_state': [seed_value],
        'ccp_alpha': [0, ccp_alfa]
    }

    best_acc_tunned = 0
    best_scoring = ''
    best_cv = 0
    for s in ['roc_auc', 'accuracy']:
        for i in range(4,6):
            grid_search = GridSearchCV(clf,
                                       grid, 
                                       cv=i,
                                       scoring=s)
            
            grid_search.fit(X_train, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = DecisionTreeClassifier(**best_params)
            clf_tunned.fit(X_train, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('\nBest Params for Decision Tree Classifier')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = DecisionTreeClassifier(**best_params_all)
    clf_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid, y_valid)
    print("\nDecision Tree Tunned Accuracy: {0:.2f}%".format(acc_tunned*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf_tunned.predict(X_valid))}\n')

    # see the tree
    # tree.plot_tree(clf_tunned)

    return clf_tunned

def Random_Forest(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value):

    # define the model to use and training the model
    clf = RandomForestClassifier(random_state=seed_value)
    clf.fit(X_train, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid, y_valid)
    print("Random Forest Free Accuracy: {0:.2f}%".format(acc_free*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf.predict(X_valid))}\n')
    print('-'*50)

    # Tunning the Tree

    # define grid 
    grid = { 
        'n_estimators': range(50, 1000, 50),
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt','log2'],
        'max_depth': [None, range(4,12)],
        'random_state': [seed_value],
        'ccp_alpha': [0, ccp_alfa]
    }

    best_acc_tunned = 0
    best_scoring = ''
    best_cv = 0
    for s in ['roc_auc', 'accuracy']:
        for i in range(3,6):
            grid_search = GridSearchCV(clf,
                                       grid, 
                                       cv=i,
                                       scoring=s)
            
            grid_search.fit(X_train, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = RandomForestClassifier(**best_params)
            clf_tunned.fit(X_train, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('\nBest Params for Random Forest Classifier')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = RandomForestClassifier(**best_params_all)
    clf_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid, y_valid)
    print("\nRandom Forest Tunned Accuracy: {0:.2f}%".format(acc_tunned*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf_tunned.predict(X_valid))}\n')
    return clf_tunned

def XGBoost(X_train, X_valid, y_train, y_valid, seed_value):

    # define the model to use and training the model
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # find the score of model based in validation data
    acc_free = model.score(X_valid, y_valid)
    print("XGBoost Free Accuracy: {0:.2f}%".format(acc_free*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, model.predict(X_valid))}\n')
    print('-'*50)

    # Tunning Model

    # define grid 
    grid = { 
        'eval_metric':['logloss', 'mlogloss', 'auc'],
        'n_estimators': range(100, 300, 10),
        'learning_rate': np.arange(0.1, 0.5, 0.1),
        'max_depth': range(4,7),
        'random_state': [seed_value]
    }

    # {'max_depth': hp.quniform("max_depth", 3, 18, 1),
    #     'gamma': hp.uniform ('gamma', 1,9),
    #     'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
    #     'reg_lambda' : hp.uniform('reg_lambda', 0,1),
    #     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
    #     'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
    #     'n_estimators': 180,
    #     'seed': 0
    # }

    best_acc_tunned = 0
    best_scoring = ''
    best_cv = 0
    for s in ['roc_auc', 'accuracy']:
        for i in range(3,6):
            grid_search = GridSearchCV(model,
                                    grid, 
                                    cv=i,
                                    scoring=s)
            
            grid_search.fit(X_train, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            model_tunned = XGBClassifier(**best_params)
            model_tunned.fit(X_train, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = model_tunned.score(X_valid, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('\nBest Params for XGBoost Classifier')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    model_tunned = XGBClassifier(**best_params_all)
    model_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = model_tunned.score(X_valid, y_valid)
    print("\nXGBoost Tunned Accuracy: {0:.2f}%".format(acc_tunned*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, model_tunned.predict(X_valid))}\n')
    return model

def SVM_SVC(X_train, X_valid, y_train, y_valid, seed_value):
    # define the model to use and training the model
    clf = SVC(random_state=seed_value)
    clf.fit(X_train, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid, y_valid)
    print("SVC Free Accuracy: {0:.2f}%".format(acc_free*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf.predict(X_valid))}\n')
    print('-'*50)

    # Tunning the Tree

    # define grid 
    grid = { 
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': range(3, 6),
        'random_state': [seed_value]
    }

    best_acc_tunned = 0
    best_scoring = ''
    best_cv = 0
    for s in ['roc_auc', 'accuracy']:
        for i in range(3,6):
            grid_search = GridSearchCV(clf,
                                       grid, 
                                       cv=i,
                                       scoring=s)
            
            grid_search.fit(X_train, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = SVC(**best_params)
            clf_tunned.fit(X_train, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('\nBest Params for SVC')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = SVC(**best_params_all)
    clf_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid, y_valid)
    print("\nSVC Tunned Accuracy: {0:.2f}%".format(acc_tunned*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, clf_tunned.predict(X_valid))}\n')
    return clf_tunned

def KNN(X_train, X_valid, y_train, y_valid):
    # define the model to use and training the model
    neigh = KNeighborsClassifier()
    neigh.fit(X_train, y_train)

    # find the score of model based in validation data
    acc_free = neigh.score(X_valid, y_valid)
    print("KNN Free Accuracy: {0:.2f}%".format(acc_free*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, neigh.predict(X_valid))}\n')
    print('-'*50)

    # Tunning the Model

    # define grid 
    grid = { 
        'n_neighbors': range(4, 8),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': range(20,50)
    }

    best_acc_tunned = 0
    best_scoring = ''
    best_cv = 0
    for s in ['roc_auc', 'accuracy']:
        for i in range(3,6):
            grid_search = GridSearchCV(neigh,
                                       grid, 
                                       cv=i,
                                       scoring=s)
            
            grid_search.fit(X_train, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            neigh_tunned = KNeighborsClassifier(**best_params)
            neigh_tunned.fit(X_train, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = neigh_tunned.score(X_valid, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('\nBest Params for KNN')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    neigh_tunned = KNeighborsClassifier(**best_params_all)
    neigh_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = neigh_tunned.score(X_valid, y_valid)
    print("\nKNN Tunned Accuracy: {0:.2f}%".format(acc_tunned*100))
    print(f'Confusion Matrix:\n{confusion_matrix(y_valid, neigh_tunned.predict(X_valid))}\n')
    return neigh_tunned

def Ensemble_Voting(zip_models, X_train, y_train):
    # Making the final model using voting classifier
    model_ensemble = VotingClassifier(
        estimators=list(zip_models), 
        voting='hard')
    
    # training all the model on the train dataset
    model_ensemble.fit(X_train, y_train)
    return model_ensemble

def prediction_scoring(model, X_valid, y_valid, cv_value=5):

    # Predition
    y_pred = model.predict(X_valid)
    acc_score = accuracy_score(y_valid, y_pred)
    roc_score = roc_auc_score(y_valid, y_pred)
    print("Accuracy: {0:.2f}%\nROC Score: {1:.2f}%\n".format(acc_score*100, roc_score*100))

    # Predition with cross validation
    y_pred_cv = cross_val_predict(model, X_valid, y_valid, cv=cv_value)
    acc_score_cv = accuracy_score(y_valid, y_pred_cv)
    roc_score_cv = roc_auc_score(y_valid, y_pred_cv)
    print("With Cross Validation: \nAccuracy: {0:.2f}%\nROC Score: {1:.2f}%\n".format(acc_score_cv*100, roc_score_cv*100))

def result(X_test, model):
    X_test_np = X_test.to_numpy()

    y_pred1 = model.predict(X_test_np)
    
    X_test['Survived'] = y_pred1
    X_test = X_test[['Survived']].copy()
    
    X_test.to_csv('result.csv')
    print('Result finished')

def main():

    # Seed
    seed_value = 0

    # imagem of Titanic Deck Plan
    # https://images.liverpoolmuseums.org.uk/2020-01/titanic-deck-plan-for-titanic-resource-pack-pdf.pdf
    # https://www.encyclopedia-titanica.org/titanic-deckplans/

    # get the dataset for train and test
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # define PassengerId like index
    train = train.set_index('PassengerId', drop=True)
    test = test.set_index('PassengerId', drop=True)

    df = pd.concat([train, test])

    print(df.head())
    print('\n' + '-'*50)
    # Categorical: Survived, Sex, and Embarked
    # Ordinal: Pclass
    # Continous: Age, Fare
    # Discrete: SibSp, Parch

    data_info(train, test, df)

    # working with missing values

    # i'll works in Age, Fare Embarked with KNNImputer after all

    # checked whether the person stayed in the cabin and which ones
    df['InCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    df['Deck'] = df['Cabin'].str[:1]
    df.loc[pd.isnull(df.Deck), 'Deck'] = 'N/D'
    
    # getting the surname and title
    df[['Surname','Title']] = [i.split(',') for i in df['Name'].values]
    
    df.Surname = df.Surname.str.strip()
    # # case the surname in train doesnt in test will receved Others
    # df.loc[~df.Surname.isin(df.loc[pd.isnull(df.Survived), 
    #                                'Surname']
    #                           .to_list()), 'Surname'] = 'Others'

    df.Title = [i.strip().split('.')[0] for i in df.Title.values]
    
    # for understand the relation of title
    for t in df.Title.unique():
        print(f'{t}: {df[df.Title == t].shape[0]}, {df[df.Title == t].Sex.unique()}')
    print('\n' + '-'*50 + '\n')
    df.loc[(df.Title == 'Mme') | (df.Title == 'Ms'), 'Title'] = 'Mrs'
    df.loc[(df.Title != 'Mr') & (df.Title != 'Mrs') & (df.Title != 'Miss'), 'Title'] = 'Others'

    # # case the title in train doesnt in test will receved Others
    # df.loc[~df.Title.isin(df.loc[pd.isnull(df.Survived), 
    #                              'Title']
    #                         .to_list()), 'Title'] = 'Others'

    # another parameters based in Age and Relationships
    df['Children'] = df.Age.apply(lambda x: 1 if x<18 else 0)
    df['Aged'] = df.Age.apply(lambda x: 1 if x>65 else 0)
    df['Alone'] = df.apply(lambda x: 1 if (x.Parch == 0) & (x.SibSp == 0) else 0, axis=1)

    # see the distribution of data by survived
    # data_visualization(df)

    # separate the columns labels and targets
    # the PassengerId, Name and Ticket columns were ignored because 
    # they were interpreted as not necessary for training the model
    X_train = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin']).copy()
    y_train = df[['Survived']].copy()

    category_columns = ['Sex', 'Embarked', 'Deck', 'Surname', 'Title']

    # fill np.nan values with technique KNN Imputer
    X_train = fill_null_values(X_train, category_columns)

    # separate the category columns in multiples columns
    X_train = segregate_category(X_train, category_columns)

    # defined the test dataset
    X_test = X_train.drop(train.index.to_list(), axis = 0)

    # remove outliers from columns
    # as seen in the data visualization stage, the 'Age' and 'Fare' columns 
    # presented outliers that could harm the model
    X_train, y_train = drop_outliers(X_train, y_train, ['Age', 'Fare'])

    # identify the test index
    idx_test = y_train[pd.isnull(y_train.Survived)].index

    # as only the distribution of 'Age' and 'Fare' is 'Gaussian", they will be scaled by StandardScaler
    # the other columns will be applied to MinMaxScaler method
    min_max_col = ['Pclass', 'SibSp', 'Parch']
    std_scal_col = ['Age', 'Fare']
    # invert the values of Pclass because the 1 are better than 3
    X_train['Pclass'] = X_train['Pclass'].rank(method='dense', ascending=False)
    X_train[min_max_col] = MinMaxScaler().fit_transform(X_train[min_max_col])
    X_train[std_scal_col] = StandardScaler().fit_transform(X_train[std_scal_col])

    # remove the index of test for training model
    X_train = X_train.drop(idx_test.to_list(), axis = 0) 
    y_train = y_train.drop(idx_test.to_list(), axis = 0) 
    
    # ML models works better with numpy array
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()

    # create a dataset for validation the train
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=seed_value)

    model_logistic_regression = Logistic_Regression(X_train, X_valid, y_train, y_valid, seed_value)
    print('#'*50 + '\n')
    ccp_alfa = cost_pruning_tree(X_train, X_valid, y_train, y_valid, seed_value)
    model_decision_tree = Decision_Tree(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value)    
    print('#'*50 + '\n')
    model_random_forest = Random_Forest(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value)
    print('#'*50 + '\n')
    model_xgboost = XGBoost(X_train, X_valid, y_train, y_valid, seed_value)
    print('#'*50 + '\n')
    model_svc = SVM_SVC(X_train, X_valid, y_train, y_valid, seed_value)
    print('#'*50 + '\n')
    model_knn = KNN(X_train, X_valid, y_train, y_valid)
    print('#'*50 + '\n')

    names_list = ['lr', 'rf', 'xgb', 'svc', 'knn']
    model_list = [model_logistic_regression, model_random_forest, model_xgboost, model_svc, model_knn]
    model_voting = Ensemble_Voting(zip(names_list, model_list), X_train, y_train)
    
    prediction_scoring(model_voting, X_valid, y_valid, cv_value=5)
    print('-'*50)
    
    result(X_test, model_voting)

# if __name__ == "__main__":
#     main()

################################################################################################################################################################################
################################################################################################################################################################################

# # Results

# seed = 0

################################################################################################################################################################################

# Logistic Regression Free Accuracy: 81.88%
# Confusion Matrix:
# [[91 11]
#  [16 31]]

# --------------------------------------------------

# Best Params for Logistic Regression
# max_iter: 20
# penalty: l2
# random_state: 0
# solver: lbfgs
# cv: 2
# scoring: roc_auc

# Logistic Regression Tunned Accuracy: 82.55%
# Confusion Matrix:
# [[91 11]
#  [15 32]]

################################################################################################################################################################################

# Decision Tree Free Accuracy: 83.22%
# Confusion Matrix:
# [[94  8]
#  [17 30]]

# --------------------------------------------------

# Best Params for Decision Tree Classifier
# ccp_alpha: 0.0025422005287777097
# criterion: gini
# max_depth: None
# max_features: None
# random_state: 0
# cv: 5
# scoring: accuracy

# Decision Tree Tunned Accuracy: 86.58%
# Confusion Matrix:
# [[97  5]
#  [15 32]]

################################################################################################################################################################################

# Random Forest Free Accuracy: 85.23%
# Confusion Matrix:
# [[97  5]
#  [17 30]]

# --------------------------------------------------

# Best Params for Random Forest Classifier
# ccp_alpha: 0
# criterion: gini
# max_depth: None
# max_features: sqrt
# n_estimators: 50
# random_state: 0
# cv: 4
# scoring: accuracy

# Random Forest Tunned Accuracy: 86.58%
# Confusion Matrix:
# [[97  5]
#  [15 32]]

################################################################################################################################################################################

# XGBoost Free Accuracy: 82.55%
# Confusion Matrix:
# [[93  9]
#  [17 30]]

# --------------------------------------------------

# Best Params for XGBoost Classifier
# eval_metric: logloss
# learning_rate: 0.2
# max_depth: 4
# n_estimators: 110
# random_state: 0
# cv: 5
# scoring: roc_auc

# XGBoost Tunned Accuracy: 85.91%
# Confusion Matrix:
# [[95  7]
#  [14 33]]

################################################################################################################################################################################

# SVC Free Accuracy: 81.21%
# Confusion Matrix:
# [[89 13]
#  [15 32]]

# --------------------------------------------------

# Best Params for SVC
# degree: 3
# kernel: linear
# random_state: 0
# cv: 3
# scoring: roc_auc

# SVC Tunned Accuracy: 83.22%
# Confusion Matrix:
# [[90 12]
#  [13 34]]

################################################################################################################################################################################

# KNN Free Accuracy: 79.87%
# Confusion Matrix:
# [[92 10]
#  [20 27]]

# --------------------------------------------------

# Best Params for KNN
# algorithm: ball_tree
# leaf_size: 28
# n_neighbors: 7
# weights: distance
# cv: 4
# scoring: roc_auc

# KNN Tunned Accuracy: 82.55%
# Confusion Matrix:
# [[92 10]
#  [16 31]]

################################################################################################################################################################################

# Voting
# Accuracy: 83.89%
# ROC Score: 79.63%

# With Cross Validation: 
# Accuracy: 83.89%
# ROC Score: 78.48%

################################################################################################################################################################################
################################################################################################################################################################################

# # Final Result

# Voting Accuracy: 73.684%