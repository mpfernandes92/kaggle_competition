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
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, roc_auc_score 
from xgboost import XGBClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

################################################################################################################################################################################

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
    
    sns.countplot(df, x='InCabin', hue='Survived').tick_params(bottom=False)
    plt.show()
    
    sns.countplot(df, x='Embarked', hue='Survived').tick_params(bottom=False)
    plt.show()

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
    df_1 = df_1.drop(idx_outlier, axis = 0).reset_index(drop=True) 
    df_2 = df2.drop(idx_outlier, axis = 0).reset_index(drop=True) 
    return df_1, df_2

def segregate_category(df1, col_category):
    # create columns to separate the columns for a classification
    # i used OneHotEncoder because i dont want the bias for category types
    onehotencoder = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), col_category)], remainder='passthrough')
    df_1 = onehotencoder.fit_transform(df1)

    # return to dataframe
    df_1 = pd.DataFrame(df_1, columns=onehotencoder.get_feature_names_out())
    name_col = []
    for col in df_1.columns:
        if col.startswith('OneHot'):
            name_col.append(col[8:])
        elif col.startswith('remainder'):
            name_col.append(col[11:])
    df_1.columns = name_col
    return df_1

def Logistic_Regression(X_train, X_valid, y_train, y_valid, seed_value):

    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_train_mod = imputer.fit_transform(X_train)
    X_valid_mod = imputer.fit_transform(X_valid)

    # define the model to use and training the model
    clf = LogisticRegression(random_state=seed_value)
    clf.fit(X_train_mod, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid_mod, y_valid)
    print("Logistic Regression Free Accuracy: {0:.2f}%\n".format(acc_free*100))

    # Tunning model

    # define grid 
    grid = { 
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': range(100,1000, 100),
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
            
            grid_search.fit(X_train_mod, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = LogisticRegression(**best_params)
            clf_tunned.fit(X_train_mod, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid_mod, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('Best Params for Logistic Regression')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = LogisticRegression(**best_params_all)
    clf_tunned.fit(X_train_mod, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid_mod, y_valid)
    print("Logistic Regression Tunned Accuracy: {0:.2f}%\n".format(acc_tunned*100))
    return clf_tunned

def cost_pruning_tree(X_train, X_valid, y_train, y_valid, seed_value):

    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_train_mod = imputer.fit_transform(X_train)
    X_valid_mod = imputer.fit_transform(X_valid)

    # found the cost for pruning the tree
    ccp_alphas = DecisionTreeClassifier(random_state=seed_value)\
                 .cost_complexity_pruning_path(X_train_mod, y_train).ccp_alphas
    
    # create the condition of with Decision Tree
    cond_dtr = []
    for ccp_alpha in ccp_alphas:
        dtr = DecisionTreeClassifier(random_state=seed_value, ccp_alpha=ccp_alpha)
        dtr.fit(X_train_mod, y_train)
        cond_dtr.append(dtr)
    
    # found the score for with condition
    test_scores = [dtr.score(X_valid_mod, y_valid) for dtr in cond_dtr]
    
    # define the best alpha for the lower cost
    best_ccp_alpha = ccp_alphas[test_scores.index(max(test_scores))]
    return best_ccp_alpha

def Decision_Tree(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value):

    # define the model to use and training the model
    clf = DecisionTreeClassifier(random_state=seed_value)
    clf.fit(X_train, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid, y_valid)
    print("Decision Tree Free Accuracy: {0:.2f}%\n".format(acc_free*100))

    # see the tree
    # tree.plot_tree(clf)
    
    # Tunning the Tree

    # define grid 
    grid = { 
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt','log2'],
        'max_depth': range(2,10),
        'random_state': [seed_value],
        'ccp_alpha': [ccp_alfa]
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
    print('Best Params for Decision Tree Classifier')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = DecisionTreeClassifier(**best_params_all)
    clf_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid, y_valid)
    print("Decision Tree Tunned Accuracy: {0:.2f}%\n".format(acc_tunned*100))

    # see the tree
    # tree.plot_tree(clf_tunned)

    return clf_tunned

def Random_Forest(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value):

    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_train_mod = imputer.fit_transform(X_train)
    X_valid_mod = imputer.fit_transform(X_valid)

    # define the model to use and training the model
    clf = RandomForestClassifier(random_state=seed_value)
    clf.fit(X_train_mod, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid_mod, y_valid)
    print("Random Forest Free Accuracy: {0:.2f}%\n".format(acc_free*100))

    # Tunning the Tree

    # define grid 
    grid = { 
        'n_estimators': range(200, 500, 100),
        'max_features': ['sqrt','log2'],
        'max_depth': range(6,10),
        'random_state': [seed_value],
        'ccp_alpha': [ccp_alfa]
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
            
            grid_search.fit(X_train_mod, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = RandomForestClassifier(**best_params)
            clf_tunned.fit(X_train_mod, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid_mod, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('Best Params for Decision Tree Classifier')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = RandomForestClassifier(**best_params_all)
    clf_tunned.fit(X_train_mod, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid_mod, y_valid)
    print("Random Forest Tunned Accuracy: {0:.2f}%\n".format(acc_tunned*100))

    return clf_tunned

def XGBoost(X_train, X_valid, y_train, y_valid, seed_value):

    # define the model to use and training the model
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    # find the score of model based in validation data
    acc_free = model.score(X_valid, y_valid)
    print("XGBoost Free Accuracy: {0:.2f}%\n".format(acc_free*100))

    # Tunning Model

    # define grid 
    grid = { 
        'eval_metric':['mlogloss', 'auc'],
        'use_label_encoder': [False],
        'n_estimators': range(100, 300, 100),
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
    print('Best Params for Decision Tree Classifier')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    model_tunned = XGBClassifier(**best_params_all)
    model_tunned.fit(X_train, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = model_tunned.score(X_valid, y_valid)
    print("Random Forest Tunned Accuracy: {0:.2f}%\n".format(acc_tunned*100))

    return model

def SVM_SVC(X_train, X_valid, y_train, y_valid, seed_value):
    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_train_mod = imputer.fit_transform(X_train)
    X_valid_mod = imputer.fit_transform(X_valid)

    # define the model to use and training the model
    clf = SVC(random_state=seed_value)
    clf.fit(X_train_mod, y_train)

    # find the score of model based in validation data
    acc_free = clf.score(X_valid_mod, y_valid)
    print("SVC Free Accuracy: {0:.2f}%\n".format(acc_free*100))

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
            
            grid_search.fit(X_train_mod, y_train)

            # best params from the grid
            best_params = grid_search.best_params_

            # training the tunning tree
            clf_tunned = SVC(**best_params)
            clf_tunned.fit(X_train_mod, y_train)

            # find the score of tunned model based in validation data
            acc_tunned = clf_tunned.score(X_valid_mod, y_valid)

            if acc_tunned > best_acc_tunned:
                best_acc_tunned = acc_tunned
                best_params_all = best_params
                best_scoring = s
                best_cv = i

    # show the best values
    print('Best Params for SVC')
    for param, value in best_params_all.items():
        print(f"{param}: {value}")
    print(f'cv: {best_cv}')
    print(f'scoring: {best_scoring}')

    # training the tunning tree
    clf_tunned = SVC(**best_params_all)
    clf_tunned.fit(X_train_mod, y_train)

    # find the score of tunned model based in validation data
    acc_tunned = clf_tunned.score(X_valid_mod, y_valid)
    print("SVC Tunned Accuracy: {0:.2f}%\n".format(acc_tunned*100))

    return clf_tunned

def Ensemble_Voting(zip_models, X_train, y_train):
    # Making the final model using voting classifier
    model_ensemble = VotingClassifier(
        estimators=list(zip_models), 
        voting='hard')
    
    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_train_mod = imputer.fit_transform(X_train)

    # training all the model on the train dataset
    model_ensemble.fit(X_train_mod, y_train)
    return model_ensemble

def prediction_scoring(model, X_valid, y_valid, cv_value=5):

    # because X_train had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_valid_mod = imputer.fit_transform(X_valid)

    # Predition
    y_pred = model.predict(X_valid_mod)
    acc_score = accuracy_score(y_valid, y_pred)
    roc_score = roc_auc_score(y_valid, y_pred)
    print("Accuracy: {0:.2f}%\nROC Score: {1:.2f}%\n".format(acc_score*100, roc_score*100))

    # Predition with cross validation
    y_pred_cv = cross_val_predict(model, X_valid_mod, y_valid, cv=cv_value)
    acc_score_cv = accuracy_score(y_valid, y_pred_cv)
    roc_score_cv = roc_auc_score(y_valid, y_pred_cv)
    print("With Cross Validation: \nAccuracy: {0:.2f}%\nROC Score: {1:.2f}%\n".format(acc_score_cv*100, roc_score_cv*100))

def result(min_max_col, std_scal_col, model):
    # predict values for test.csv
    test = pd.read_csv('test.csv')
    test['InCabin'] = test['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
    X_test = test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']).copy()
    X_test = segregate_category(X_test, ['Sex', 'Embarked'])
    X_test[min_max_col] = MinMaxScaler().fit_transform(X_test[min_max_col])
    X_test[std_scal_col] = StandardScaler().fit_transform(X_test[std_scal_col])
    X_test = X_test.to_numpy()

    # because X_test had nan values i used this part for complete values missings
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    X_test_mod = imputer.fit_transform(X_test)

    y_pred1 = model.predict(X_test_mod)
    
    test_pred = test[['PassengerId']].copy()
    test_pred['Survived'] = y_pred1
    
    test_pred.to_csv('result.csv', index=False)
    print('Result finished')

def main():

    # Seed
    seed_value = 1703

    # get the dataset for train
    train = pd.read_csv('train.csv')

    # checked whether the person stayed in the cabin
    train['InCabin'] = train['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

    # see the distribution of data by survived
    # data_visualization(train)

    # separate the columns labels and targets
    # the PassengerId, Name and Ticket columns were ignored because 
    # they were interpreted as not necessary for training the model
    X_train = train.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin']).copy()
    y_train = train[['Survived']].copy()

    # remove outliers from columns
    # as seen in the data visualization stage, the 'Age' and 'Fare' columns 
    # presented outliers that could harm the model
    X_train, y_train = drop_outliers(X_train, y_train, ['Age', 'Fare'])

    # separate the category columns in multiples columns
    X_train = segregate_category(X_train, ['Sex', 'Embarked'])

    # as only the distribution of 'Age' and 'Fare' is "Gaussian", they will be scaled by StandardScaler
    # the other columns will be applied to MinMaxScaler method
    min_max_col = ['Pclass', 'SibSp', 'Parch']
    std_scal_col = ['Age', 'Fare']
    X_train[min_max_col] = MinMaxScaler().fit_transform(X_train[min_max_col])
    X_train[std_scal_col] = StandardScaler().fit_transform(X_train[std_scal_col])

    # ML models works better with numpy array
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()

    # create a dataset for validation the train
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=seed_value)

    model_logistic_regression = Logistic_Regression(X_train, X_valid, y_train, y_valid, seed_value)
    ccp_alfa = cost_pruning_tree(X_train, X_valid, y_train, y_valid, seed_value)
    model_decision_tree = Decision_Tree(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value)    
    model_random_forest = Random_Forest(X_train, X_valid, y_train, y_valid, ccp_alfa, seed_value)
    model_xgboost = XGBoost(X_train, X_valid, y_train, y_valid, seed_value)
    model_svc = SVM_SVC(X_train, X_valid, y_train, y_valid, seed_value)

    names_list = ['lr', 'rf', 'xgb', 'svc']
    model_list = [model_logistic_regression, model_random_forest, model_xgboost, model_svc]
    model_voting = Ensemble_Voting(zip(names_list, model_list), X_train, y_train)
    
    prediction_scoring(model_voting, X_valid, y_valid, cv_value=5)
    
    result(min_max_col, std_scal_col, model_svc)

# if __name__ == "__main__":
#     main()

################################################################################################################################################################################
################################################################################################################################################################################

# # Results

################################################################################################################################################################################

# Logistic Regression Free Accuracy: 87.74%

# Best Params for Logistic Regression
# max_iter: 100
# random_state: 1703
# solver: liblinear
# cv: 2
# scoring: roc_auc
# Logistic Regression Tunned Accuracy: 87.74%

################################################################################################################################################################################

# Decision Tree Free Accuracy: 83.23%

# Best Params for Decision Tree Classifier
# ccp_alpha: 0.007182955788574936
# criterion: gini
# max_depth: 6
# max_features: sqrt
# random_state: 1703
# cv: 2
# scoring: roc_auc
# Decision Tree Tunned Accuracy: 89.68%

################################################################################################################################################################################

# Random Forest Free Accuracy: 80.00%

# Best Params for Decision Tree Classifier
# ccp_alpha: 0.007182955788574936
# max_depth: 8
# max_features: sqrt
# n_estimators: 200
# random_state: 1703
# cv: 4
# scoring: roc_auc
# Random Forest Tunned Accuracy: 89.03%

################################################################################################################################################################################

# XGBoost Free Accuracy: 83.87%

# Best Params for Decision Tree Classifier
# eval_metric: mlogloss
# learning_rate: 0.1
# max_depth: 6
# n_estimators: 200
# random_state: 1703
# use_label_encoder: False
# cv: 4
# scoring: roc_auc
# Random Forest Tunned Accuracy: 83.23%

################################################################################################################################################################################

# SVC Free Accuracy: 87.10%

# Best Params for SVC
# degree: 3
# kernel: poly
# random_state: 1703
# cv: 3
# scoring: accuracy
# SVC Tunned Accuracy: 88.39%

################################################################################################################################################################################

# Ensemble Voting Accuracy: 87.74%
# Ensemble Voting Cross Validation Accuracy: 89.68%

################################################################################################################################################################################
################################################################################################################################################################################

# # Final Result

# Decision Tree Tunned: 77.751%
# SVC Tunned Accuracy: 75.358%