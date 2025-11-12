"""
Created by Justin Wheelock (justin.wheelock@yale.edu)
22 October 2025
Adapted from published code in M. Fernandes Et al 2023, Epilepsia
"""

import sys 
import os 
from os.path import join as opj 
import re 
import nltk
import numpy as np 
import pandas as pd
from datetime import datetime, date, timedelta
import time
from dateutil.relativedelta import relativedelta
import dill
import warnings
from future import standard_library
standard_library.install_aliases()
import builtins
warnings.filterwarnings("ignore")
from utils.notes_function import notes_fnc

def build_cohort_deidentified(**kwargs):
    """
    Requirements: 
        patients: dataframe containing columns "PatientID" and "admit_date"
        notes: dataframe containing all of the notes themselves. columns should be "PatientID", "Date", "NoteID", NoteTXT"
    """

    for key, value in kwargs.items():
        if key == 'patients':
            d = value
        if key == 'notes':
            notes = value
        if key =='path':
            path = value

    # create barriers for time window (injury to 2 years, ignoring the first 7 days)
    d['Date_before'] = d['admit_date'].astype("datetime64[ns]").apply(lambda x: x + relativedelta(days=7))
    d['Date_after'] = d['admit_date'].astype("datetime64[ns]").apply(lambda x: x + relativedelta(years=2))

    d = d.drop_duplicates()

    # now set up our notes properly and filter to the time window we need
    notes.Date = notes.Date.astype("datetime64[ns]")
    notes = pd.merge(d[['PatientID','Date_before','Date_after']], notes, on =['PatientID'], how='outer').drop_duplicates()
    
    notes = notes[(notes.Date >= notes.Date_before) & (notes.Date <= notes.Date_after)]
    
    notes = notes.drop(columns=['Date_before','Date_after'])

    col_notes = 'NoteTXT'

    # preprocess and extract binary features from these notes
    df_notes = notes_fnc(notes, col_notes, path) # check inside function 

    df_notes.Date = df_notes.Date.astype("datetime64[ns]")
    df_notes = df_notes.drop(columns='NoteTXT') # can uncomment if you want to retain the note text itself 

    return df_notes



def assign_scores(**kwargs):
    """
    Requirements: 
        df_notes: output from build_cohort_deidentified containing the 
        path_train: folder containing the original training dataframes and model from the baseline phenotyping algorithm
    """

    for key, value in kwargs.items():
        if key == 'df_notes':
            df_test = value
        if key == 'path_train':
            path_train = value

    #sys.path.insert(0, path_train) # insert path

    #------------------------------------------------------------------------
    # Load model - check directory folder
    #------------------------------------------------------------------------
    
    # model_name = 'lr_text_only'
    # features_name = 'no_prodigy'
    # class_name = 'binary'
    
    # ## Best model
    # filename = '{}_{}_{}_model.sav'.format(model_name,features_name,class_name)
    # filepath = opj(path_train,filename)

    ## repaired version of 'lr_text_only_no_prodigy_binary_model.sav': was optimized for python 2 and gave errors when using in python 3
    filename = 'lr_text_only_py3_repaired.sav'
    filepath = opj(path_train,filename)
    
    # import model
    clf = dill.load(open(filepath, 'rb'))

    #------------------------------------------------------------------------
    # import reference training data
    #------------------------------------------------------------------------
    X_train = pd.read_csv(opj(path_train,'X_train.csv'))
    y_train = pd.read_csv(opj(path_train,'y_train.csv'))
    
    exclude = list(['convulsions seizures','epilepsy and recurrent seizures',
                'syncope','n_icds', 'Age', 'Sex',
                'n_meds', 'Acetazolamide', 'Brivaracetam', 'Cannabidiol',
                'carbamezapine', 'cenobamate', 'clobazam', 'clonazepam', 'clorazepate',
                'diazepam', 'eslicarbazepine', 'ethosuximide', 'ezogabine', 'felbamate',
                'gabapentin', 'ketamine', 'lacosamide', 'lamotrigine', 'levetiracetam',
                'lorazepam', 'methsuximide', 'midazolam', 'oxcarbazepine', 'perampanel',
                'phenobarbital', 'phenytoin', 'pregabalin', 'primidone', 'rufinamide',
                'tiagabine', 'topiramate', 'valproic acid', 'zonisamide'])

    for i in exclude:
        X_train = X_train.drop(columns=i)
        
    #%% Features for modeling #########################################
    
    outcome = 'outcome'
    labels = ['NO', 'YES']
    #------------------------------------------------------------------------
    # adjust format of testing data to match training set
    #------------------------------------------------------------------------
    df_test = df_test.fillna(0) # fill any missing features
    df_test = df_test.loc[:, ~df_test.columns.duplicated()] # drop any duplicated columns
    X_test = df_test.drop(columns=['PatientID', 'Date'])

    cols_missing = list(X_train.columns[~(X_train.columns.isin(X_test.columns))])
    
    for i in cols_missing:
        X_test[i] = 0
    
    X_test = X_test[list(X_train.columns)]

    #------------------------------------------------------------------------
    # Test model
    #------------------------------------------------------------------------
    
    y_train_pred = clf.predict_proba(X_train)[:,1]
    y_test_pred = clf.predict_proba(X_test)[:,1]
    
    #%% Threshold functions #######################################################
    
    # Optimal Threshold for Precision-Recall Curve (Imbalanced Classification)
    
    from sklearn.metrics import precision_recall_curve
        
    def optimal_threshold_auc(target, predicted):
        precision, recall, threshold = precision_recall_curve(target, predicted)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        return threshold[ix]
      
    # Threshold in train
    threshold = optimal_threshold_auc(y_train, y_train_pred) 
    
    y_pred = (clf.predict_proba(X_test*1)[:,1] >= threshold).astype(int)
    
    probs = clf.predict_proba(X_test)
    
    # Assign scores
    
    df_scores = pd.concat([df_test, pd.DataFrame(probs,columns=['prob_NO','prob_YES'])], axis = 1)
    df_scores = pd.concat([df_scores, pd.DataFrame(y_pred, columns=['model_answer'])], axis = 1)
    
    df_scores.to_csv(opj(path_train,'dataset_with_baseline_scores.csv'), index=False)
    
    return df_scores
    
    
        
        