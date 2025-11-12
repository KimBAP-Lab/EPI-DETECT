"""
Created by Justin Wheelock (justin.wheelock@yale.edu)
22 October 2025
"""

import sys
import os
from os.path import join as opj
import pandas as pd 
import numpy as np
import pickle
from utils import trend_helper

def score(**kwargs):
    """
    Requirements: 
        patients: dataframe containing columns "PatientID" and "admit_date"
        scores: output of baseline algorithm
        temp_path: /utils. place to store any intermediate files
        model_directory: directory containing saved models 
    """

    for key, value in kwargs.items():
        if key == 'patients':
            patients = value
        if key == 'scores':
            scores = value
        if key =='temp_path':
            temp_path = value
        if key == 'model_directory':
            model_path = value

    
    #------------------------------------------------------------------------
    # Load model 
    #------------------------------------------------------------------------

    filepath = opj(model_path,'parseFalsePositives_byNote.sav')

    with open(filepath, "rb") as f:
        clf = pickle.load(f)

    # extract feature columns 
    feature_names = list(clf.feature_names_in_)

    #------------------------------------------------------------------------
    # set up features 
    #------------------------------------------------------------------------
    positive_notes = scores[scores.model_answer==1]
    
    test_df = positive_notes[list(positive_notes)[:-5]]
    
    cols_missing = set(feature_names) - set(list(test_df))

    for i in cols_missing:
        test_df[i] = 0 # every dataset may not have every text feature
    
    test_df = test_df[feature_names]

    
    #------------------------------------------------------------------------
    # evaluate 
    #------------------------------------------------------------------------
    threshold = 0.5
    positive_notes['adjusted_answer'] = (clf.predict_proba(test_df)[:,1] >= threshold).astype(int)
    positive_notes['adjusted_probability'] = clf.predict_proba(test_df)[:,1]

    # now just add these adjusted probabilities back into the main prediction dataframe

    # first isolate all the negatives and label them accordingly
    negative_scores = scores[scores.model_answer!=1]
    negative_scores['adjusted_answer'] = 0
    negative_scores['adjusted_probability'] = scores[scores.model_answer!=1].prob_YES # maintained from baseline model

    keep_cols = ['PatientID','Date','model_answer','prob_YES','adjusted_answer','adjusted_probability']
    final_scores = pd.concat([positive_notes[keep_cols],negative_scores[keep_cols]]).sort_values(by=['PatientID','Date'],ascending=True)

    final_scores = final_scores.rename(columns={'model_answer':'baseline_answer','prob_YES':'baseline_probability'})

    return final_scores
    
    
