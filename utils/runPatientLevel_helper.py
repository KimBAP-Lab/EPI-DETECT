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

    # figure out which patients were flagged as possible epilpsy
    positive_ids = np.unique(scores[scores.model_answer==1].PatientID.to_list())
    
    #------------------------------------------------------------------------
    # Load model 
    #------------------------------------------------------------------------

    filepath = opj(model_path,'parseFP_epilepsy_lr.sav')

    with open(filepath, "rb") as f:
        clf = pickle.load(f)
    
    #------------------------------------------------------------------------
    # set up features
    #------------------------------------------------------------------------
    
    # set up time bins. we want 3 month intervals up to two years
    blockStarts, blockEnds, blockLabels = trend_helper.makeBins(window='3month',nYears=2)
    
    admitDates = patients[['PatientID','admit_date']]

    # extract the time-based features for everyone in these bins
    timeTrends = trend_helper.parseScores(ScoreData=scores, AdmitDates=admitDates, id_col='PatientID', blockStarts=blockStarts, blockEnds=blockEnds, blockLabels=blockLabels, score_date_col='Date', model_answer_col='model_answer', prob_col='prob_YES', admit_date_col='admit_date')

    # format for running the classificatin algorithm 
    regData, cols = trend_helper.gen_regFeats(data=timeTrends, id_col='PatientID', time_col='Time Interval', blockLabels=blockLabels)
    
    #------------------------------------------------------------------------
    # evaluate
    #------------------------------------------------------------------------
    ids = regData['PatientID']
    testData = regData[cols[1:]]

    threshold = 0.3
    regData['prediction'] = (clf.predict_proba(testData)[:,1] >= threshold).astype(int)
    regData['probability'] = clf.predict_proba(testData)[:,1]

    patientLevel_results = pd.merge(patients[['PatientID','admit_date']],regData[['PatientID','prediction','probability']],on='PatientID',how='left').sort_values(by='PatientID').reset_index(drop=True)

    print('Identified {} patients with aquired epilepsy out of {} patients'.format(len([patientLevel_results[patientLevel_results.prediction==1]]),len(patients)))

    # fill 0 in for any patients that were labeled non epilepsy from the start (wouldn't have been given a prediction
    patientLevel_results = patientLevel_results.fillna(0)
    
    return patientLevel_results

    
    
    

    