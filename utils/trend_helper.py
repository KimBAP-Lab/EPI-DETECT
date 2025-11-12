"""
Created by Justin Wheelock (jrw79)
2025 August 15
"""

# defines functions for extracting time-based features from epilepsy NLP output 

import pandas as pd
import numpy as np
import time
from datetime import datetime
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

def makeBins(**kwargs):
    """
    Function for creating both the array of indices and labels for time bins, relative to an injury onset 
    These indices will then be used for extracting time-based features, plotting, etc 

    Inputs:
    window: width of each time time, REQUIRED
        week - define 1-week bins from injury onset to the specific endpoint
        month - define 1-month bins from injury onset to the specific endpoint
        3month - define 3-month bins from injury onset to the specific endpoint
        6month - define 6-month bins from injury onset to the specific endpoint
    nYears: the span of time you need these bins for, REQUIRED
        # integer value for number of years
    """

    for key, value in kwargs.items():
        if key == 'window':
            window = value
        if key == 'nYears':
            nYears = value

    startTime = 0
    # end time will vary depending on window and nYears
    if window=='week':
        endTime = nYears*52+1
        step = 1
        l = 'week'
    if window=='month':
        endTime = nYears*12+1
        step = 1
        l = 'month'
    if window=='3month':
        endTime = nYears*12+1
        step = 3
        l = 'month'
    if window=='6month':
        endTime = nYears*12+1
        step = 6
        l = 'month'

    blockStarts = np.arange(startTime,endTime-step,step)
    blockEnds = np.arange(startTime+step,endTime,step)

    blockLabels = []

    for t in range(0,len(blockStarts)):
        thisLabel = str(blockStarts[t])+'to'+str(blockEnds[t])+l
        blockLabels.append(thisLabel)

    print(f'constructed {len(blockLabels)} time blocks in {step} {l} intervals up to {nYears} year(s)')

    return blockStarts, blockEnds, blockLabels



def parseScores(**kwargs):
    """
    Function for extracting time-based features from the NLP output. Uses relativeDelta to filter the scores for each patient to a given time window relative to their admission.

    Inputs:
        ScoreData: dataframe containing all scores we want to parse by time. REQUIRED
            ** this absolutely needs to have columns for patientID, date, model_answer, and probability
        AdmitDates: dataframe containing the patient id and admission date for each patient. REQUIRED
            ** needs a column for patient id (must match name of ID column in ScoreData) and reference date
        blockStarts: array of indices for where each time bin begins. REQUIRED
        blockEnds: array of indices for where each time bin ends. REQUIRED
        blockLabels: array of labels for each bin
        id_col: REQUIRED unless column is 'BDSPPatientID'
            column name for patient IDs. MUST BE SAME FOR BOTH.
        score_date_col: REQUIRED unless column is 'date'
            column name for the dates in the score sheet
        model_answer_col: REQUIRED unless column is 'model_answer'
            column name for the model output in the score sheet
        prob_col: REQUIRED unless column is 'prob_YES'
            column name for the epilepsy probability column in the score sheet
        admit_date_col: REQUIRED unless column is 'admit_date'
            column name for reference date in the AdmitDates dataframe
    """

    # defaults 
    id_col = 'BDSPPatientID'
    score_date_col = 'date'
    model_answer_col = 'model_answer'
    prob_col = 'prob_YES'
    admit_date_col = 'admit_date'
    
    for key, value in kwargs.items():
        if key == 'ScoreData':
            scoreData = value
        if key == 'AdmitDates':
            admitDates = value
        if key == 'id_col':
            id_col = value
        if key == 'blockStarts':
            blockStarts = value
        if key == 'blockEnds':
            blockEnds = value
        if key == 'blockLabels':
            blockLabels = value
        if key == 'score_date_col':
            score_date_col = value
        if key == 'model_answer_col':
            model_answer_col = value
        if key == 'prob_col':
            prob_col = value
        if key == 'admit_date_col':
            admit_date_col = value

    timeTrends = pd.DataFrame()

    print(np.shape(scoreData))

    ptList = list(np.unique(scoreData[id_col]))

    # loop through each block 
    for b in range(0,len(blockLabels)):
        thisBlock = blockLabels[b]
        blockStart = blockStarts[b]
        blockEnd = blockEnds[b]

        # identify the true date of the beginning and end of this time window 
        if 'week' in thisBlock:
        # add the start and stop of this window to admit date df 
            admitDates['Date_before'] = admitDates[admit_date_col].astype("datetime64[ns]").apply(lambda x: x + relativedelta(weeks=int(blockStart)))
            admitDates['Date_after'] = admitDates[admit_date_col].astype("datetime64[ns]").apply(lambda x: x + relativedelta(weeks=int(blockEnd)))
        if 'month' in thisBlock:
            admitDates['Date_before'] = admitDates[admit_date_col].astype("datetime64[ns]").apply(lambda x: x + relativedelta(months=blockStart))
            admitDates['Date_after'] = admitDates[admit_date_col].astype("datetime64[ns]").apply(lambda x: x + relativedelta(months=blockEnd))
            
        # merge with scores
        scores = pd.merge(admitDates[[id_col,'Date_before','Date_after']], scoreData, on=id_col, how='right').drop_duplicates()    
        # get any scores from within the time window of interest
        scores = scores[(scores[score_date_col].astype('datetime64[ns]') >= scores['Date_before'].astype('datetime64[ns]')) & (scores[score_date_col].astype('datetime64[ns]') <= scores['Date_after'].astype('datetime64[ns]'))]

        scores = scores.drop(columns=['Date_before','Date_after']) # don't need these any more

        # now loop through all patients in this dataset and extract the time-based features for them 
        for s in range(0,len(ptList)):
            sub_scores = scores[scores[id_col]==ptList[s]]
            # number of hits in this time window
            n = len(sub_scores[sub_scores[model_answer_col]==1])
            # create a temp dataframe for this patient
            temp = pd.DataFrame()
            temp.at[s,id_col] = ptList[s]
            temp.at[s,'Time Interval'] = thisBlock
            temp.at[s,'t'] = b
            temp.at[s,'Number of Hits'] = n
            if len(sub_scores)>0: # if they have scores in this window, pull some quantitative info from them 
                temp.at[s,'Highest Probability'] = np.mean(sub_scores.sort_values(by=prob_col,axis=0,ascending=False)[prob_col][0:1])
            else:
                temp.at[s,'Highest Probability'] = 0.2138259917276594 # the baseline probability w/ no information
            timeTrends = pd.concat([timeTrends,temp])
            
    return timeTrends


def gen_regFeats(**kwargs):
    """
    Function to take the features generated in the above function and format them in a way that is suitable for a regression 

    Inputs: 
        data: should be the timeTrends exported with the above function. REQUIRED
        id_col: REQUIRED unless column is 'BDSPPatientID'
        time_col
        blockLabels: exported from the function to generate your time bins. otherwise it will get them automatically as the unique values
    """

    # set any defaults 
    id_col = 'BDSPPatientID'
    
    for key, value in kwargs.items():
        if key == 'data':
            data = value
        if key == 'id_col':
            id_col = value
        if key == 'time_col':
            time_col = value
        if key == 'blockLabels':
            blockLabels = value

    ptList = list(np.unique(data[id_col]))

    # initialize dataframe
    regData = pd.DataFrame()
    for s in range(0,len(ptList)):
        subj = ptList[s]
    
        subjData = data[data[id_col]==subj].reset_index()
        # loop through time blocks
        temp = pd.DataFrame()
        temp.at[s,id_col] = subj
        
        for b in range(0,len(blockLabels)):
            thisBlock = blockLabels[b]
            subjData_thisBlock = subjData[subjData[time_col]==thisBlock].reset_index()
            # add probability for this time block
            temp.at[s,'p_'+thisBlock] = subjData_thisBlock['Highest Probability'][0]
            temp.at[s,'n_'+thisBlock] = subjData_thisBlock['Number of Hits'][0]
        
        regData = pd.concat([regData,temp])
    cols = list(regData)

    return regData, cols
            
            
        
    
    