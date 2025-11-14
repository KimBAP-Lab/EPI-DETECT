"""
Created by Justin Wheelock (justin.wheelock@yale.edu)
23 October 2025
"""

import sys
import os
from os.path import join as opj
import pandas as pd 
import numpy as np
from utils import trend_helper
import matplotlib.pylab as plt
import matplotlib.axes as axes
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def plot(**kwargs):
    """
    Requirements: 
        patients: dataframe containing columns "PatientID" and "admit_date"
        scores: output of adjusted scores
        output_path: place to store the figures
    """

    for key, value in kwargs.items():
        if key == 'patients':
            patients = value
        if key == 'scores':
            scores = value
        if key =='output_path':
            output_path = value

    if not os.path.isdir(output_path):
        os.mkdir(output_path) # set up folder for figures if it doesn't exist
        
    # set up time bins. we want 1 week intervals up to two years
    blockStarts, blockEnds, blockLabels = trend_helper.makeBins(window='week',nYears=2)

    admitDates = patients[['PatientID','admit_date']]

    # extract the time-based features for everyone in these bins
    timeTrends_original = trend_helper.parseScores(ScoreData=scores, AdmitDates=admitDates, id_col='PatientID', blockStarts=blockStarts, blockEnds=blockEnds, blockLabels=blockLabels, score_date_col='Date', model_answer_col='baseline_answer', prob_col='baseline_probability', admit_date_col='admit_date')

    timeTrends_adjusted = trend_helper.parseScores(ScoreData=scores, AdmitDates=admitDates, id_col='PatientID', blockStarts=blockStarts, blockEnds=blockEnds, blockLabels=blockLabels, score_date_col='Date', model_answer_col='adjusted_answer', prob_col='adjusted_probability', admit_date_col='admit_date')

    # do this separately for each patient and save
    for p in range(0,len(patients)):

        plt.figure(figsize= (7,5))
        figData_baseline = timeTrends_original[timeTrends_original.PatientID==patients.PatientID[p]]
        figData_adjusted = timeTrends_adjusted[timeTrends_adjusted.PatientID==patients.PatientID[p]]
        # make overlapping line plots
        g = sns.lineplot(x='Time Interval',y='Highest Probability',data=figData_baseline, ci=68,color='grey',sort=False,linewidth=1.5,alpha=0.9,linestyle=':')
        g = sns.lineplot(x='Time Interval',y='Highest Probability',data=figData_adjusted, ci=68,color='#77C6BA',sort=False,alpha=1,linewidth=2)
        plt.ylim(0,1.0)
        plt.ylabel('Epilepsy Probability',fontsize=13)
        plt.xticks(np.arange(0,105,26), ['0','6 months','1 year','1.5 years', '2 years'])
        plt.xticks(np.arange(0,105,13/3))
        plt.xlabel('Time Post-Injury',fontsize=13)
        sns.despine()
        figName = str(patients.PatientID[p]) + '_epilepsy_probability_lineplot.pdf'
        plt.savefig(opj(output_path,figName), bbox_inches='tight')
        plt.close()
        

    

    
