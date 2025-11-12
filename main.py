"""
Created by Justin Wheelock (justin.wheelock@yale.edu)
22 October 2025 10:39
"""

# main function for running the acquired epilepsy detection algorithm

"""
Required inputs (put inside /inputs): 
    patients.csv:
        containing columns "PatientID" and "admit_date" 
    notes.csv:
        containing columns "PatientID", "Date","NoteID", "NoteTXT"
"""

import os 
from os.path import join as opj 
import pandas as pd 
from utils import runBaseline_helper
from utils import runPatientLevel_helper
from utils import runNoteLevel_helper
from utils import plotEpilepsyTrends

path = os.getcwd()

patients = pd.read_csv(opj(path,'input','patients.csv'))
notes = pd.read_csv(opj(path,'input','notes.csv'))

# decide whether you want to create plots for each patient
plot_patients = 1

# remove later, but for testing purposes limit to 5 patients
patients = patients[10:15].reset_index(drop=True)
notes = notes[notes.PatientID.isin(patients.PatientID.to_list())]

#------------------------------------------------------------------------
# Run baseline phenotyping algorithm
#------------------------------------------------------------------------
print('step one: running baseline phenotyping algorithm...')

df_notes = runBaseline_helper.build_cohort_deidentified(patients=patients,notes=notes,path=path)

scores = runBaseline_helper.assign_scores(df_notes=df_notes,path_train=opj(path,'utils'))

print('done! on to step 2...')
print('')

#------------------------------------------------------------------------
# method 1: patient-level identification
#------------------------------------------------------------------------
print('step two: running patient-level epilepsy identification...')

patientLevel_output = runPatientLevel_helper.score(patients=patients,scores=scores,temp_path=opj(path,'utils'),model_directory=opj(path,'models'))

patientLevel_output.to_csv(opj(path,'output','patient_level_predictions.csv'))
print('')

#------------------------------------------------------------------------
# method 2: note-level identification
#------------------------------------------------------------------------
print('step three: running note-level epilepsy identification...')

noteLevel_output = runNoteLevel_helper.score(patients=patients,scores=scores,temp_path=opj(path,'utils'),model_directory=opj(path,'models'))

noteLevel_output.to_csv(opj(path,'output','note_level_predictions.csv'))

print('done! plot individual patients?')

#------------------------------------------------------------------------
# optional: plot individual patient trajectories
#------------------------------------------------------------------------

if plot_patients:
    plotEpilepsyTrends.plot(patients=patients,scores=noteLevel_output,output_path=opj(path,'output','figs'))
    print('figures saved to output/figs')
