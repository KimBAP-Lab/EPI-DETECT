
Written by Justin Wheelock (justin.wheelock@yale.edu)
23 October 2025

Codes associated with manuscript Wheelock J. et al (2025). In preparation. From Clinical Narrative to Diagnosis: Scalable Identification of Acquired Epilepsy

Contained is the script for running the nlp-based machine learning algorithm for identifying patients with acquired epilepsy following an acute brain injury/insult 

This script integrates the epilepsy phenotyping nlp algorithm by M. Fernandes et al (2023 Epilepsia, DOI: 10.1111/epi.17589) in a single step, with two additional ML algorithms 

###################################################

Requirements: 

Before you begin, place your input data in /input, formatted accordingly. Column names are case sensitive
    patients.csv only needs an id for each patient you want to get the outcome for ('PatientID') and the date of their admission for an acute brain injury ('admit_date')
    notes.csv should contain every note for the patients you want to run. Each row should contain the patient id ('PatientID'), the date (and time) of that note ('Date'), a unique identifier for that note which can be random ('NoteID') and the text of the note itself ('NoteTXT')

###################################################

Running: 

1) Clone the directory to your local machine

2) Activate the environment epilepsy_nlp found in /envs from an anaconda prompt (or terimal on mac). make sure you have navigated this terminal to the directory where you cloned these codes
    - If inputs were formatted properly and placed in the correct directories, you should not need to modify any lines of code

3) python main.py

- outputs will populate in /output
- patient epilepsy trend figures will populate in /output/figs

