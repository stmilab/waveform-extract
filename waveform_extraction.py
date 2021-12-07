import os
import pickle
import argparse
import csv
import wfdb

import pandas as pd
import numpy as np
import neurokit2 as nk
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser("Waveform Extraction")

    parser.add_argument('--patient_list', type=str, default='example_patient_list.csv',
                  help='Path to patient list')
    parser.add_argument('--extract_beats', action='store_true', default=False,
                  help='Extract beats')
    parser.add_argument('--destination_folder', type=str, default='data',
                  help='Folder to save extracted data')

    return parser.parse_args()

def create_zero_padded_beats(waveform):
    max_len = 0
    padded_x = []
    beat_2_beat_wf = []
    sbp_data = []
    dbp_data = []
    max_len_save = []
    _, rpeaks = nk.ecg_peaks(waveform['ECG'].values, sampling_rate=125)
    rpeaks = rpeaks['ECG_R_Peaks']

    for i in range(len(rpeaks)-1):
        if rpeaks[i+1] - rpeaks[i] > 250:
            continue

        if waveform[rpeaks[i]:rpeaks[i+1]][['PPG','ECG','ABP']].isna().sum().sum() > 0:
            continue

        if rpeaks[i+1] - rpeaks[i] > max_len:
            max_len = rpeaks[i+1] - rpeaks[i]

        sbp_data.append(np.amax(waveform[rpeaks[i]:rpeaks[i+1]]['ABP'].values))
        dbp_data.append(np.amin(waveform[rpeaks[i]:rpeaks[i+1]]['ABP'].values))
        beat_2_beat_wf.append(np.array(waveform[rpeaks[i]:rpeaks[i+1]][['PPG','ECG']].values))

    for beat in beat_2_beat_wf:
        tmp_zeros = np.zeros((max_len-len(beat),2))
        pad_x = np.concatenate((tmp_zeros, beat),axis=0)
        padded_x.append(pad_x)

    return sbp_data, dbp_data, rpeaks, max_len, padded_x

args = parse()

with open(args.patient_list, newline='\n') as f:
    reader = csv.reader(f)
    patient_list = [patient_id[0] for patient_id in reader]

print("downloading waveforms...")
waveforms = {name: pd.DataFrame() for name in patient_list}

for i, pid_complete in enumerate(tqdm(patient_list)):
    pn_dir = 'mimic3wdb/matched/' + patient_list[i][:3] + '/' + patient_list[i][0:7] + '/'
    signals, fields = wfdb.rdsamp(pid_complete, pn_dir = pn_dir)

    waveforms[pid_complete]['ECG'] = signals[:,0]
    waveforms[pid_complete]['PPG'] = signals[:,4]
    waveforms[pid_complete]['ABP'] = signals[:,5]

cwd = os.getcwd()
len_save = []

for i, pid_complete in enumerate(patient_list):
    bp = pd.DataFrame()
    r_peaks = pd.DataFrame()

    if i == 14: #not enough data 2.5hours
        continue

    print(pid_complete)
    print("==========================")

    if args.extract_beats:
        print("Extracting Beats...")
        bp['SBP'], bp['DBP'], r_peaks['index'], max_len, zero_padded = create_zero_padded_beats(waveforms[pid_complete])

    save_path = os.path.join(cwd, args.destination_folder, pid_complete)
    print("saving to {}".format(save_path))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    waveform_path = os.path.join(save_path, 'waveforms.csv')
    waveforms[pid_complete].to_csv(waveform_path, index=False)

    if args.extract_beats:
        bp_path = os.path.join(save_path, 'bp.csv')
        rpeaks_path = os.path.join(save_path, 'rpeaks.csv')
        bp.to_csv(bp_path, index=False)
        r_peaks.to_csv(rpeaks_path, index = False)
    
        ppg_ecg_zero_padded = os.path.join(save_path, "ppg_ecg_zero_padded.p")
        with open(ppg_ecg_zero_padded, "wb") as output_file:
            pickle.dump(zero_padded,output_file)
            
        len_save.append(max_len)

if args.extract_beats:
    signal_len = pd.DataFrame()
    signal_len['PID'] = patient_list
    signal_len['len'] = len_save
    signal_len.to_csv(os.path.join(save_path, 'signals_length.csv'))
