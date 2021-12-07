# Waveform Extraction
Code for extracting waveform data from our server. Additionally, a basic PyTorch dataset class can be found in `pytorch_dataset.py`.

# How to use

## Data Extraction
This code can be run with the following command

`python waveform_extraction.py --patient_list <your list of patients>`

An example of the a patient list can be found in the repo named example_patient_list.csv.

## PyTorch Dataset
First extract the data to CSVs using `waveform_extraction.py`

The dataset class uses a sliding window to grab frames from the waveform data. 

The class has the following required arguments:
- root (str): path to data
- window_size (int): size of sliding window
- stride (int): stride size of sliding window

Additionally, this class allows you to pass in your own processing or PyTorch transform witht the following arguments:
- preprocess (function): preprocessing function
- transform (torch.transform): transform function

Lastly, data is split 80% training and 20% validation. The splits can be accessed using the following argument:
- train (bool): 
  - if true, returns training split
  - if false, return test split

# TODO
- [ ] Extract beats
- [ ] Preprocess
