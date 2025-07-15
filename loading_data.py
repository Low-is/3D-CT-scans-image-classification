import pydicom
import numpy as np
import os

dicom_folder_path = r"path\to\ct_scan_series_folder"
dicom_files = [os.path.join(dicom_folder_path, f) for f in os.listdir(dicom_folder_path) if f.endswith('.dcm')]

dicom_fiiles = []
for f in os.listdir(dicom_folder_path):
  lung_subtype_folder = 
  if f.endswith('.dcm'):
    dicom_files.append(os.path.join(dicom_folder_path, f))
