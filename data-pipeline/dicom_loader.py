from collections import defaultdict
import os


def group_dicom_files(dicom_folder_path):
  """
  Group DICOM files by subtype and patient.

  Output format:
  dicom_files_grouped[subtype][patient] = [list of .dcm file paths]
  """
  
  dicom_files_grouped = defaultdict(lambda: defaultdict(list)) # currently an empty defaultdict

  for root, dirs, files in os.walk(dicom_folder_path):
      for f in files:
          if f.endswith('.dcm'):
              full_path = os.path.join(root, f)
            
              # patient folder is the parent directory
              patient = os.path.basename(root)
            
              # extract subtype: the letter after the dash in folder name
              # e.g., "Lung_Dx-A0001" -> "A"
              try:
                  subtype = patient.split('-')[1][0]
              except IndexError:
                  print(f"Folder name not matching expected pattern: {patient}")
                  continue
            
              dicom_files_grouped[subtype][patient].append(full_path)
            
return dicom_files_grouped
