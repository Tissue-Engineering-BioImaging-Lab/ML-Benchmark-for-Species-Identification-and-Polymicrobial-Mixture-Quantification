import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# dictionary for value insertion for target column
targets = {
    "1-9": "1:9 | Staph : PAO",
    "3-7": "3:7 | Staph : PAO",
    "5-5": "5:5 | Staph : PAO",
    "7-3": "7:3 | Staph : PAO",
    "9-1": "9:1 | Staph : PAO"
}

def read_all_csvs_to_df(folder_path):
    dataframes = []
    reference_axis = None

    file_list = sorted(f for f in os.listdir(folder_path) if f.endswith(".csv"))

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        raw_df = pd.read_csv(file_path, header=None)

        wavenumbers = pd.to_numeric(raw_df.iloc[0], errors="coerce").to_numpy(dtype=float)
        spectra = raw_df.iloc[1:].apply(pd.to_numeric, errors="coerce")

        valid_wavenumbers = ~np.isnan(wavenumbers)
        wavenumbers = wavenumbers[valid_wavenumbers]
        spectra = spectra.loc[:, valid_wavenumbers].to_numpy(dtype=float)

        sort_idx = np.argsort(wavenumbers)
        wavenumbers = wavenumbers[sort_idx]
        spectra = spectra[:, sort_idx]

        if reference_axis is None:
            reference_axis = wavenumbers
        elif len(wavenumbers) != len(reference_axis) or not np.allclose(wavenumbers, reference_axis):
            spectra = np.vstack([
                np.interp(reference_axis, wavenumbers, row)
                for row in spectra
            ])

        df = pd.DataFrame(spectra, columns=reference_axis)

        # adding the target column values based on the target dictionary above
        target_value = next((value for key, value in targets.items() if key in file_name), None)  

        if target_value is None:
            raise ValueError(f"No target mapping found for file: {file_name}")

        df["target"] = target_value
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # label encoding for RF model 
    label_encoder = LabelEncoder()
    combined_df["target_encoded"] = label_encoder.fit_transform(combined_df["target"])

    # exporting target value map
    target_map = {
        str(original): int(encoded)
        for original, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    }

    json_path = os.path.join(os.getcwd(), "target_encoding_map.json")
    with open(json_path, "w") as f:
        json.dump(target_map, f, indent=4)


    return combined_df


# accessing the main data folder
folder_path = os.path.join("..", "data", "mixed_ratio")
df = read_all_csvs_to_df(folder_path)

# for testing
#print(df.head())

print(df.groupby("target").size().to_string())

