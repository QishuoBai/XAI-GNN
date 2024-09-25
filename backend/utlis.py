import pandas as pd
import numpy as np

def get_edge_feature_values(data, id, feature_keys):
    row = data[data['ID'] == id]
    values = row[feature_keys].values[0].tolist()
    return values