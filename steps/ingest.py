from steps.config import *

import pandas as pd

def load_dataset():
    _data = pd.read_csv(data_path)
    return _data