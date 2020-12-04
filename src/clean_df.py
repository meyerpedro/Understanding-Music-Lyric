import matplotlib.pyplot as plt
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

d = pd.DataFrame({'CoL 1': [1, 2], 'cO_l2': [3, 4]})

def clean_df(df):
    # columns to lower case and no spaces

    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    return df

if __name__ == "__main__": 

    print(clean_df(d))