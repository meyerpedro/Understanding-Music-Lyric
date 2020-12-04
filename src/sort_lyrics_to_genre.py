import matplotlib.pyplot as plt
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ly = pd.read_csv('../data/1950_2019_with lyrics.csv')

def sort_by_genre(df, genre, col='genre'):
    # subset original df per genre 
    new_df = df.loc[df[col] == genre]

    return new_df

def lyrics_to_genre(df, col):
    # creates new df with genre and all lyrics within that genre
    l = []

    for lyrics in sort_by_genre(df, col)['lyrics']:
        l.append(lyrics)

    out = ' '.join(l)
    
    # breakpoint()

    new_df = pd.DataFrame({'genre':col, 'lyrics': [out]})
    return new_df


if __name__ == "__main__": 

    # print(sort_by_genre(ly, 'rock'), '\n')
    print(lyrics_to_genre(sort_by_genre(ly, 'rock'), 'rock'))