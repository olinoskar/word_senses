import pandas as pd
import numpy as np
import os
import subprocess 
from collections import Counter

PATH = 'train/'


def main(): 

    files = get_files(path = PATH)
    dfs = get_dataframes(files)
    print_interesting_words(dfs)

def get_files(path):
    cmd = 'ls {}*.csv | grep -v _not_annotated'.format(path)
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    output = pipe.read().decode()
    return output.split()

def get_dataframes(files):
    dfs = {}
    for fname in files:
        word = fname.replace(PATH, '').replace('.csv','')
        df = pd.read_csv(fname)
        dfs[word] = df
    return dfs

def print_interesting_words(dfs):

    df_res = pd.DataFrame(columns = [
        'Nr. Senses', 'Mean', 'Min', 'Max', 'Total'
        ])

    for word, df in dfs.items():

        senses = df['sense'].values
        counter = Counter(senses)

        df_res.loc[word] = [
            len(counter), np.mean(list(counter.values())),
            min(counter.values()), max(counter.values()), sum(counter.values())
        ]
    df_res = df_res.applymap(lambda num: round(num))

    df_res = df_res[df_res.Total >= 20]

    print(df_res)






if __name__ == '__main__':
    main()