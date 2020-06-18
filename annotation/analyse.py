import git
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




def main():
    root = git_root()
    path = os.path.join(root, 'annotation/annotated/')

    data = {}

    for fname in os.listdir(path):
        fname = os.path.join(path, fname)
        data[fname2word(fname)] = pd.read_csv(fname)

    run(data)




def run(data):


    if len(data) <= 3:
        fig, axs = plt.subplots(ncols = len(data))
    elif len(data) <= 6:
        fig, axs = plt.subplots(nrows = int((len(data)+1)/2), ncols = 2)
    else:
        fig, axs = plt.subplots(nrows = int((len(data)+1)/3), ncols = 3)


    print(type(axs))

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, (word, df) in enumerate(data.items()):
        cos = cosine(df)

        sns.heatmap(cos, ax = axs[i], square = True)
        axs[i].set_title(word)
        axs[i].axis('equal')


    plt.tight_layout()
    plt.show()







def cosine(df):
    crosstab = pd.crosstab(df.label, df.wordnet_sense)
    cos = 1 - cosine_similarity(crosstab)
    print(cos)
    return cos











def git_root(path = os.getcwd()):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def fname2word(fname):
    return fname.split('/')[-1].replace('.csv', '')


if __name__ == '__main__':
    main()