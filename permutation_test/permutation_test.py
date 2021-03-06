import argparse
import pandas as pd
from transformers import BertTokenizer, BertModel
import os
import torch
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
from collections import Counter
from nltk.corpus import stopwords
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import warnings 
import ast


parser = argparse.ArgumentParser(
    description = 'Permutation test of senses',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('--data_dir', type=str, default="../train",
    help="Directory of data files. All csv-files will be used in this directory if 'word_file' is not passed")
parser.add_argument('-f','--word_file', type=str, default=None, help='Csv file with texts.')
parser.add_argument('-m','--maxlen', type=int, default=128, help='Maximum number of tokens passed to BERT.')
parser.add_argument('-bs', '--batch_size', type=int, default=200, help='Batch size for BERT.')
parser.add_argument('-i', '--iterations', type=int, default=3000, help='Number of iterations for permutation test.')
parser.add_argument('-d', '--display_figs', type=str, default=None, help='Whether to display figures or not.')
parser.add_argument('-s', '--signlevel', type=float, default=0.05, help='Significance level of the permutation test')
parser.add_argument('-k', '--k', type=int, default=3, help='k-nearest neighbors in dahlberg merge')


args = parser.parse_args()

DATA_DIR = args.data_dir
WORD_FILE = args.word_file
MAXLEN = args.maxlen
BATCH_SIZE = args.batch_size
ITERATIONS = args.iterations
DISPLAY_FIGS = args.display_figs
SIGN_LEVEL = args.signlevel
K = args.k


def main():

    warnings.filterwarnings("ignore")
    print('Loading Bert model')

    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if DATA_DIR != "":
        SAVE_DIR = DATA_DIR
    else:
        SAVE_DIR = "."

    if WORD_FILE:
        files = [WORD_FILE]
    else:
        files = [file_ for file_ in os.listdir(DATA_DIR) if file_.endswith('.csv') and not file_.endswith('_not_annotated.csv') ]

    for fname in files:
        try:
            path = os.path.join(DATA_DIR, fname)
            df, data = get_data(path, tokenizer)
            if len(df) >= 750:
                continue
            if "/" in fname:
                fname = fname[fname.rfind("/")+1:]

            save_dir = os.path.join(DATA_DIR, fname.replace('.csv', ''))
            if not os.path.exists(save_dir): # Create directory if it doesn't exist
                os.mkdir(save_dir)


            #PLOT BEFORE
            sense_dict = df["sense"].value_counts()
            df_plot = pd.DataFrame({'sense':sense_dict.index, 'count':sense_dict.values})

            fig = plt.figure(figsize = (6,7))
            sns.barplot(x = "count", y = "sense", data = df_plot)
            save_file = os.path.join(save_dir, 'sense_hist1.png')
            plt.savefig(save_file , bbox_inches='tight')

            X = bert_embed(data, bert_model)
            y = df['label'].values
            df = test(X, y, df, word = fname.replace('.csv', ''))


            #PLOT AFTER
            sense_dict = df["sense"].value_counts()
            df_plot = pd.DataFrame({'sense':sense_dict.index, 'count':sense_dict.values})

            fig = plt.figure(figsize = (6,7))
            sns.barplot(x = "count", y = "sense", data = df_plot)
            save_file = os.path.join(save_dir, 'sense_hist2.png')
            plt.savefig(save_file , bbox_inches='tight')
            df.drop(['embedding', 'ind'], axis=1, inplace=True) 
            df = df.rename(columns={'old_ind': 'ind'})
            df.to_csv(os.path.join(save_dir, fname), index=False)

        except Exception as e:
            print(fname, str(e))


    if DISPLAY_FIGS:
        plt.show()



def get_data(fname, tokenizer):

    print("***********************************")
    print('Reading file:', fname)
    df = pd.read_csv(fname)

    
    for key, val in df["label"].value_counts().items():
        if val <= 10: #If a sense has less than 10 data points then remove it!
            df = df.loc[df["label"] != key]
    df.reset_index(drop=True, inplace=True)
    
     
    df["embedding"] = 0
    old_ind = df['ind']
    df = df.apply(lambda row: tokenize_input(row, tokenizer), axis = 1)
    df['old_ind'] = old_ind
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    le = preprocessing.LabelEncoder()
    le.fit(df["label"].values)
    df["label"] = le.transform(df["label"].values)
            
    padded_tokenized_input = np.zeros((len(df), MAXLEN),dtype=np.int16)

    for ind, voc_vals in enumerate(df["embedding"]):
        voc_vals = voc_vals[:MAXLEN]
        padded_tokenized_input[ind,:] = np.array(voc_vals + [0]*(MAXLEN-len(voc_vals)))
    attention_masks = np.where(padded_tokenized_input != 0, 1, 0)


    return df, {
        'input_ids': torch.Tensor(padded_tokenized_input).to(torch.int64),
        'attention_masks': torch.Tensor(attention_masks),
        'labels': torch.Tensor(df['label'].values),
        'indices': torch.Tensor(df['ind'].values)
    }


def test(X, y, df, word):
    print('Performing permutation test')

    df["new_label"] = df["label"]

    save_dir = os.path.join(DATA_DIR, word)

    tsne_model = TSNE(perplexity=30, n_components=2, init='random', n_iter=750, metric='euclidean')
    new_values = tsne_model.fit_transform(X)

    colors = ["red", "blue", "green", "yellow", "purple", "black", "orange"]

    plt.figure(figsize=(18, 12)) 
    for i in range(len(new_values)):
        plt.scatter(new_values[i][0], new_values[i][1], color = colors[int(y[i])])

    plt.title("tSNE for {}".format(word))
    fname = os.path.join(save_dir, 'tSNE.png') 
    plt.savefig(fname)

    file = open(os.path.join(save_dir, 'merging_results.txt'), 'w')

    while True:
        classes = set(df['new_label'].values)
        break_outer=False
        for classA in classes:
            for classB in classes: #Do permutation test between all pair of senses.
                if classA == classB:
                    continue
                p_value, _, _ = permutation_test(classA, classB, X, y)
                if p_value >= SIGN_LEVEL:
                    df = merge(classA, classB, p_value, df, file)
                    break_outer=True
                    break

            if break_outer:
                break
        if not break_outer:
            break

    file.close()

    return df


def tokenize_input(row, tokenizer):
        
    text = row["text"].split()
    label_ind = row["ind"]
    
    tokens = [101] #[CLS] token
    for ind, word in enumerate(text): #need to tokenize one word at a time so can keep track of the index of the ambiguous word!
        token = tokenizer.encode(word, add_special_tokens=False)
        if ind == label_ind:
            row["ind"] = len(tokens) if len(tokens) < MAXLEN else None
        tokens += token
    tokens += [102] #[SEP] token
    row["embedding"] = tokens
    return row

def bert_embed(data, bert_model, BATCH_SIZE = 16, MAX_LEN = 128):
    """
    data should contain fields 'input_ids', 'attention_masks' and 'indices'
    """
    
    dataset = TensorDataset(
        data['input_ids'], data['attention_masks'], data['indices']
    )
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on ' + device.type)
    if device.type == 'cuda':
        bert_model.cuda() # put bert in training mode
        
    N = data['indices'].shape[0]
    X = np.zeros((N, 768))
    pos = 0
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_masks, b_indices = batch
        
        with torch.no_grad():
            embeddings = bert_model(
                b_input_ids.view(-1, MAX_LEN),
                b_input_masks.view(-1, MAX_LEN)
            )[2]
        # Take the mean of the last 4 hidden states
        embeddings = (embeddings[-4] + embeddings[-3] + embeddings[-2] + embeddings[-1])/4
        for j, label_ind in enumerate(b_indices.cpu().detach().numpy()):
            X[pos,:] = embeddings[j, int(label_ind), :].cpu().detach().numpy()
            pos+=1
    return X


def calculate_distances(X):

    X_dist = np.zeros((len(X), len(X)))

    for i in range(len(X)):   
        for j in range(i, len(X), 1):
            distance = np.linalg.norm(X[i,:] - X[j,:])
            X_dist[i,j], X_dist[j,i] =  distance, distance

    return X_dist


#This is the test statistic.
def calculate_distance_permtest(inds_A, inds_B, XX_dist): 
    distance_intA, distance_intB, distance_ext = 0,0,0
    
    distances = []
    for ind_1A in inds_A:
        for ind_2A in inds_A:  
            if ind_1A == ind_2A:
                continue
            distances += [XX_dist[ind_1A, ind_2A]] 
    distances = np.array(distances)     
    distance_intA = np.mean(distances)
    
    distances = []
    for ind_1B in inds_B:
        for ind_2B in inds_B:  
            if ind_1B == ind_2B:
                continue
            distances += [XX_dist[ind_1B, ind_2B]]   
    distances = np.array(distances)     
    distance_intB = np.mean(distances) 

    distances = []
    for ind_A in inds_A:
        for ind_B in inds_B:
            distances += [XX_dist[ind_A, ind_B]]
    distances = np.array(distances)     
    distances = distances[distances.argsort()[:int(0.5*len(distances))]]  
    distance_ext = np.mean(distances)  
    
    return 2*distance_ext/(distance_intA + distance_intB)

def permutation_test(classA, classB, X, y):
    
    yy = y[(y == classA) | (y == classB)]
    XX = X[(y == classA) | (y == classB)]
    XX_dist = np.zeros((len(XX), len(XX)))
    size_classA = sum(yy == classA)

    for i in range(len(XX)):   
        for j in range(i, len(XX), 1):
            distance = np.linalg.norm(XX[i,:] - XX[j,:])
            XX_dist[i,j], XX_dist[j,i] =  distance, distance

    
    def parallel_function():
        inds_A = set(np.random.choice(len(XX), size_classA, replace=False))
        inds_B = set(np.arange(len(XX))).difference(inds_A)  
        return calculate_distance_permtest(inds_A, inds_B, XX_dist)

    distance = calculate_distance_permtest( (yy == classA).nonzero()[0], (yy == classB).nonzero()[0], XX_dist)

    #the test is splitted up for speed by first doing 1/3 of the iterations and if haven't observed any more extreme observation then reject.
    #if have observed any more extreme observation then continue doing the other 2/3 iterations.
    distances1 = Parallel(n_jobs=mp.cpu_count(), verbose=3)(delayed(parallel_function)() for _ in range(int(ITERATIONS/3)))
    if sum(distances1 >= distance) == 0:
        return (float(3/ITERATIONS), distances1, distance)
    distances2 = Parallel(n_jobs=mp.cpu_count(), verbose=3)(delayed(parallel_function)() for _ in range(int(2*ITERATIONS/3)))

    p_value = (sum( distances1 >= distance) + sum( distances2 >= distance) + 1)/(ITERATIONS+1)
    return (p_value, np.concatenate((distances1, distances2)), distance)


def merge(classA, classB, p_value, df, file):

    if len(df.loc[df["new_label"] == classB]) < len(df.loc[df["new_label"] == classA]):
    
        file.write('merging ' + df.loc[df["new_label"] == classB, "sense"].iloc[0] + ' to ' +  df.loc[df["new_label"] == classA, "sense"].iloc[0] + ' p-value: ' + str(p_value) + '\n')
        df.loc[df["new_label"] == classB, "sense"] = df.loc[df["new_label"] == classA, "sense"].iloc[0]
        df.loc[df["new_label"] == classB, "new_label"] = classA

    else:
    
        file.write('merging ' + df.loc[df["new_label"] == classA, "sense"].iloc[0] + ' to ' +  df.loc[df["new_label"] == classB, "sense"].iloc[0] + ' p-value: ' + str(p_value) + '\n')
        df.loc[df["new_label"] == classA, "sense"] = df.loc[df["new_label"] == classB, "sense"].iloc[0]
        df.loc[df["new_label"] == classA, "new_label"] = classB 

    return df


if __name__ == '__main__':
    main()








