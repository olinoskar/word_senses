import argparse
import hyperopt
from hyperopt import hp, fmin, tpe, space_eval
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adamax, RMSprop
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import warnings 
import pandas as pd
from transformers import BertTokenizer, BertModel
import os
import torch
from tqdm import tqdm
import numpy as np
import json
from training_functions import *

parser = argparse.ArgumentParser(
    description = 'Hypertuning and training of classifier(s)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('--data_dir', type=str, default="", help="Directory containing directories of the ambigious words.")
parser.add_argument('-f','--word_file', type=str, default=None, help='Directory of the word to train classifier on')
parser.add_argument('-m','--maxlen', type=int, default=128, help='Maximum number of tokens passed to BERT.')
parser.add_argument('-bs', '--batch_size', type=int, default=200, help='Batch size for BERT.')
parser.add_argument('--iter', type=int, default=300, help='Number of iterations in HyperOpt')
parser.add_argument('--save_path', type=str, default="results.json", help = "Path of the json dump to be saved")


args = parser.parse_args()
DATA_DIR = args.data_dir
MAXLEN = args.maxlen
BATCH_SIZE = args.batch_size
ITERATIONS = args.iter
SAVE_PATH = args.save_path

def main():

	warnings.filterwarnings("ignore")
	print('Loading Bert model')
	bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True) 
	bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 

	save_dict = {}
	for folder in os.listdir(DATA_DIR):
		d = train_word(folder, bert_model, bert_tokenizer)
		save_dict[folder] = d
		with open(folder + "_results.json", 'w') as f:
			json.dump(d, f)

	with open(SAVE_PATH, 'w') as f:
	    json.dump(save_dict, f)
	
def train_word(folder, bert_model, bert_tokenizer):

	try:
		data_file = os.path.join(DATA_DIR, folder, folder+'.csv')

		X, df = get_data(data_file, bert_model, bert_tokenizer)

		le = preprocessing.LabelEncoder() #Relabel so labels are 0,1,2,3..... Could be something else after merging of senses
		le.fit(df["label"].values)
		df["label"] = le.transform(df["label"].values)

		#This is the hyperparameter space with use in hyperopt
		space = {'classifier': hp.choice('classifier',[
		        {'model': "kNN",
		        'param': {
		            'n_neighbors':hp.choice('n_neighbors',range(1,10)),
		            'algorithm':hp.choice('algorithm',['ball_tree','kd_tree']),
		            'leaf_size':hp.choice('leaf_size',range(1,40)),
		            'metric':hp.choice('metric', ["euclidean","manhattan","chebyshev"])} 
		        },
		        {'model': "SVC",
		        'param':{
		            'C':hp.lognormal('C',0,1),
		            'kernel':hp.choice('kernel',['rbf','poly','sigmoid']), 
		            'degree':hp.choice('degree',range(1,8)),
		            'gamma':hp.uniform('gamma',0.001,5000)}
		        },
		        {'model': 'Neural net',
		        'param':{ 
		                       
		            'layer2':hp.choice('layer2', ['yes', 'no']),
		            'layer3':hp.choice('layer4', ['yes', 'no']),
		            'layer4':hp.choice('layer5', ['yes', 'no']), 
		            'neurons_input':hp.choice('neurons_input', [768]),
		            'neurons_layer2':hp.choice('neurons_layer2', range(128, 512)),
		            'neurons_layer3':hp.choice('neurons_layer4', range(64, 128)),
		            'neurons_layer4':hp.choice('neurons_layer5', range(16, 62)),
		            'neurons_output':hp.choice('neurons_output', [len(np.unique(df["label"].values))]), 
		            'activation2':hp.choice('activation2', ["relu", "tanh", "sigmoid", "leaky relu"]),
		            'activation3':hp.choice('activation4', ["relu", "tanh", "sigmoid", "leaky relu"]),
		            'activation4':hp.choice('activation5', ["relu", "tanh", "sigmoid", "leaky relu"]),
		            'optimizer': hp.choice('optimizer', ['Adam', 'Adamax', 'RMSprop']),
		            'learning_rate': hp.uniform('learning_rate', 1e-5, 1e-1),
		            'dropout_prob': hp.uniform('dropout_prob', 0.0, 0.5),
		            'epochs':hp.choice('epochs', range(3,15,1)),
		            'batch_size':hp.choice('batch_size', [8,16,32,64]),
		            'layer2_dropout':hp.choice('layer2_dropout', ['yes', 'no']),
		            'layer3_dropout':hp.choice('layer4_dropout', ['yes', 'no']),
		            'layer4_dropout':hp.choice('layer5_dropout', ['yes', 'no'])}  
		        },
		        {'model': 'RFC',
		        'param':{
		            "n_estimators":hp.choice('n_estimators', range(10,175)),
		            "criterion":hp.choice ('criterion', ['gini', 'entropy']),
		            "max_depth":hp.choice('max_depth', range(1,17))}
		        },
		        {'model' : 'GaussianNB',
		         'param':{        
		        }
		        }])}
		

		y = df["label"].values.astype(int)
		y = torch.Tensor(y).long()

		X = torch.Tensor(X)

		best = fmin(lambda args: hyperopt(args, X.clone(), y.clone()), space, algo=tpe.suggest, max_evals = ITERATIONS)
		best = space_eval(space, best)
		accuracy = -hyperopt(best, X, y)

		print(folder, accuracy, best['classifier'])

		d = {}
		d['accuracy'] = accuracy
		d["classifier"] = best['classifier']
		d['senses'] = []
		d['total'] = len(y)
		for sense in np.unique(df['sense'].values):
			ratio = len(df.loc[df['sense'] == sense])/len(df)
			d['senses'] += [{'name':sense, 'ratio':ratio}]
			d['number_of_senses'] = len(set(df['label'].values))

	except Exception as e:
		print(folder, str(e))
		return {}

	return d


def get_data(fname, bert_model, bert_tokenizer):

	print("***********************************")
	print('Reading file:', fname)
	df = pd.read_csv(fname)

	getter = FeatureVectorGetterBERT(bert_model, bert_tokenizer)
	return getter.get_feature_vector(df)


#Function to maximize
def hyperopt(args, X, y):
	classifier = args["classifier"]['model']
	params = args['classifier']['param']

	skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
	res = []
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

            
		if classifier == "Neural net":
			params['neurons_input'] = 768

			net = BuildNetwork(params)  

			net = trainNN(X_train, y_train, net, params)

			accuracy = report_scoreNN(net, X_test, y_test, params)
			res += [accuracy]
			continue


		elif classifier == "kNN":
			n_neighbors = params['n_neighbors']
			algorithm = params['algorithm']
			leaf_size = params['leaf_size']
			metric = params['metric']
			clf = KNeighborsClassifier(n_neighbors=n_neighbors,
			                       algorithm=algorithm,
			                       leaf_size=leaf_size,
			                       metric=metric)               
		elif classifier == "SVC":
			C = params['C']
			kernel = params['kernel']
			degree = params['degree']
			gamma = params['gamma']
			clf = SVC(C=C, kernel=kernel, degree=degree,gamma=gamma, max_iter=1500)


		elif classifier == "RFC":
			n_estimators = params['n_estimators']
			criterion = params['criterion']
			max_depth = params['max_depth']
			clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
		else:
		    clf = GaussianNB()
          
		clf.fit(X_train,y_train)

		y_pred = clf.predict(X_test)
		accuracy = accuracy_score(y_test, y_pred)
		res += [accuracy]
    
	accuracy = -np.array(res).mean()
	return accuracy


class FeatureVectorGetterBERT:
	def __init__(self, bert_model, bert_tokenizer, MAXLEN = 128, BATCH_SIZE = 200):
		self.MAXLEN = MAXLEN
		self.BATCH_SIZE = BATCH_SIZE
		self.bert_model = bert_model
		self.bert_tokenizer = bert_tokenizer

	def tokenize_input(self, row):
        
		text = row["text"].split()
		label_ind = row["ind"]
        
		tokens = [101]
		for ind, word in enumerate(text):
			token = self.bert_tokenizer.encode(word, add_special_tokens=False)
			if ind == label_ind:
				row["ind"] = len(tokens) if len(tokens) < self.MAXLEN else None
			tokens += token
		tokens += [102]
		row["embedding"] = tokens
		return row

	def get_feature_vector(self, df): #A dataframe with columns 'text' and 'ind'.

		df["embedding"] = 0      
		df = df.apply(lambda row: self.tokenize_input(row), axis = 1)
		df.dropna(inplace=True)
		df.reset_index(drop=True, inplace=True)

		padded_tokenized_input = np.zeros((len(df), self.MAXLEN),dtype=np.int16)

		for ind, voc_vals in enumerate(df["embedding"]):
			voc_vals = voc_vals[:self.MAXLEN]
			padded_tokenized_input[ind,:] = np.array(voc_vals + [0]*(self.MAXLEN-len(voc_vals)))
		attention_masks = np.where(padded_tokenized_input != 0, 1, 0)

		input_ids = torch.Tensor(padded_tokenized_input).to(torch.int64) 
		attention_masks = torch.Tensor(attention_masks)
		indices =  df['ind'].values.astype(int)

		X = self.word_type_embed(input_ids, attention_masks, indices)
		return X, df #return BERT vectors and the new data frame!


	def word_type_embed(self, input_ids, attention_masks, indices):
		X = np.zeros((len(input_ids), 768))

		with torch.no_grad():
			for i in range(0, len(input_ids), self.BATCH_SIZE):
				end = min(len(input_ids), i+self.BATCH_SIZE)
				embeddings = self.bert_model(
				    input_ids[i:end].view(-1, self.MAXLEN),
				    attention_masks[i:end].view(-1, self.MAXLEN)
				)[2]

				for j in range(end-i):
					pos = i + j
					label_ind = indices[pos]
					X[pos,:] = 1/4*(embeddings[9][j, label_ind, :].detach().numpy()+embeddings[10][j, label_ind, :].detach().numpy()+
									embeddings[11][j, label_ind, :].detach().numpy()+embeddings[12][j, label_ind, :].detach().numpy())                        
		return X

if __name__ == '__main__':
	main()



	


