import re
from pprint import pprint
import pandas as pd
import os
from bs4 import BeautifulSoup
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
import lemmy
from collections import Counter
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import warnings 
import ast
import json

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(
    description = 'Word alignment of texts in en, fr, es. Must be in GIZA++ folder when executing this script',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('--path1', type=str, default="../../data/data.csv", help = 'Path to data.csv file')
parser.add_argument('--path2', type=str, default="../../data/lemmas.txt", help = 'Directory to save data')
parser.add_argument('--path5', type=str, default="../../train", help = 'Directory to save data')

args = parser.parse_args()

PATH1 = args.path1
PATH2 = args.path2
PATH5 = args.path5

with open(PATH2, 'r') as f:
    words = f.read().split('\n')


def main():

    warnings.filterwarnings("ignore")

    #Read data
    df = pd.read_csv(PATH1)
    df['text_en']= df['text_en'].astype(str)
    df['text_fr']= df['text_fr'].astype(str)
    df['text_es']= df['text_es'].astype(str)

    spacy_es = spacy.load("es_core_news_md")
    spacy_en = spacy.load('en_core_web_md')
    spacy_fr = spacy.load("fr_core_news_md")

    #No swedish model exists in spacy. Use english tokenizer for swedish
    tokenizers = [spacy_en.tokenizer, spacy_fr.tokenizer, spacy_es.tokenizer]
    lang_to_tokenizers = {"en" : tokenizers[0], "fr" : tokenizers[1], "es":tokenizers[2]}

    #Do the word alignment
    
    clean()
    prepare_texts("fr", "en", df, lang_to_tokenizers); align_texts("fr", "en"); align_texts("en", "fr")
    prepare_texts("es", "en", df, lang_to_tokenizers); align_texts("es", "en"); align_texts("en", "es")
    
    
    # Find senses

    stop_words = set(stopwords.words('english'))

    texts = df["text_en"].apply(lambda text: clean_text(text, tokenization=False)).values
    words = Counter([token.lemma_ for text in texts for token in spacy_en(text) \
                    if token.text not in stop_words and len(token.text) >= 3 and not token.text[0].isdigit()])

    words = set([key for key in words.keys() if words[key] >= 80])
    print("Number of words:" + str(len(words)))
    words_to_inds_map = words_to_inds(df, spacy_en)

    if not os.path.exists(PATH5): # Create directory if it doesn't exist
        os.mkdir(PATH5)

    Parallel(n_jobs=mp.cpu_count(), verbose=10)(delayed(parallel_function)(word, df, words_to_inds_map) for word in words)


def clean_text(text, tokenizer=None, tokenization=True):
    text = BeautifulSoup(text, features="lxml").get_text(separator = " ").lower()
    text = re.sub("\n", "", text)
   
    if tokenization: #Want to avoid double tokenization. Double tokenization is not same as single tokenization....
        tokens = tokenizer(text)
        tokens = ' '.join([token.text for token in tokens if not token.text.isspace()])
        return tokens
    else:
        text = ' '.join(text.lower().split())
        return text


#Write the texts we want to align to two files, e.g "en" and .
def prepare_texts(lang1, lang2, df, lang_to_tokenizers):
    
    file1 = open("{}".format(lang1), "w")
    file2 = open("{}".format(lang2), "w")
    
    for ind, row in df.iterrows():
        text1 = row["text_{}".format(lang1)]
        text1 = clean_text(text1, lang_to_tokenizers[lang1])
        file1.write(text1 + "\n")
        
        text2 = row["text_{}".format(lang2)]
        text2 = clean_text(text2, lang_to_tokenizers[lang2])
        file2.write(text2 + "\n")

    file1.close(); file2.close()
    
    os.system("./../mkcls-v2/mkcls -p{} -V{}.vcb.classes".format(lang1, lang1))
    os.system("./../mkcls-v2/mkcls -p{} -V{}.vcb.classes".format(lang2, lang2))
    
      
def align_texts(source, target):
    os.system("./plain2snt.out {} {}".format(source, target))

    if not os.path.exists("myout_{}_{}".format(source, target)):
        os.system("mkdir myout_{}_{}".format(source, target))
        
    cmd =  './GIZA++ -S {}.vcb -T {}.vcb -C {}_{}.snt -outputpath myout_{}_{} -o align'
    os.system(cmd.format(source, target, source, target, source, target, source, target))

def clean():
    os.system("make clean") 
    os.system("rm -rf myout*") 
    os.system("rm -f *vcb*")
    os.system("rm -f *snt")
    os.system("rm -f *cooc")
    os.system("rm -f en fr es")
    os.system("make")



def find_senses(lemma, df_train, df, spacy_en, spacy_fr, spacy_es, words_to_inds_map):
    not_annotated = []
    senses_to_inds = dict()
    jacket_occurences = 0
    
    file_fr_en = open("myout_fr_en/align.VA3.final", "r")
    file_es_en = open("myout_es_en/align.VA3.final", "r")
    
    file_en_fr = open("myout_en_fr/align.VA3.final", "r")
    file_en_es = open("myout_en_es/align.VA3.final", "r")
        
    lang_lines_all_to_en = [file_fr_en.readlines(), file_es_en.readlines()]
    lang_lines_en_to_all = [file_en_fr.readlines(), file_en_es.readlines()]
    
    file_fr_en.close(); file_en_fr.close()
    file_es_en.close(); file_en_es.close()
    
    for ind in words_to_inds_map[lemma]:
        en_text = clean_text(df.loc[ind, "text_en"], tokenization=False)
        doc = spacy_en(en_text)
        jacket_indices_and_words = [(token.i, token.text) for token in doc if token.lemma_ == lemma]
        #The index of the lemma (jacket) in the spacy tokenized text. It is the spacy tokenized texts that
        #are input to the classifiers!
        
        jacket_pos_tags = []
        for token in doc:
            if token.lemma_ == lemma:
                pos_tag = token.pos_
                if pos_tag == "PROPN": #proper noun to noun
                    pos_tag = "NOUN"
                elif pos_tag == "ADV": #adverb to adjective
                    pos_tag = "ADJ"
                jacket_pos_tags += [pos_tag]
                
        jacket_occurences += len(jacket_indices_and_words)
            
        if len(jacket_indices_and_words) == 0:
            continue
            
        alignments_all_to_en = [lines[3*ind+2].split(" ({") for lines in lang_lines_all_to_en]     
        alignments_en_to_all = [lines[3*ind+2].split(" ({") for lines in lang_lines_en_to_all] 
        
        for i, (jacket_index, jacket_word) in enumerate(jacket_indices_and_words):
            jacket_pos = jacket_pos_tags[i]
            sense = [lemma]
            pos_tags = [jacket_pos]
            foreign_indices = []
            
            #LOOP 1
            for lang_ind, alignment in enumerate(alignments_all_to_en):               
                found = False
                for i in range(1, len(alignment) - 1, 1):
                    if found:
                        break
                        
                    sen1 = alignment[i]
                    sen2 = alignment[i+1]
                    break_ind2 = sen2.index("}")
                    break_ind1 = sen1.index("}")
                    my_ind = i-1
                    
                    word = sen1[break_ind1+2:].strip()
                    inds = sen2[:break_ind2].split()
                            
                    break_it = False
                    for word_ind in inds:
                        if break_it:
                            break                            
                        if jacket_index + 1 == int(word_ind):
                            found = True
                            break_it = True
                            
                                                                            
                            if lang_ind == 0:                                
                                foreign_text = clean_text(df.loc[ind, "text_fr"], tokenization=False)
                                doc = spacy_fr(foreign_text)
                                for token in doc:
                                    if token.i == my_ind:
                                        sense += [token.lemma_]
                                        pos_tag = token.pos_
                                        if pos_tag == "PROPN":
                                            pos_tag = "NOUN"
                                        elif pos_tag == "ADV":
                                            pos_tag = "ADJ"
                                        pos_tags += [pos_tag]
                                        foreign_indices += [my_ind]
                                        break
                                        
                            else:
                                foreign_text = clean_text(df.loc[ind, "text_es"], tokenization=False)
                                doc = spacy_es(foreign_text)
                                for token in doc:
                                    if token.i == my_ind:
                                        sense += [token.lemma_]
                                        pos_tag = token.pos_
                                        if pos_tag == "PROPN":
                                            pos_tag = "NOUN"
                                        elif pos_tag == "ADV":
                                            pos_tag = "ADJ"
                                        pos_tags += [pos_tag]
                                        foreign_indices += [my_ind]
                                        break    
                                        
            if len(sense) != 3:
                not_annotated += [[clean_text(df.loc[ind, "text_en"], tokenizer=spacy_en.tokenizer), jacket_index]]
                continue
            
            #LOOP 2 (This is for bidirectional alignment. If not align back then ignore this training point.)   
            is_ok = True
            for lang_ind, alignment in enumerate(alignments_en_to_all):  
                if not is_ok:
                    break                 
                found = False
                target_index = foreign_indices[lang_ind]
                for i in range(1, len(alignment) - 1, 1):
                    if found or not is_ok:
                        break
                                            
                    sen1 = alignment[i]
                    sen2 = alignment[i+1]
                    break_ind2 = sen2.index("}")
                    break_ind1 = sen1.index("}")
                    
                    word = sen1[break_ind1+2:].strip()
                    inds = sen2[:break_ind2].split()
                            
                    break_it = False
                    for word_ind in inds:
                        if break_it:
                            break                            
                        if target_index + 1 == int(word_ind):
                            found = True
                            break_it = True
                            if word != jacket_word:
                                is_ok = False
                                                                                                           
            if is_ok:   
                
                #if not same POS-tag then skip it
                c = Counter(pos_tags)
                pos_tag, count = c.most_common()[0]
                if count != 3:
                    not_annotated += [[clean_text(df.loc[ind, "text_en"], tokenizer=spacy_en.tokenizer), jacket_index]]
                    continue
                    
                sense = tuple(sense)
                df_train.loc[len(df_train)] = [clean_text(df.loc[ind, "text_en"], tokenizer=spacy_en.tokenizer), jacket_index, sense, -1]
                                                            
                if sense not in senses_to_inds:
                    senses_to_inds[sense] = set()
                senses_to_inds[sense].add(len(df_train)-1)

            else:
                not_annotated += [[clean_text(df.loc[ind, "text_en"], tokenizer=spacy_en.tokenizer), jacket_index]]

    return (jacket_occurences, senses_to_inds, not_annotated)


def cut_off_senses(senses_to_inds, tot, threshold = 0.05):
    
    senses_to_inds_copy = senses_to_inds.copy()
    for sense, inds in senses_to_inds.items():
        if len(inds) < int(tot*threshold):
            senses_to_inds_copy.pop(sense)         
    return senses_to_inds_copy

def label_data(senses_to_inds, df_train):
    df_train = df_train.copy(deep = True)
    for label, (sense, inds),  in enumerate(senses_to_inds.items()):
        df_train.loc[inds, "label"] = label
        for ind in inds:
            df_train.loc[ind, 'sense'] = sense
    return df_train.loc[df_train["label"] != -1]
        
def generate_training_data(word, df, spacy_en, spacy_fr, spacy_es, words_to_inds_map, save_hist = True):
    
    df_train = pd.DataFrame(columns = ["text", "ind", "sense", "label"])
    occurences, senses_to_inds, not_annotated = find_senses(word, df_train, df, spacy_en, spacy_fr, spacy_es, words_to_inds_map)

    tot = sum([len(inds) for _, inds in senses_to_inds.items()])
    keep_percentage = round(len(df_train)/occurences*100,1)

    senses_to_inds = cut_off_senses(senses_to_inds, tot)
    
    df_train = label_data(senses_to_inds, df_train)
   
    df_train.reset_index(drop=True, inplace=True)    

    df_not_annotated = pd.DataFrame(columns=["text", "ind"])
    for ind, item in enumerate(not_annotated):
        df_not_annotated.loc[ind, "text"] = item[0]
        df_not_annotated.loc[ind, "ind"] = item[1]

    senses_to_count = dict.fromkeys(set(senses_to_inds))
    for key, val in senses_to_inds.items():
        senses_to_count[key] = len(val)


    df_sense = pd.DataFrame.from_dict(senses_to_count, orient = "index").reset_index()
    df_sense.rename(columns = {"index" : "sense", 0 : "count"}, inplace=True)
    
    if len(df_train["label"].unique()) < 2:
        return pd.DataFrame(), pd.DataFrame()
    
    if save_hist:
        plt.ioff()
        fig = plt.figure(figsize = (6,7))
        sns.barplot(x = "count", y = "sense", data = df_sense).set_title("Keeping {}%".format(keep_percentage))
        plt.savefig(os.path.join(PATH5, word + ".png"),bbox_inches='tight')
        plt.close()
    else:
        fig = plt.figure(figsize = (6,7))
        sns.barplot(x = "count", y = "sense", data = df_sense)

    return df_train, df_not_annotated

#Creates two dataframes with training data for this word. The first one is annotated data and second one is non-annotated
def parallel_function(word, df, words_to_inds_map):

    try:

        print("Doing word " + word)

        spacy_es = spacy.load("es_core_news_md")
        spacy_en = spacy.load('en_core_web_md')
        spacy_fr = spacy.load("fr_core_news_md")

         #No swedish model exists in spacy. Use english tokenizer for swedish
        tokenizers = [spacy_en.tokenizer, spacy_fr.tokenizer, spacy_es.tokenizer]
        lang_to_tokenizers = {"en" : tokenizers[0], "fr" : tokenizers[1], "es":tokenizers[2]}

        df_train, df_not_annotated = generate_training_data(word, df, spacy_en, spacy_fr, spacy_es, words_to_inds_map, save_hist=True)

        if not df_train.empty:
            df_train["text"] = df_train["text"].apply(lambda text: re.sub("\n", "", text))     
            df_train.to_csv(os.path.join(PATH5, word + ".csv") , index=False)

            if not df_not_annotated.empty:
                df_not_annotated["text"] = df_not_annotated["text"].apply(lambda text: re.sub("\n", "", text))
                df_not_annotated.to_csv(os.path.join(PATH5, word + "_not_annotated.csv") , index=False)

    except Exception as e:
        print(word, str(e))

#Words mapping to indices in the dataframe that it occurs in. This is for speed.
def words_to_inds(df, spacy_en):
    words_to_inds_map = dict()
    for ind, _ in df.iterrows(): 
        en_text = clean_text(df.loc[ind, "text_en"], tokenization=False)
        doc = spacy_en(en_text)
        lemmas = [token.lemma_ for token in doc]
        for lemma in lemmas:
            if lemma not in words_to_inds_map:
                words_to_inds_map[lemma] = set()
            words_to_inds_map[lemma].add(ind)
    return words_to_inds_map


if __name__ == '__main__':
    main()
            




