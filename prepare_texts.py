import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm

"""
Got to the link below for books in multiple languages: 
http://farkastranslations.com/bilingual_books.php
"""

URLS = [
    'http://farkastranslations.com/books/Verne_Jules-Ile_mysterieuse-fr-en-es-hu.html',
    'http://farkastranslations.com/books/Dumas_Alexandre-Trois_Mousquetaires-fr-en-hu-es-nl.html',
    ]
LANGUAGES = [
    'English', 'French', 'Spanish'
]

FNAME_WORD_ALIGNMENT = 'data/word_alignment.txt'
FNAME_BERT = 'data/bert.txt'

try:
    with open('data/lemmas.txt', 'r') as f:
        WORDS = [word.strip() for word in f.readlines()]
except FileNotFoundError:
    WORDS = None

print(WORDS)



def main():

    df = get_data_from_urls(URLS)

    prepare_for_word_alignment(df, fname = FNAME_WORD_ALIGNMENT)
    prepare_for_bert(df, fname = FNAME_BERT)

    print_stats(df)

    




###############################
# Fetch data

def get_data_from_urls(urls):
    df = pd.DataFrame()
    for url in urls:
        df_url = get_data_from_url(url)
        df = df.append(df_url)
    return df

def get_data_from_url(url):
    resp = requests.get(url)
    df = pd.read_html(resp.content)[0]
    df.columns = df.loc[0].values
    df = df.drop(0).reset_index(drop=True)
    return df[LANGUAGES].dropna()



###############################
# Prepare and write data to files

def prepare_for_word_alignment(df, fname = None):
    if fname:
        df.to_csv(fname, index = False)
    else:
        pass

def prepare_for_bert(df, fname = None):
    texts = []
    for text in df.English:
        sents = [sent.strip() for sent in text.strip().split('.')]
        sents = '.\n'.join(sents)
        texts.append(sents.strip())
    string = '\n\n'.join(texts)
    if fname:
        with open(fname, 'w') as f:
            f.write(string)
    return string





###############################
# Print and visualize data


def print_stats(df, language = 'English'):

    print("Loading Spacy...")
    sp = spacy.load('en_core_web_md')
    print("Spacy loaded\n")

    print('\nSTATS FOR', language.upper())
    print('-'* (10 + len(language)))
    print('Number of texts:', len(df))


    series = df[language].dropna()

    lemmas = []

    num = 0
    for text in tqdm(series, desc = 'Counting lemmas and words'):
        for token in sp(text):
            #if token.lemma_ not in ['-PRON-', ',', '.', '.', '"', '!', '?', ';', '-']
            if token.pos_ in ["PUNCT", "PRON"]:
                continue
            lemmas.append(token.lemma_)
        num += len(text.split(' '))


    print('Number of words:', num)
    print('Number of lemmas:', len(set(lemmas)))

    print()

    occurences = {}
    for word in lemmas:
        try:
            occurences[word] += 1
        except KeyError:
            occurences[word] = 1

    df = pd.DataFrame(occurences.items(), columns = ['Lemma', 'Occurences'])
    df = df.sort_values(by = 'Occurences', ascending = False)
    df = df.reset_index(drop = True)

    if WORDS:
        print(df[df['Lemma'].isin(WORDS)])
    else:
        print(df.loc[:50])
    



if __name__ == '__main__':
    main()



