import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
import sys
from pprint import pprint


"""
Go to the link below for books in multiple languages: 
http://farkastranslations.com/bilingual_books.php
"""

URLS = [
    'http://farkastranslations.com/books/Verne_Jules-Ile_mysterieuse-fr-en-es-hu.html',
    'http://farkastranslations.com/books/Dumas_Alexandre-Trois_Mousquetaires-fr-en-hu-es-nl.html',
    'http://farkastranslations.com/books/Bronte_Charlotte-Jane_Eyre-en-fr-es-it-de-hu.html',
    'http://farkastranslations.com/books/Verne_Jules-20000_lieues_sous_les_mers-fr-en-hu-es-nl.html',
    ]
LANGUAGES = [
    'English', 'French', 'Spanish'
]

SHORT_LANGS_MAP = {
    'English': 'en', 'French': 'fr', 'Spanish': 'es',
}

FNAME_WORD_ALIGNMENT = 'data/word_alignment.csv'
FNAME_BERT = 'data/bert.txt'
FNAME_OCCURENCES = 'data/occurences.csv'

try:
    with open('data/lemmas.txt', 'r') as f:
        WORDS = [word.strip() for word in f.readlines()]
except FileNotFoundError:
    WORDS = None

print(WORDS)



def main():

    urls = get_book_urls()




    print_old_stats()

    sys.exit()

    df = get_data_from_urls(urls)


    prepare_for_word_alignment(df, fname = FNAME_WORD_ALIGNMENT)
    prepare_for_bert(df, fname = FNAME_BERT)

    print_stats(df)

    
###############################
# Fetch data

def get_data_from_urls(urls):
    df = pd.DataFrame()
    for url in urls:
        print(url)
        df_url = get_data_from_url(url)
        df = df.append(df_url)
    return df

def get_data_from_url(url):
    resp = requests.get(url)
    df = pd.read_html(resp.content)[0]
    df.columns = df.loc[0].values
    df = df.drop(0).reset_index(drop=True)
    return df[LANGUAGES].dropna()


def get_book_urls():

    root_url = 'http://farkastranslations.com/bilingual_books.php'
    resp = requests.get(root_url)

    soup = BeautifulSoup(resp.text, 'html.parser')
    urls = set()

    langs = [SHORT_LANGS_MAP[lang] for lang in LANGUAGES]

    for tag in soup.find_all('a'):
        url = tag['href']

        if not url.endswith('.html'):
            continue

        all_in_url = True
        for lang in langs:
            if lang not in url.replace('.html', '').split('-'):
                all_in_url = False
                break
        if not all_in_url:
            continue

        urls.add(url)




        print(url)
    pprint(urls)

    urls = ['http://farkastranslations.com/' + url for url in urls]
    return urls



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


def print_old_stats():
    df = pd.read_csv(FNAME_OCCURENCES)
    df = df.dropna()

    short_words = set([word for word in df['Lemma'] if len(word) <= 3])

    stop_words = set(stopwords.words('english')) 
    stop_words.add('-PRON-')
    stop_words = stop_words.union(short_words) 

    df = df[~df['Lemma'].isin(stop_words)]
    print(df[:50])



def print_stats(df, language = 'English'):

    print('\nSTATS FOR', language.upper())
    print('-'* (10 + len(language)))
    print('Number of texts:', len(df))

    print("Loading Spacy...")
    sp = spacy.load('en_core_web_md')
    print("Spacy loaded\n")

    


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
    df.to_csv(FNAME_OCCURENCES, index = False)

    if WORDS:
        print(df[df['Lemma'].isin(WORDS)])
    else:
        print(df.loc[:50])
    



if __name__ == '__main__':
    main()



