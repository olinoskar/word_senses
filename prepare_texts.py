import pandas as pd
import requests
from bs4 import BeautifulSoup

"""
Got to the link below for books in multiple languages: 
http://farkastranslations.com/bilingual_books.php
"""

URLS = [
    'http://farkastranslations.com/books/Verne_Jules-Ile_mysterieuse-fr-en-es-hu.html',
    #'http://farkastranslations.com/books/Doyle_Arthur_Conan-A_Study_in_Scarlet-en-fr-es.html'
    ]
LANGUAGES = [
    'English', 'French', 'Spanish'
]

FNAME_WORD_ALIGNMENT = 'data/word_alignment.txt'
FNAME_BERT = 'data/bert.txt'


def main():

    df = get_data_from_urls(URLS)
    prepare_for_word_alignment(df, fname = FNAME_WORD_ALIGNMENT)
    prepare_for_bert(df, fname = FNAME_BERT)




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
# Prepare and write data

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
    print('Number of texts:', len(df))
    print(df)

    series = df[language].dropna()

    unique_words = set()
    num = 0
    for text in series:
        num += len(text.split(' '))
        for word in text.split(' '):
            unique_words.add(word)

    print('Number of words:', num)
    print('Number of unique words:', len(unique_words))



if __name__ == '__main__':
    main()



