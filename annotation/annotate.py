import pandas as pd
from pprint import pprint
import time


import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
print('\n\n\n')



def main():
    
    c = Color()

    path_anno = 'annotated/{}.csv'
    path = '../data/train_wsd/{}/{}.csv'

    while True:
        q = c.bold("Which word do you want to annotate? ")
        word = input(q)
        try:
            df = pd.read_csv(path_anno.format(word))
            break
        except FileNotFoundError as e:
            print(e)
            try:
                df = pd.read_csv(path.format(word, word))
                df['wordnet_sense'] = [None] * len(df)
                break
            except FileNotFoundError:
                print('Error! Could not find file. Try another word.')


    synsets = wordnet.synsets(word)
    data = [(s.name(), s.pos(), s.definition(), s.examples()) for s in synsets]
    df_wordnet = pd.DataFrame(data, columns = ['Name', 'POS', 'Definition', 'Examples'])





    run(word, df, df_wordnet)


def run(word, df, df_wordnet):

    time.sleep(1)

    c = Color()

    text_indices  = list(df[df['wordnet_sense'].isnull()].index)
    index = text_indices[0]



    while len(text_indices) > 0:


        print("WordNet senses.")
        print(df_wordnet)
        print("\nWrite 'examine' to examine a particular WordNet sense.")
        print("Write 'skip' wait with this text.")
        print("Write 'save' to save current progress.\n")


        text = df.loc[index, 'text']
        word_ind = df.loc[index, 'ind']

        words = text.split()
        words[word_ind] = c.bold(c.red(words[word_ind]))

        text = ' '.join(words)

        print('Number of texts left:', len(text_indices))
        print(text_indices)

        print('\n\n')
        print(c.header(c.underline('TEXT:')))
        print(text)
        print('\n')

        q = c.bold("Which WordNet sense? ")
        inp = input(q)

        try:
            sense = int(inp)
            if sense in df_wordnet.index:
                df.loc[index, 'wordnet_sense'] = sense
                print(c.header('Text labeled!'))
                time.sleep(0.45)
                print('\n\n' + '=' * 30 + '\n\n')
                time.sleep(0.45)

                text_indices = text_indices[1:]
                index = text_indices[0]
            else:
                time.sleep(0.5)
                print("\nNot a WordNet sense!\n")
                time.sleep(0.5)

        except ValueError:
            if inp == 'examine':
                examine_senses(df_wordnet)
            elif inp == 'skip': # Skip for now and add it last in the queue
                text_indices = text_indices[1:] + [index]
                index = text_indices[0]
            elif inp == 'save':
                fname = 'annotated/{}.csv'.format(word)
                print('Saving progress to', fname)
                time.sleep(0.5)
                print(df)
                df.to_csv(fname)
                time.sleep(0.5)



        print('\n\n\n')

def examine_senses(df_wordnet):
    c = Color()

    while True:
        try:
            q = c.bold("Which sense would you like to examine? ")
            sense = int(input(q))

            if sense not in df_wordnet.index:
                print("\nNot a WordNet sense!\n")
                time.sleep(0.5)
                continue

            row = df_wordnet.loc[sense]

            print('\n\n')
            print(c.header(c.underline('Sense ' + str(sense))))
            print(c.bold('Name:'), row.Name)
            print(c.bold('POS:'), row.POS)
            print(c.bold('Definition:'), row.Definition)
            print(c.bold('Examples:'), row.Examples)
            print('\n\n')
            
            q = c.bold('Continue examining senses? ')
            if input(q) in ['no', 'No', 'NO', '0', 'False', 'false']:
                break

        except ValueError:
            pass



class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def green(self, string):
        return self.OKGREEN + string + self.ENDC
    def blue(self, string):
        return self.OKBLUE + string + self.ENDC
    def red(self, string):
        return self.FAIL + string + self.ENDC
    def bold(self, string):
        return self.BOLD + string + self.ENDC
    def underline(self, string):
        return self.UNDERLINE + string + self.ENDC
    def header(self, string):
        return self.HEADER + string + self.ENDC






if __name__ == '__main__':
    main()