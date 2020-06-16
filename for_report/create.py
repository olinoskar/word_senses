import pandas as pd
import json
from pprint import pprint
from collections import Counter


def main():
    create_wsd_table()
    create_senses_table()



def create_wsd_table():
    print('Creating WSD table')

    fname = '../data/results_from_wsd2.csv'
    df = pd.read_csv(fname)


    df.columns = [col.replace('nbr. of', '\\#') for col in df.columns]
    df = df.sort_values(by = 'word')
    average = ['average'] + list(df.mean().values)


    for i, val in enumerate(average):
        if i in [1, 3]:
            average[i] = float2str(val, dec = 4)
        elif i in [2, 4, 5]:
            average[i] = float2str(val, dec = 1)



    df['acc.before'] = df['acc.before'].apply(lambda num: float2str(num, dec = 4))
    df['acc.after'] = df['acc.after'].apply(lambda num: float2str(num, dec = 4))
    df['\\# senses before'] = df['\\# senses before'].apply(round)
    df['\\# senses after'] = df['\\# senses after'].apply(round)

    df.loc['average'] = [bold(str(val)) for val in average]

    print(df)

    df = df[['word','total','acc.before','\\# senses before','acc.after','\\# senses after']]

    df.columns = [bold(col) for col in df.columns]

    tex = df.to_latex(
        index = False,
        column_format = '|'.join(['c']*len(df.columns)),
        escape = False,
        label = 'tab:wsd_table',
        bold_rows = True,
        caption = 'Number of senses and WSD-accuracy before and after merging senses. We can see that the WSD-accuracy increases after merging senses.'
        )

    with open('tables/wsd_table.tex', 'w') as f:
        f.write(tex)


def create_senses_table():

    print('Creating senses table')

    fname = '../data/results_from_wsd2.csv'
    df = pd.read_csv(fname)
    words = list(df.word.values)

    words = [
        'cry',
        'evening',
        'face',
        'fall',
        #'leave',
        'master',
        #'remain',
        'reply',
        'return',
        'room',
        'turn',
        #'vessel',
        #'wall',
        'work',
    ]

    words.sort()
    print(words)
    



    # Get data
    fname_before = '../data/train/{}.csv'
    data_before = {}
    for word in words:
        data_before[word] = {'senses':[]}

        fname = fname_before.format(word)
        df = pd.read_csv(fname)
        print(df)
        print()
        counter = Counter(list(df['sense'].values))
        senses = {key:val for key, val in counter.items() if val > 10}

        for key, val in senses.items():
            data_before[word]['senses'].append(
                {
                    'name': key,
                    'ratio': val / sum(senses.values())
                }
            )
        print(counter)
        print(senses)
        print()


    #pprint(data_before)
    #fname_before = '../data/training_wsd_before2/results.json'
    

    #with open(fname_before, 'r') as f:
    #    data_before = json.loads(f.read())

    fname_after = '../data/training_wsd_after/results.json'
    with open(fname_after, 'r') as f:
        data_after = json.loads(f.read())

    data = {'before': {}, 'after': {}}

    for word in words:
        data['before'][word] = data_before[word]['senses'] if data_before[word] != {} else {}
        data['after'][word] = data_after[word]['senses'] if data_after[word] != {} else {}


    # Make latex from data
    lines = [
        "\\begin{table}[h]", "\\caption{Sense definitions and ratios (R) of each sense in the training data\
         of a few words before and after merging.}",
        "\\label{tab:senses_table}","\\begin{tabular}{@{}c|c|c|c|c@{}}", "\\toprule", "  \\textbf{word} &",
        "  \\textbf{senses before} &", "  \\textbf{R before} &", "  \\textbf{senses after} &",
        "  \\textbf{R after} \\\\", "\\midrule\n"  
    ]

    tex = '\n'.join(lines)

    for word in words:

        tex_word = word + " &\n"
        lines = []
        for temp in ['before', 'after']:
            names, ratios = [], []
            for sense in data[temp][word]:
                names.append(sense['name'].replace("'", ''))
                ratios.append(str(round(sense['ratio'], 2)))

            lines += ["  \\begin{tabular}[c]{@{}c@{}}" + '\\\\ '.join(names) + "\\end{tabular} "]
            lines += ["  \\begin{tabular}[c]{@{}c@{}}" + '\\\\ '.join(ratios) + "\\end{tabular}" ]
        
        tex_word += ' &\n'.join(lines)
        tex_word += ' \\\\ \\hline\n'
        tex += tex_word

    tex += "\n\\bottomrule \n\\end{tabular} \n\\end{table}"


    pprint(words)
 

    with open('tables/senses_table.tex', 'w') as f:
        f.write(tex)

    





#########################################################
# Helper functions

def bold(string):
    return "\\textbf{%s}" % string

def float2str(num, dec = 3):
    try:
        res = str(round(num, dec))
        res += '0' * (2+dec - len(res) )
        return res #str(round(num, dec))
    except TypeError:
        return num


def str2tuple(string):
    try:
        s = ast.literal_eval(str(string))
        if type(s) == tuple:
            return s
        return
    except:
        return



if __name__ == '__main__':
    main()