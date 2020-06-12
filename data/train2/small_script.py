import os
import pandas as pd
from collections import Counter

'''
write_file = open('merg.results.txt', 'w')
for file in os.listdir():
	try:
		if os.path.isdir(file):
			df = pd.read_csv(os.path.join(file, file + '.csv'))
			diff = len(set(df['label'].values)) - len(set(df['new_label'].values))
			if diff >= 1:
				write_file.write(file + " " + str(diff) + "\n")
	except Exception as e:
		pass
write_file.close()
'''

'''
file = open('merg.results.txt', 'r')
words = [word[:word.index(' ')] for word in file.readlines()]

for word in words:
	os.system('cp ' + word + '.csv ' + '../train2')

file.close()
'''

write_file = open('top_10_words.txt', 'w')
words = {}

for file in os.listdir():
	if os.path.isdir(file):
		length = len(pd.read_csv(os.path.join(file, file + '.csv')))
		words[file] = length

counter = Counter(words)
for key, val in counter.most_common(20):
	write_file.write(key  + ' ' + str(val) + '\n')

write_file.close()
		
