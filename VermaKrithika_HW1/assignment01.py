import sys
import os
import re
from os import walk
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize	as nl
from nltk.tokenize import RegexpTokenizer
import time
import datetime

#start_time = time.process_time()

start_time= datetime.datetime.now()
print (start_time)
freq = { }
i_dir = str(sys.argv[1])
o_dir = str(sys.argv[2])

print("Input directory is:", i_dir)
print("output directory is:", o_dir)


files = os.listdir(i_dir)
for entry in files:
	path = os.path.join(i_dir, entry)
	with open(path, 'r', encoding ='utf-8', errors= 'ignore') as f:
		soup = BeautifulSoup(f, 'html.parser')
		content = soup.get_text()

		
		reg = RegexpTokenizer(r'\w+')
		words = reg.tokenize(content) #[word for word in content.split() if word.isalpha()]
		tokens = [word.lower() for word in words if word.isalpha()]

		#output directory containing tokenized files
		output_path = os.path.join(o_dir, entry +'.txt')
		with open(output_path, 'w', encoding = 'utf-8', errors= 'ignore') as outfile:
			for words in tokens:
				a = outfile.write(str(words))
				outfile.write('\n')

		
		for i in tokens:
			count = freq.get(i,0)
			freq[i] = count+1

# file of tokens & frequencies sorted by tokens
with open (r'C:\Users\Vrindavan\Downloads\IR\alpha.txt', 'w', encoding = 'utf-8', errors= 'ignore') as alphafile:
	#for j in aa:
	for key in sorted(freq.keys()):
		alphafile.write(key)
		alphafile.write('\n')
		alphafile.write(str(freq[key]))
		alphafile.write('\n')

# file of tokens & frequencies sorted by frequency
with open (r'C:\Users\Vrindavan\Downloads\IR\freq.txt', 'w', encoding = 'utf-8', errors= 'ignore') as freqfile:
	for key, value in sorted(freq.items(), key = lambda item: (item[1], item[0])): #refered from https://www.geeksforgeeks.org/python-sort-python-dictionaries-by-key-or-value/
		freqfile.write(key)
		freqfile.write('\n')
		freqfile.write(str(value))
		freqfile.write('\n')


elapsed = datetime.datetime.now() - start_time
print(int(elapsed.total_seconds()*1000))
#print (time.process_time() - start_time, 'seconds')