import sys
import os
import re
from os import walk
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import time
import datetime
import io
import math
from collections import Counter
import glob
from pathlib import Path
import pandas as pd
import csv


add = []
freq = {}
d = {}
position = 1

t_dir = r'C:\Users\Vrindavan\Downloads\Krithika_Verma_HW1\TF_IDF'
dict_query_wt = Counter()  # dictionary for query terms and its weight if provided
dict_rel_doc = Counter()   # dictionary containing relevant documents and its weight for all the query terms



tokens_files3 = os.listdir(t_dir)
df1_of_all_files = []                 # Dataframe listing the contents of all files
for entry6 in tokens_files3:
    path6 = os.path.join(t_dir,entry6)
    df2 = pd.read_csv(path6, delimiter = ',')   # reading each file and storing in df2
    df1_of_all_files.append(df2)  

connect = pd.concat(df1_of_all_files, axis=0)     # This contains all the files data in one single file
connect.to_csv('tf_idf_all_files.csv',index =None)

df3 = pd.read_csv('tf_idf_all_files.csv',delimiter = ',')
# converting the word, tf_idf value of the word, DocID into a Term document matrix
df4 = df3.pivot_table(values = 'TfIdf',index= 'Term',columns='DocId') 
# replacing NaN values in matrix with value 0
df5 = df4.fillna(value= 0, method= None, axis= None) 

df6 = df5.to_dict('index')
# converting the matrix into nested dictionary
TDM = {k:{k1:v1 for k1,v1 in v.items() if v1 != 0} for k,v in df6.items()}


#writing DocID and the normalized weight from the nested dictionary to posting file
with open('posting.txt','w', encoding = 'ASCII', errors = 'ignore') as f6:
    for key, value in TDM.items():
        for key1,val1 in sorted(value.items()):
            f6.write(str(key1)+','+str(val1)+'\n')

#calculating the frequency of the words in the posting file
for k2,v2 in TDM.items():
    frequency = len(v2)
    d[k2] = frequency

#writing word, frequency of word in posting file and first record position of word in posting file
with open('dictionary.txt','w', encoding = 'ASCII', errors = 'ignore' ) as f7:
    for k3, v3 in d.items():
        d[k3] = position
        position = position + v3
        f7.write(str(k3)+'\n'+str(v3)+'\n'+str(d[k3])+'\n')


# Retrieve relevant documents based on query and its weights
def retrieve_query_with_weights():
	query_t = zip(sys.argv[2::2], sys.argv[3::2])
	for wt, term in query_t:
		stop_file = open('stoplist.txt', 'r', encoding = 'utf-8', errors = 'ignore') # removes stop words from query if present
		stopwords = stop_file.read()
		stopwords = stopwords.split()
		if (len(term) == 1) or term in stopwords:
			pass
		else:
			final_term = term.lower()  # lower case the alphabet
		if len(final_term) > 0:
			dict_query_wt[final_term] = wt # store the query weight in dictionary

	# travers the Term document matrix to find the relevant documents
	for k4,v4 in dict_query_wt.items():
		dict_mul_wts = Counter()
		for k5, v5 in TDM.items():
			if k4 == k5:
				for k6, v6 in v5.items():
					dict_mul_wts[k6] = float(v6) * float(v4)
				break

		for k7, v7 in dict_mul_wts.items():
			if k7 not in dict_rel_doc.items():
				dict_rel_doc[k7] = v7
			else:
				dict_rel_doc[k7] += v7

	top_10 = dict_rel_doc.most_common(10) # display top 10 relevant documents
	if (not dict_rel_doc):
		print("The search engine has no relevant documents for the query provided")
	else:
		for rel in top_10:
			print("Doc Id: " +str(rel[0])+ ".html Score: "+str(rel[1]))



f_term = []

# function to find relevant documents when only query terms is provided
def retrieve_only_query():
	for i in range(len(sys.argv)):
		query = sys.argv[1:]
	for term in query:
		stop_file = open('stoplist.txt', 'r', encoding = 'utf-8', errors = 'ignore') # remove stop words from the query if present
		stopwords = stop_file.read()
		stopwords = stopwords.split()
		if (len(term) == 1) or term in stopwords:
			pass
		else:
			f_term.append(term.lower())

	for term in f_term:
		dict_wts = {}
		for l2, r2 in TDM.items():  # traverse the TDM to find relevant documents
			if(term == l2):
				for l3, r3 in r2.items():
					dict_wts[l3] = r3
				break

		for l5, r5 in dict_wts.items():
			if l5 not in dict_rel_doc.keys():
				dict_rel_doc[l5] = r5
			else:
				dict_rel_doc[l5] += r5
			

	top_10 = dict_rel_doc.most_common(10) # display top 10 relevant documents
	if (not dict_rel_doc):
		print("The search engine has no relevant documents for the query provided")
	else:
		for rel in top_10:
			print("Doc Id: " +str(rel[0])+ ".html Score: "+str(rel[1]))
		



if (sys.argv[1] == 'wt'):
	retrieve_query_with_weights()   # call this function if query weight is provided along with query terms
else:
	retrieve_only_query()           # call this function if only query terms are provided







