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

start_time= datetime.datetime.now()
print (start_time)
i_dir = str(sys.argv[1])
o_dir = str(sys.argv[2])
print("Input directory is:", i_dir)
print("output directory is:", o_dir)

s_dir = r'C:\Users\Vrindavan\Downloads\Krithika_Verma_HW1\stop'
t_dir = r'C:\Users\Vrindavan\Downloads\Krithika_Verma_HW1\TF_IDF'
add = []
freq = {}

files = os.listdir(i_dir)
stopwords = open(r'C:\Users\Vrindavan\Downloads\Krithika_Verma_HW1\stoplist.txt', 'r').read().split()

# tokenizing the raw html files 
for entry in files:
    path = os.path.join(i_dir, entry)
    with open(path, 'r', encoding ='utf-8', errors= 'ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')
        content = soup.get_text()
        
        reg = RegexpTokenizer(r'\w+')
        words = reg.tokenize(content) #[word for word in content.split() if word.isalpha()]
        tokens = [word.lower() for word in words if word.isalpha()]
        add.append(tokens)
        
        # writing the tokens to output files 
        file_name3 , file_extension3 = os.path.splitext(entry)
        output_path2 = os.path.join(o_dir, file_name3 +'.txt')
        with open (output_path2, 'w', encoding = 'utf-8', errors= 'ignore') as outfile1:
            for P in tokens:
                a = outfile1.write(str(P))
                outfile1.write('\n')
                
        for J in tokens:
            count = freq.get(J,0)
            freq[J] = count+1

# file containing tokens and frequencies sorted by tokens
with open (r'C:\Users\Vrindavan\Downloads\Krithika_Verma_HW1\alpha.txt', 'w', encoding = 'utf-8', errors= 'ignore') as alphafile:
    #for j in aa:
    for key in sorted(freq.keys()):
        #if (freq[key] > 1):
        alphafile.write(key)
        alphafile.write('\n')
        alphafile.write(str(freq[key]))
        alphafile.write('\n')

# file of tokens & frequencies sorted by frequency
with open (r'C:\Users\Vrindavan\Downloads\Krithika_Verma_HW1\freq.txt', 'w', encoding = 'utf-8', errors= 'ignore') as freqfile:
    for key, value in sorted(freq.items(), key = lambda item: (item[1], item[0])): 
        freqfile.write(key)
        freqfile.write('\n')
        freqfile.write(str(value))
        freqfile.write('\n')  


#Removing stop words and length of one word in entire corpus           
tokens_files = os.listdir(o_dir)
for entry2 in tokens_files:
    path2 = os.path.join(o_dir,entry2)
    with open (path2, 'r', encoding = 'utf-8', errors= 'ignore') as outfile2:
        line = outfile2.read()
        words3= line.split()
        
    nostop = [p for p in words3 if p not in stopwords]
    
    store = {}
    for kri in nostop:
        count1 = store.get(kri,0)
        store[kri] = count1+1
     
    # writing tokens and count after removing stop words in csv   
    file_name2 , file_extension2 = os.path.splitext(entry2)
    path3 = os.path.join(s_dir,file_name2 + '.csv')
    with open(path3,'w', encoding = 'utf-8', newline = '', errors = 'ignore' ) as j:
        writer = csv.writer(j)
        for tri in list(store.keys()):
            if (freq[tri] > 1):
                writer.writerow([str(tri), str(store[tri])])
        
            
    df = pd.read_csv(path3, header = None)
    df.rename(columns={0 : 'term', 1:'count', 2: 'idfs', 3: 'news'}, inplace = True)
    df.to_csv(path3,index = False)
    
    # calculating The TF values for each word in the document
    df['tf'] = df['count'].div(df['count'].sum())
    df.to_csv(path3, index = False)

 
# Creating a list of all 500 files   
all_txt_files = []
for file in Path(r'C:\\Users\\Vrindavan\\Downloads\\Krithika_Verma_HW1\\stop').rglob("*.csv"):
    all_txt_files.append(file.parent / file.name)


# creating dictionary to store the number of documents containing the word
DF = Counter()
for i in all_txt_files:
    
    reader = pd.read_csv(i)
    terms = tuple(reader.term)
    for tokens in terms:
        if tokens not in DF.keys():
            DF[tokens] = 1
        else:
            DF[tokens] += 1
            


N = len(all_txt_files)
idf = Counter()
import math

# calculating IDF and storing the results of all words in the corpus in idf dictionary    
for ( term, term_freq) in DF.items():
    idf[term]= math.log(float(N) / term_freq)

# Multiplying TF and IDF
tokens_files2 = os.listdir(s_dir)
for entry5 in tokens_files2:
    path5 = os.path.join(s_dir,entry5)    
    df55 = pd.read_csv(path5)
    tf_idf = {}
    tf_dict = dict(zip(df55.term, df55.tf))
    for key, val in tf_dict.items():
        tf_idf[key] = float(val) * idf[key]             # storing tf * idf in a dictionary

    # writing the term weights in a text file
    file_name5 , file_extension5 = os.path.splitext(entry5)
    output_path5 = os.path.join(t_dir, file_name5 +'.txt')
    with open (output_path5, 'w', encoding = 'utf-8', errors= 'ignore') as outfile5:
        for key2, val2 in sorted(tf_idf.items(), key= lambda item: (item[1],item[0]), reverse = True):
            outfile5.write(str(key2)+' '+str(val2)+'\n')

elapsed = datetime.datetime.now() - start_time
print(int(elapsed.total_seconds()))
