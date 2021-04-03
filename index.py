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
d = {}
position = 1

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


#Removing stop words and length of one word in entire corpus           
tokens_files = os.listdir(o_dir) 
for entry2 in tokens_files:
    path2 = os.path.join(o_dir,entry2)
    with open (path2, 'r', encoding = 'utf-8', errors= 'ignore') as f3:
        line = f3.read()
        words3= line.split()
        
    nostop = [p for p in words3 if p not in stopwords]
    
    store = {}
    for k in nostop:
        count1 = store.get(k,0)
        store[k] = count1+1
     
    # writing tokens and count after removing stop words in csv   
    file_name2 , file_extension2 = os.path.splitext(entry2)
    path3 = os.path.join(s_dir,file_name2 + '.csv')
    with open(path3,'w', encoding = 'utf-8', newline = '', errors = 'ignore' ) as f4:

        writer = csv.writer(f4)
        for t in list(store.keys()):
            if (freq[t] > 1):
                writer.writerow([str(t), str(store[t])])
        
         
    df = pd.read_csv(path3, header = None)
    df.rename(columns={0 : 'term', 1:'count', 2: 'tf'}, inplace = True)
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
counter1 = 1
tokens_files2 = os.listdir(s_dir)
for entry5 in tokens_files2:
    path5 = os.path.join(s_dir,entry5)    
    df55 = pd.read_csv(path5)
    tf_idf = {}
    tf_dict = dict(zip(df55.term, df55.tf))
    for key, val in tf_dict.items():
        tf_idf[key] = float(val) * idf[key]           # storing tf * idf in a dictionary
    

     # writing the term weights in a text file
    file_name5 , file_extension5 = os.path.splitext(entry5)
    output_path5 = os.path.join(t_dir, file_name5 +'.csv')
    with open (output_path5, 'w', encoding = 'utf-8', newline = '', errors= 'ignore') as outfile5:
        writer1 = csv.writer(outfile5)
        for key2,val2 in tf_idf.items(): #key= lambda item: (item[1],item[0]), reverse = True):
            writer1.writerow([str(key2),str(counter1),str(tf_idf[key2])])
        
    counter1 += 1 # storing the DocId in each file 
    df2 = pd.read_csv(output_path5, header = None)
    df2.rename(columns={0 : 'Term', 1:'DocId', 2: 'TfIdf'}, inplace = True)
    df2.to_csv(output_path5,index = False)


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
df7 = {k:{k1:v1 for k1,v1 in v.items() if v1 != 0} for k,v in df6.items()}


#writing DocID and the normalized weight from the nested dictionary to posting file
with open('posting.txt','w', encoding = 'ASCII', errors = 'ignore') as f6:
    for key, value in df7.items():
        for key1,val1 in sorted(value.items()):
            f6.write(str(key1)+','+str(val1)+'\n')

#calculating the frequency of the words in the posting file
for k2,v2 in df7.items():
    frequency = len(v2)
    d[k2] = frequency

#writing word, frequency of word in posting file and first record position of word in posting file
with open('dictionary.txt','w', encoding = 'ASCII', errors = 'ignore' ) as f7:
    for k3, v3 in d.items():
        d[k3] = position
        position = position + v3
        f7.write(str(k3)+'\n'+str(v3)+'\n'+str(d[k3])+'\n')

elapsed = datetime.datetime.now() - start_time
print(int(elapsed.total_seconds()))
