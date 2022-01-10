import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from collections import Counter
from sklearn import preprocessing
import pickle




def main():
    ps = PorterStemmer()
    table = str.maketrans(dict.fromkeys(string.punctuation))
    stopwords = {'i', 'me', 'my', 'myself', 'we', 
    'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
    'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', 
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "covid-19", 
    "covid", "covid19", "coronavirus", "covid19s", "covids"}
    
    path = 'BC7-LitCovid-Train.csv'
    title_abstract_word_frequency = []
    bag_of_words = set()
    df = pd.read_csv(path, sep=',')
    mega_index = 24960
    
    #separeting multi labels and adding them one by one into the dataset
    '''
    for index, row in enumerate(df['label']):
        if ";" in row:
            arr = row.split(";")
            df.loc[index,['label']] = arr[0]
            for i in range(1, len(arr)):
                df.loc[mega_index, ['abstract']] = df.loc[index, ['abstract']]
                df.loc[mega_index, ['title']] = df.loc[index, ['title']]
                df.loc[mega_index,['label']] = arr[i]
                mega_index += 1
    
    write_file = open("df.pkl", "wb")
    
    
    pickle.dump(df, write_file)
    '''
	
	#encoding the labels
    '''
    #le = preprocessing.LabelEncoder()
    #le.fit(['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report'])
    #arr = le.transform(df['label'])
    #print(arr)
    '''
    

    '''
    df = pickle.load(open("df.pkl", "rb"))
    labels = ['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report']
    for label in labels:
        tokens = []
        xf = df[df["label"].str.contains(label)]
        #Y = df["label"]
        title_abstracts = xf['title'] + xf['abstract']
        print(len(title_abstracts))
    
        for t in title_abstracts:
            t = t.lower()
            t = t.translate(table)
            t = t.split()
            new_t = []
            for word in t:
                stemmed_t = ps.stem(word)
                if stemmed_t not in stopwords and word not in stopwords and not word.isnumeric():
                    new_t.append(stemmed_t)
                    tokens.append(stemmed_t)
            #title_abstracts[index] = "".join(new_t)
            new_t_word_freq = Counter(new_t)
            title_abstract_word_frequency.append(dict(new_t_word_freq))
    
    write_file = open("title_abstract_word_frequency.pkl", "wb")
    pickle.dump(title_abstract_word_frequency, write_file)

        
        x = Counter(tokens)
        for word, count in x.items():
            if count > 20:
                bag_of_words.add(word)

    write_file = open("new_tokens.pkl", "wb")
    pickle .dump(bag_of_words, write_file)
    '''


    

    title_abstract_word_frequency = pickle.load(open("title_abstract_word_frequency.pkl", "rb"))
    tokens = pickle.load(open("new_tokens.pkl", "rb"))
    df = pickle.load(open("df.pkl", "rb"))

   
    dx = pd.DataFrame((title_abstract_word_frequency), columns=tokens)
    dx = dx.fillna(0)

    
    le = preprocessing.LabelEncoder()
    le.fit(['Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report'])
    arr = le.transform(df['label'])
    
    write_file = open("train_X.pkl", "wb")
    pickle.dump(dx, write_file)

    write_file = open("train_Y.pkl", "wb")
    pickle.dump(arr, write_file)


    #Y = pd.DataFrame(arr)

    #Y = Y.rename(columns={'label': 'labels'})
    
    #dx = pd.concat([dx, Y], axis=1)
    #print(dx)


    #X = pickle.load(open("X.pkl", "rb"))
    
    #X.iloc[0].replace(to_replace=120, value = 130)
    #write_file = open("X.pkl", "wb")
    #pickle.dump(dx, write_file)
    #x = dx.head(1)
   
    

    #df = pickle.load(open("df.pkl", "rb"))
    #print(df)
    

if __name__ == "__main__":
    main()
