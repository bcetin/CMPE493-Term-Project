from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import string
import pickle





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



def naive_bayes(X, Y):
    multinomial_bayes = MultinomialNB()
    bayes_model = multinomial_bayes.fit(X, Y)
    #bayes_y = multinomial_bayes.predict(X_test)
    return bayes_model


def document_cleaner(text, ps):
    global stopwords
    text = text.lower()
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = text.translate(table)
    doc_for_tf_idf= []
    text = text.split()
    for word in text:
        stemmed_word = ps.stem(word)
        if stemmed_word not in stopwords and word not in stopwords and not word.isnumeric():
            doc_for_tf_idf.append(stemmed_word)
    doc_for_tf_idf = ' '.join(doc_for_tf_idf)
            
    return doc_for_tf_idf


def main():

	#concatenating title and abstract and writing to a file
    '''
    ps = PorterStemmer()
    df = pickle.load(open("df.pkl", "rb"))

    df['abstract'] = df['abstract'].map(lambda document : document_cleaner(document, ps))
    df['title'] = df['title'].map(lambda document : document_cleaner(document, ps))
    title_abstracts = df['title'] + " " + df['abstract']
    
    write_file = open("title_abstracts.pkl", "wb")
    pickle.dump(title_abstracts, write_file)
    

    '''
	
	
    #get title_abstracts 
    title_abstracts = pickle.load(open("output/title_abstracts.pkl", "rb")).tolist()
    train_Y = pickle.load(open("output/train_Y.pkl", "rb"))
    
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df = 5, norm='l2')
    X = tfidf.fit_transform(title_abstracts).toarray()
    
	#feature selection
    features = tfidf.get_feature_names_out()
    #select best 100 feature
    selector = SelectKBest(chi2, k=100)
    train_X = selector.fit_transform(X, train_Y)
    print(selector.get_feature_names_out(features))


	#train naive bayes model
    model = naive_bayes(train_X, train_Y)
    write_file = open("output/naive_bayes_model.pkl", "wb")
    pickle.dump(model, write_file) 
   



    
    
    












if __name__ == "__main__":
    main()
