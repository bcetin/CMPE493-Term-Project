from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
import string
import pickle
import os.path
features = ""

#preprossing
def document_cleaner(text, ps):
    global stopwords
    global features
    text = text.lower()
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = text.translate(table)
    doc_for_tf_idf= []
    text = text.split()
    for i,word in enumerate(text):
        stemmed_word = ps.stem(word)
        doc_for_tf_idf.append(stemmed_word)
        '''
        if stemmed_word in features:
            doc_for_tf_idf.append(stemmed_word)
        if i+1==len(text):
            continue
        stemmed_next = ps.stem(text[i+1])
        pair = stemmed_word + ' ' + stemmed_next
        if pair in features:
            doc_for_tf_idf.append(pair)
        '''
    doc_for_tf_idf = ' '.join(doc_for_tf_idf)
            
    return doc_for_tf_idf



def main():
    
    #get title and abstract of test set and write to file
    if not os.path.exists("output/test_title_abstracts.pkl") or True:
        global features
        ps = PorterStemmer()
        path = "dataset/BC7-LitCovid-Dev.csv"
        test_df = pd.read_csv(path, sep=',')
        features = pickle.load(open("output/features.pkl", "rb"))
        print(features)
        test_df['abstract'] = test_df['abstract'].map(lambda document : document_cleaner(document, ps))
        test_df['title'] = test_df['title'].map(lambda document : document_cleaner(document, ps))
        test_title_abstracts = test_df['title'] + " " + test_df['abstract']
        
        write_file = open("output/test_title_abstracts.pkl", "wb")
        pickle.dump(test_title_abstracts, write_file)
    else:
        test_title_abstracts = pickle.load(open("output/test_title_abstracts.pkl", "rb")).tolist()
    #load test_title_abstract and test the model
    path = "dataset/BC7-LitCovid-Dev.csv"
    test_df = pd.read_csv(path, sep=',')
    pmid_ids = test_df['pmid']
    del test_df
    features = pickle.load(open("output/features.pkl", "rb"))
    tfidf = TfidfVectorizer(vocabulary=features,ngram_range = (1,2))
    test_X = tfidf.fit_transform(test_title_abstracts).toarray()
    print(tfidf.get_feature_names_out())
    model = pickle.load(open("output/naive_bayes_model.pkl", "rb"))
    Y = model.predict_proba(test_X)
    #Y = nb_model.predict(test_X)
    '''
    print(Y[5])
    for i in range(len(Y)):
        if Y[i] == 5:
            print(i)
    '''
    result = []
    
    #append the predictions
    for index, element in enumerate(Y):
        test_Y = []
        test_Y.append(pmid_ids[index])
        test_Y.append(element[6])
        test_Y.append(element[1])
        test_Y.append(element[4])
        test_Y.append(element[3])
        test_Y.append(element[5])
        test_Y.append(element[2])
        test_Y.append(element[0])
        result.append(test_Y)
    
    columns = ['PMID', 'Treatment', 'Diagnosis', 'Prevention', 'Mechanism', 'Transmission', 'Epidemic Forecasting', 'Case Report']
    dx = pd.DataFrame(result, columns = columns)
    dx.to_csv("output/prediction.csv", index=False)

    
    
    














if __name__ == "__main__":
    main()
