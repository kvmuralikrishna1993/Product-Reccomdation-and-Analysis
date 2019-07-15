import pandas as pd
import numpy as np
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
import pickle

import ssl
try:
	_create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
	pass
else:
	ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import models
from gensim.models.coherencemodel import CoherenceModel

#Representing wordcloud.
from wordcloud import WordCloud, STOPWORDS

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython import get_ipython

#Ignoring warnings.
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore',category = DeprecationWarning)
positive_sen = []
negative_sen = []


with open('sentiment_train.json', 'r') as fp:
  cl = NaiveBayesClassifier(fp, format="json")
print("Sucessfully built the classifier ....")

print("calculating the accuracy of classifier...")
with open('sentiment_test.json', 'r') as test:
  print("classifier accuracy:")
  print(cl.accuracy(test,format="json"))
#73%

sentiment_classifier = open("naivebayes.pickle","wb")
print("Generating pickle file....")

print("Dumping pickle....")
pickle.dump(cl, sentiment_classifier)

sentiment_classifier.close()
print("Pickle file created.....")


# with open("naivebayes.pickle", "rb") as classifier_f:
# 	cl = pickle.load(classifier_f)
# classifier_f.close()
# print("output for sample doc:")
# print(cl.classify("i think he doesnt like me, but there is a chance that he can love me"))

#classifying the text
def classify(review):
	blob = TextBlob(review, classifier=cl)
	blob.lower()
	blob.correct()
	for sentence in blob.sentences:
		if sentence.classify() == "neg" and len(str(sentence)) > 3:
			negative_sen.append(str(sentence))
		elif sentence.classify() == "pos" and len(str(sentence)) > 3:
			positive_sen.append(str(sentence))


print("loading reviews")
data = pd.read_csv('topic.csv', error_bad_lines=False);
data_text = data[['Reviews']]
data_text['index'] = data_text.index
documents = data_text
print("Number of documents: "+str(len(documents))) #-001


print("classifying reviews...")
documents['Reviews'].map(classify)
print("reviews classified....")

print()
print("No of positive sentences: "+str(len(positive_sen)))
print(str(positive_sen[0:10]))
print()
print("No of negative sentences: "+str(len(negative_sen)))
print(negative_sen[0:10])

def get_classfied_files(filename):
    unseen_document = ""
    for line in open(filename, 'r'):
        unseen_document = unseen_document+""+line
    return unseen_document

pos_str = (" ").join(positive_sen)
file1 = open("positive.txt","w")
file1.writelines(pos_str) 
file1.close()

neg_str = (" ").join(negative_sen)
file2 = open("negative.txt","w")
file2.writelines(neg_str) 
file2.close()

pos_str = get_classfied_files("positive.txt")
neg_str = get_classfied_files("negative.txt")

def preprocess(text):
	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return result

def lemmatize_stemming(text):
	stemmer = PorterStemmer()
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='a'))


processed_docs = documents['Reviews'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
print(len(dictionary)) #-- 01

del_list = ["galaxi","iphon","samsung","soni","xperia","month","love","time","problem","like","good"]
del_ids = [k for k,w in dictionary.items() if w in del_list]

dictionary.filter_tokens(bad_ids=del_ids)
# new_dict = copy.deepcopy(dictionary)
dictionary.filter_extremes(no_below=70, no_above=0.2, keep_n=1700)

#printing final dictionary
for k, v in dictionary.iteritems():
	print(k, v)
print()
# words = ' '
# stopwords =set(STOPWORDS) 
# stopwords = []
# for k,v in dictionary.iteritems():
# 	words = words + v + ' '
# wordcloud = WordCloud(width = 800, height = 800,background_color ='white',stopwords = stopwords,min_font_size = 10).generate(words)
# 	#plot the WordCloud image                        
# plt.figure(figsize = (8, 8), facecolor = None) 
# plt.imshow(wordcloud) 
# plt.axis("off") 
# plt.tight_layout(pad = 0)  
# plt.show()

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=2, workers=3)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} Words: {}'.format(idx+1, topic))
print()


coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score_lda: ', coherence_lda)
print()

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=4, id2word=dictionary, passes=2, workers=3)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx+1, topic))
print()


coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score_tdif: ', coherence_lda)
print()

vis = pyLDAvis.gensim.prepare(lda_model_tfidf,corpus_tfidf, dictionary)
pyLDAvis.save_html(vis, 'LDA_Vis_tfidf.html')

#for the positve sentences
bow_vector = dictionary.doc2bow(preprocess(pos_str))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    # print("Score: {}\t Topic: {}".format(round(score*100), lda_model_tfidf.print_topic(index+1, 4)))
    print("Topic: {}\t Score: {}".format(dictionary.get(lda_model_tfidf.get_topic_terms(index, 5)[0][0]), round(score*100)))
print()



#for the negative sentences
bow_vector = dictionary.doc2bow(preprocess(neg_str))
for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1]):
    # print("Score: {}\t Topic: {}".format(round(score*100), lda_model_tfidf.print_topic(index, 4)))
    print("Topic: {}\t Score: {}".format(dictionary.get(lda_model_tfidf.get_topic_terms(index, 5)[0][0]), round(score*100)))
print()


