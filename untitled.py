
import os,sys,json,math,operator
import sqlite3
import tensorflow as tf
import xmltodict
import numpy as np
import platform
from nltk import word_tokenize
from collections import Counter, OrderedDict
from elasticsearch import Elasticsearch


def getTopics():
    sqlite_file = ".\\trec-dd-jig\\jig\\truth.db" #address of the topic database
    topic_table = "topic"
    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    #c.execute('SELECT * FROM {tt}'.format(tt = topic_table))
    #all_rows = c.fetchall()
    c.execute('SELECT topic_name FROM {tt}'.format(tt = topic_table))
    all_topics = c.fetchall()
    print(all_topics)
    conn.commit()
    conn.close()
    return all_topics


def get_data():

	return query_list, docs_dictionary_path


def load_corpus(dictionary,es):
	''' param dictionary: folder path where corpus xml files are 
		return: 
	'''
	for i,filename in enumerate(os.listdir(dictionary)):
		if filename.endswith('.xml'):
			with open(dictionary+'/'+filename) as xml_content:
				x = xmltodict.parse(xml_content.read())
				j = json.dumps(x)
				es.index(index='nyt_corpus',doc_type=filename,id=i,body=x)


def get_query(query_str,weight = None):
	
	if weight == None:
		weight = [1.0] * len(query.split())

	terms = ''

	for n,word in enumerate(query_str.split()):
		term = {'match':{'docdata':word,'boost':weight[n]}}
		terms = terms + term+ ','
	query = '"query":{"bool":{"should":[terms]}}'

	return query


def es_search(es,query_str,weight=None):
	''' param es: search engine
			  query: one topic to search
		return: five most high ranked docs retrieved by search engine
	'''
	doc_ids=[]

	if weight == None:
		weight = [1.0] * len(query.split())
	
	query = get_query(query_str,weight)

	res = es.search(index='nyt_corpus',body={"query":{'match':{"docdata":query}}},size=5)
	for doc in res['hits']:
		doc_id = doc.meta.doc_type
		doc_ids.append(doc_id)

	return doc_ids


def doc_tokenize_freq(docs):
	''' 
	params docs:a list of doc, each doc is a string 
	return frequency of tokens in docs (unormolized)
	'''

	tokens = word_tokenize[doc]
	freq = {}
	for tok in tokens:
		if tok in freq.keys():
			freq[tok] += 1
		else:
			freq[tok] = 1
	return freq
'''
	tokens = []
	for doc in docs:
		tokens += word_tokenize[doc]
	freq = {}
	for tok in tokens:
		if tok in freq.keys():
			freq[tok] += 1
		else:
			freq[tok] = 1
	return freq
'''

def get_doc_tfidf(doc,docs):
	doc_freq = docs_tokenize_freq([doc])
	#docs_freq = docs_tokenize_freq(docs)

	tfidf_list = []
	max_f = 0.0

	for q in doc_freq.keys():
		doc_f = 0.00001
		for doc in docs:
			if q in word_tokenize(doc):
				doc_f += 1
		
		q_f = doc_freq[q]/math.log(doc_f)
		tfidf_list.append(q_f)
		if q_f > max_f:
			max_f = q_f
			max_word = q
	return max_word,tfidf_list


def get_query_tfidf(doc,docs,query):

	_, doc_tfidf = get_doc_tfidf(doc,docs)

	query_tfidf = []
	for q in query.split():
		q_f = 0.0
		for n,key in enumerate(word_tokenize(doc)):
			
			if word_tokenize(q) == key:
				q_f = doc_tfidf[n]
				query_tfidf.append(q_f)

		return query_tfidf


class ActionsAgent():
	
		

	def add(self,selected_doc,docs,query):
		add_word,_ = get_doc_tfidf(selected_doc,docs)
		query = query + ' ' +add_word
		return query


	def remove(self,doc,docs,query):
		query_list = query.split()
		tfidf_list = get_query_tfidf(doc,docs,query)
		min_index = tfidf_list.index(min(tfidf_list))
		min_word = query_list[min_index]
		query_list.remove(min_word)

		removed_query = ''
		for i in query_list:
			removed_query = removed_query + i +' '

		return removed_query[:-1]


	def weight(self,doc,docs,query,current_weight=None):
		_,tfidf_list = get_doc_tfidf(doc,docs)
		tfidf_dict = dict(zip(doc,tfidf_list))

		least_rel = sorted(tfidf_dict.items(), key=lambda kv: kv[1])[:20]
		most_rel = sorted(tfidf_dict.items(), key=lambda kv: kv[1])[-20:]

		if current_weight == None:
			current_weight = [1.0]*len(word_tokenize(query))

		new_weight = current_weight
		for i,q in enumerate(word_tokenize(query)):
			for n,(a,b) in enumerate(most_rel):
				if q == a:
					new_weight[i] = current_weight+0.2*((20-n)/20.0)
	
			for n,(a,b) in enumerate(least_rel):
				if q == a:
					new_weight[i] = current_weight-0.2*((20-n)/20.0)

		return query,weight


	def stop(self,doc,query):
		return

'''
def pick_action(docs,query,es):
	try_action = ActionsAgent()
	scores = []
	for doc in docs:
'''                            

def main():

	hparams = {'context_dim':6,
		'explore_rate':0.25,
		'learning_rate':0.01}

	#topic = getTopics()
	#print(topic)

	#es = Elasticsearch(['https://user:secret@localhost:443'])
	#es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
	load_corpus('nyt_corpus/sample',es)
	for index in es.indices.get('*'):
		print(index)


if __name__ == '__main__':
	#parser = argparse.ArgumentParser()
	#parser.add_argument("", type=str,default=)

	main()