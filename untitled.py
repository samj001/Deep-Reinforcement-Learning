
import os,sys,json,math,operator
import sqlite3
import tensorflow as tf
import xmltodict
import numpy as np
import platform
from nltk import word_tokenize
from collections import Counter, OrderedDict
from elasticsearch import Elasticsearch

__es__ = Elasticsearch([{'host': 'localhost', 'port': 9200}])

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

    topic_tuples = []
    for topic in all_topics:
    	length = len(topic.split())
    	weight = [1.0] * length
    	topic_tuples.append(topic,weight)

    return topic_tuples


def load_docs():

	return docs


def get_doc(docs_id):
	#list of one or more doc id, return list [doc1_content,doc2_content,...]
	return docs

def get_reward(doc_ids,query):

	return best_reward


def load_corpus(dictionary):
	''' param dictionary: folder path where corpus xml files are 
		return: 
	'''
	es = __es__
	for i,filename in enumerate(os.listdir(dictionary)):
		if filename.endswith('.xml'):
			with open(dictionary+'/'+filename) as xml_content:
				x = xmltodict.parse(xml_content.read())
				j = json.dumps(x)
				es.index(index='nyt_corpus',doc_type=filename,id=i,body=x)


def get_query(query_tuple):
	query_str = query_tuple[0]
	weight = query_tuple[1]

	#if weight == None:
		#weight = [1.0] * len(query.split())

	terms = ''

	for n,word in enumerate(query_str.split()):
		term = {'match':{'docdata':word,'boost':weight[n]}}
		terms = terms + term+ ','
	query_to_es = ""query":{"bool":{"should":[terms]}}"

	return query_to_es,weight


def es_search(query_tuple):
	''' param es: search engine
			  query: one topic to search
		return: five most high ranked docs retrieved by search engine
	'''
	es = __es__
	doc_ids=[]

	query,_ = get_query(query_tuple)

	res = es.search(index='nyt_corpus',body={"query":{'match':{"docdata":query}}},size=5)
	for doc in res['hits']:
		doc_id = doc.meta.doc_type
		doc_ids.append(doc_id)

	return doc_ids


def doc_tokenize_freq(doc_id):
	''' 
	params docs: doc_id
	return frequency of tokens in docs (unormolized)
	'''

	doc = get_doc(doc_id)
	tokens = word_tokenize[doc]
	freq = {}
	for tok in tokens:
		if tok in freq.keys():
			freq[tok] += 1
		else:
			freq[tok] = 1
	return freq


def get_doc_tfidf(doc_id,docs_id):
	doc_freq = doc_tokenize_freq(doc_id)
	#docs_freq = docs_tokenize_freq(docs)

	docs = get_doc(docs_id)

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


def get_query_tfidf(doc,docs,query_tuple):

	_, doc_tfidf = get_doc_tfidf(doc,docs)
	query = query_tuple[0]

	query_tfidf = []
	for q in query.split():
		q_f = 0.0
		for n,key in enumerate(word_tokenize(doc)):
			
			if word_tokenize(q) == key:
				q_f = doc_tfidf[n]
				query_tfidf.append(q_f)

		return query_tfidf


def get_context_features():
	return context_features



def take_actions(action_agent,state):
	next_rewards = []
	actions = ['add','remove','weight','stop']

	next_rewards.append(action_agent(state).add())
	next_rewards.append(action_agent(state).remove())
	next_rewards.append(action_agent(state).weight())
	next_rewards.append(action_agent(state).stop())

	best_action = actions[next_rewards.index(max(next_rewards))]

	return best_action



def learning(agent,query_list,action_agent,es):
	action = ['add','remove','weight','stop']

	stop = False

	for query in query_list():

		agent.init_actions(actions)

		while not stop:
			
			docs_id = es_search(query,es)
			reward = take_actions(action_agent,state)

			context_features = get_context_features()
			pred = agent.recommend(timestamp,context_features,actions)
			if pred == reward:
				agent.update(reward)


			if pred == 'stop':
				stop = True
				break


class ActionsAgent(): 

	def __init__(self,state):
		self.doc = self.state['doc']
		self.docs = self.state['docs']
		self.query = self.state['query'] #query tuple
		self.next_query = self.state['next_query']


	def add(self):
		query = self.query[0]
		add_word,_ = get_doc_tfidf(self.doc,self.docs)
		query = query + ' ' +add_word
		new_query_tuple = (query,self.query[1])
		next_ids = es_search(new_query_tuple)
		reward = get_reward(next_ids,self.query[0])
		return reward


	def remove(self):
		query = self.query[0]
		query_list = query.split()
		tfidf_list = get_query_tfidf(self.doc,self.docs,self.query)
		min_index = tfidf_list.index(min(tfidf_list))
		min_word = query_list[min_index]
		query_list.remove(min_word)

		removed_query = ''
		for i in query_list:
			removed_query = removed_query + i +' '

		new_query_tuple = (removed_query[:-1],self.query[1])
		next_ids = es_search(new_query_tuple)

		reward = get_reward(next_ids,self.query[0])

		return reward


	def weight(self):

		query = self.query[0]
		current_weight = self.query[1]

		doc = self.doc
		docs = self.docs

		_,tfidf_list = get_doc_tfidf(doc,docs)
		tfidf_dict = dict(zip(doc,tfidf_list))

		least_rel = sorted(tfidf_dict.items(), key=lambda kv: kv[1])[:20]
		most_rel = sorted(tfidf_dict.items(), key=lambda kv: kv[1])[-20:]

		new_weight = current_weight
		for i,q in enumerate(word_tokenize(query)):
			for n,(a,b) in enumerate(most_rel):
				if q == a:
					new_weight[i] = current_weight+0.2*((20-n)/20.0)
	
			for n,(a,b) in enumerate(least_rel):
				if q == a:
					new_weight[i] = current_weight-0.2*((20-n)/20.0)

		next_ids = es_search((query,new_weight))
		reward = get_reward(next_ids,self.query[0])

		return reward


	def stop(self,query_tuple,next_query_tuple):
		next_ids = es.search(next_query_tuple)
		reward = get_reward(next_ids,next_query_tuple[0])

		return reward


class LinUCB:
	def __init__ (self,hparams,user):

		self.alpha = hparams['explore_rate']
		self.d = hparams['context_dim']
		self.r1 = 0.6
		self.r0 = -16
		self.Aa = {}
		self.AaI = {}
		self.ba = {}
		self.a_max = 0
		self.theta = {}
		self.x = None
		self.xT = None


	def init_actions(self,actions):

		for action in actions:
			self.Aa[action] = np.identity(self.d)
			self.ba[action] = np.zeros((self.d,1))

			self.AaI[action] = np.identity(self.d)
			self.theta[action] = np.zeros((self.d,1))


	def update(self,reward):

		r = reward

		self.Aa[self.a_max] += np.dot(self.x,self.xT)
		self.ba[self.a_max] += r * self.x
		self.AaI[self.a_max] = np.linalg.inv(self.Aa[a_max])
		self.theta[self.a_max] = np.dot(self.AaI[self.a_max],self.ba[self.a_max])


	def recommend(self,timestamp,context_features,actions):
		xaT = np.array([context_features])
		xa = np.transport(xaT)

		AaI_tmp = np.array([self.AaI[action]] for action in actions)
		theta_tmp = np.array([self.thetha[action]] for action in actions)
		action_max = actions[np.argmax(np.dot(xaT,theta_tmp)+ self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))]

		self.x = xa
		self.xT = xaT

		self.a_max = action_max

		return self.a_max


                      

def main():

	hparams = {'context_dim':6,
		'explore_rate':0.25,
		'learning_rate':0.01}

	#topic = getTopics()
	#print(topic)

	#es = Elasticsearch(['https://user:secret@localhost:443'])
	
	load_corpus('nyt_corpus/sample')
	for index in es.indices.get('*'):
		print(index)


if __name__ == '__main__':
	#parser = argparse.ArgumentParser()
	#parser.add_argument("", type=str,default=)

	main()
