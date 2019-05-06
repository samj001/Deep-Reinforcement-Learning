import os,sys,json,math,operator,re

import sqlite3

import tensorflow as tf

import xmltodict

import numpy as np

import platform

import logging

from nltk import word_tokenize

from collections import Counter, OrderedDict

from elasticsearch import Elasticsearch


def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Yay Connect')
    else:
        print('Awww it could not connect!')
    return _es


__es__ = Elasticsearch([{'host': 'localhost', 'port': 9200}])



def getTopics():

	windows_database = ".\\trec-dd-jig\\jig\\truth.db"

	os_database = "./trec-dd-jig/jig/truth.db"

	if platform.system() == "Windows":
		sqlite_file = windows_database
	else:
		sqlite_file = os_database

	topic_table = "topic"

	conn = sqlite3.connect(sqlite_file)

	c = conn.cursor()

	#c.execute('SELECT * FROM {tt}'.format(tt = topic_table))

	#all_rows = c.fetchall()

	c.execute('SELECT topic_name, topic_id FROM {tt}'.format(tt = topic_table))

	all_topics = c.fetchall()

	conn.commit()

	conn.close()



	topic_list = []

	for topic in all_topics:
		temp_list = list(topic)

		length = len(topic[0].split())

		weight = [1.0] * length

		temp_list.append(weight)

		topic_list.append(temp_list)

	print(topic_list)

	return topic_list



def get_reward_record(doc_ids, t_id):
	windows_database = ".\\trec-dd-jig\\jig\\truth.db"
	
	os_database = "./trec-dd-jig/jig/truth.db"

	if platform.system() == "Windows":
		sqlite_file = windows_database
	else:
		sqlite_file = os_database

	passage_table = "passage"

	# conn = sqlite3.connect(sqlite_file)

	# c = conn.cursor()

	id_rewards = []
	
	for id in doc_ids:

		conn = sqlite3.connect(sqlite_file)

		c = conn.cursor()

		c.execute('SELECT topic_id, docno, rating FROM {pt} WHERE docno={docno} AND topic_id={topicid}'.format(pt = passage_table, docno = id, topicid = t_id))
		
		id_rate = c.fetchall()

		conn.commit()

		conn.close()

		id_rewards.append(id_rate)

	

	return id_rewards


def get_reward(doc_ids, t_id):
    datas = get_reward_record(doc_ids, t_id)
    best_reward = datas[0][0][2]
    best_id = datas[0][0][1]
    rel_n = 0
    rel_id = []
    for data in datas:
        if data[0][2]>0:
            rel_n = rel_n+1
            rel_id.append(data[0][2])
    return best_reward, best_id, rel_n, rel_id



def load_corpus(dictionary):

	''' param dictionary: folder path where corpus xml files are 

	'''

	for i,filename in enumerate(os.listdir(dictionary)):

		if filename.endswith('.xml'):

			with open(dictionary+'/'+filename) as xml_content:

				x = xmltodict.parse(xml_content.read())

				__es__.index(index='nyt_corpus',doc_type=filename,id=i,body=x)





def get_query(query_tuple):  #[query_str,topic_id,weight]

	query_str = query_tuple[0]

	weight = query_tuple[2]



	#if weight == None:

		#weight = [1.0] * len(query.split())



	terms = ''



	for n,word in enumerate(query_str.split()):

		term = {'match':{'body':word,'boost':weight[n]}}

		terms = terms + term+ ','

	query_to_es = "'query':{'bool':{'should':["+terms"]}}"



	return query_to_es


def get_content(jsonfile):
	pattern = "^<body.content>.+"
	x = re.findall(pattern,jsonfile)
	x = re.sub('[\{|\}]',' ',x)

	return x

def es_search(query_tuple):

	''' param query_tuple: [query,topic_id,query_weight] 

		return: five most high ranked docs retrieved by search engine

	'''

	doc_ids=[]
	docs_content = []


	query = get_query(query_tuple)


	res = __es__.search(index='nyt_corpus',body={query},size=5)

	for doc in res['hits']:

		doc_id = doc.meta.doc_type

		doc_ids.append(doc_id)
		
		docs_content.append(get_content(doc))



	return doc_ids,docs_content





#------------------query update strategy---------



def doc_tokenize_freq(doc_content):
	''' 
	params docs: doc_content, string
	return frequency of tokens in docs (unormolized)
	'''

	tokens = word_tokenize[doc_content]
	freq = {}
	for tok in tokens:
		if tok in freq.keys():
			freq[tok] += 1
		else:
			freq[tok] = 1
	return freq


def get_doc_tfidf(doc_content,docs_contents):
	''' param doc_content:string
			  docs_contents:a list of strings
	'''

	doc_freq = doc_tokenize_freq(doc_content)

	tfidf_list = []
	max_f = 0.0

	for q in doc_freq.keys():
		doc_f = 0.00001
		for doc in docs_contents:
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





#--------------





def take_actions(action_agent,state):

	next_rewards = []

	actions = ['add','remove','weight','stop']



	next_rewards.append(action_agent(state).add()[0])

	next_rewards.append(action_agent(state).remove()[0])

	next_rewards.append(action_agent(state).weight()[0])

	next_rewards.append(action_agent(state).stop())



	best_action = actions[next_rewards.index(max(next_rewards))]



	return best_action





def learning(agent,topic_list):
	actions = ['add','remove','weight','stop']

	topic_rewards = []
	for n,topic in enumerate(topic_list):
		stop = False
		agent.init_actions(actions)
		context = {'docs_seen':0,'last_rel':0,'rel_seen':0,
			'add':0,'remove':0,'weight':0}
		docs_seen_id = []
		rel_seen_id = []

		query = topic
		topic_reward = 0

		while not stop:
			
			docs_id,docs_contents = es_search(query)
			for i in docs_id:
				if i not in docs_seen_id:
					docs_seen_id.append(i)
					context['docs_seen']+=1

			current_reward,doc_id,rel_n,rel_id = get_reward(docs_id,topic[1])
			doc_content = docs_contents[docs_id.index(doc_id)]
			
			context['last_rel'] = rel_n
			if rel_id is not None and rel_n !=0 :
				for i in rel_id:
					if i not in rel_seen_id:
						rel_seen_id.append(i)
						context['rel_seen'] += 1

		# if not the last topic, compare a[stop]-reward with starting a new topic
		# otherwise, compare with the first topic
			if n != len(topic_list): 
				next_topic = topic_list[n+1]
			else:
				next_topic = topic_list[0]

			state = {'doc':doc_content,
				'docs':docs_contents,
				'topic':topic,
				'query':[query[0],query[2]],
				'next_topic':next_topic}

			action_agent = ActionsAgent(state)

			next_best_action,new_query = take_actions(action_agent,state)

			context_features = list(context.values())

			pred = agent.recommend(context_features,actions)
			method_to_call = getattr(action_agent,pred)
			reward = action_agent.method_to_call(state)
			agent.update(reward)

			for a in actions:
				if a == pred:
					context[a] += 1

			if pred == next_best_action:
				query = new_query

			topic_reward += reward

			if pred == 'stop':
				stop = True
				topic_rewards.append(topic_reward)

	return topic_rewards





class ActionsAgent(): 



	def __init__(self,state):
        
		self.doc = state['doc']

		self.docs = state['docs']

		self.topic = state['topic'] #topic: [str,id,weight]

		self.query = state['query'] #query: [str,weight]

		self.next_topic = state['next_topic']





	def add(self):

		query = self.query[0]

		weight = self.topic[1]

		mean = sum(weight)/float(len(weight))



		add_word,_ = get_doc_tfidf(self.doc,self.docs)

		query = query + ' ' +add_word



		new_weight = weight.append(mean)

		new_query_tuple = [query,self.topic[1],new_weight]



		next_ids,_ = es_search(new_query_tuple)

		reward,_,_,_ = get_reward(next_ids,self.topic[1])

		return reward,new_query_tuple





	def remove(self):

		query = self.query[0]

		weight = self.query[1]



		query_list = query.split()

		tfidf_list = get_query_tfidf(self.doc,self.docs,query)

		

		min_index = tfidf_list.index(min(tfidf_list))

		new_weight = weight.remove(min_index)

		min_word = query_list[min_index]

		query_list.remove(min_word)



		removed_query = ''

		for i in query_list:

			removed_query = removed_query + i +' '



		new_query_tuple = (removed_query[:-1],self.topic[1],new_weight)

		next_ids,_ = es_search(new_query_tuple)



		reward,_,_,_ = get_reward(next_ids,self.topic[1])



		return reward,new_query_tuple





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



		new_query_tuple = [query,self.topic[1],new_weight]

		next_ids,_ = es_search(new_query_tuple)

		reward,_,_,_ = get_reward(next_ids,self.topic[1])



		return reward,new_query_tuple
        
	def stop(self,query_tuple,next_query_tuple):
		next_ids = __es__.search(next_query_tuple)
		reward,_,_,_ = get_reward(next_ids,next_query_tuple[1])
		return reward





class LinUCB:

	def __init__ (self,hparams):



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



		if reward == 0:

			r = self.r0



		else:

			r = reward * self.r1



		self.Aa[self.a_max] += np.dot(self.x,self.xT)

		self.ba[self.a_max] += r * self.x

		self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])

		self.theta[self.a_max] = np.dot(self.AaI[self.a_max],self.ba[self.a_max])





	def recommend(self,context_features,actions):

		xaT = np.array([context_features])

		xa = np.transpose(xaT)



		AaI_tmp = np.array([self.AaI[action]] for action in actions)

		theta_tmp = np.array([self.theta[action]] for action in actions)

		action_max = actions[np.argmax(np.dot(xaT,theta_tmp)+ self.alpha * np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa)))]



		self.x = xa

		self.xT = xaT



		self.a_max = action_max



		return self.a_max



					  



def main():


    hparams = {'context_dim':6,

        'explore_rate':0.25,

        'learning_rate':0.01,'iterations':50}


    topics = getTopics()
    
    doc_ids = ["'0357745'", "'0828898'","'0369273'","'0326231'","'0355863'"]

    topicid = "'dd17-7'"

    print(get_reward(doc_ids, topicid))

    es = Elasticsearch(['https://user:secret@localhost:443'])
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
	

    folder_path = 'nyt_corpus'

    if platform.system() == 'Darwin':

        load_corpus(folder_path)

    else:
        if platform.system() == 'Windows':

            load_corpus('.\\trec-dd-jig\\sample_run')

    agent = LinUCB(hparams)
    iterations_results = []
    for i in range(hparams['iterations']):

        results = learning(agent,topics)
	iterations_results.append(results)





if __name__ == '__main__':

	#parser = argparse.ArgumentParser()

	#parser.add_argument("folder_path", type=str,default=)
    logging.basicConfig(level=logging.ERROR)
    main()
