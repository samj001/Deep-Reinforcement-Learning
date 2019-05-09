import os,sys,json,math,operator,random

import sqlite3

import xmltodict

import numpy as np

import platform

import logging

import xlwt

import re

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords

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


def plot_analyze(result_data):
	book = xlwt.Workbook()
	sheet1 = book.add_sheet("sheet1")
	sheet2 = book.add_sheet("sheet2")
	
	for data_each_iteration in result_data[0]:
		row = 0
		sheet1.write(row, 1, data_each_iteration)
		row = row+1
	for data_each_iteration in result_data[1]:
		row = 0
		sheet2.write(row, 1, data_each_iteration)
		row = row+1
	book.save(".\\plot.xls")



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


	return topic_list



def get_reward_record(doc_ids, t_id):
	print('doc id', doc_ids)
	#print('topic id', t_id)
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
	
	for _id in doc_ids:

		conn = sqlite3.connect(sqlite_file)

		c = conn.cursor()

		if len(_id) == 7:

			c.execute('SELECT topic_id, docno, rating FROM {pt} WHERE docno={docno} AND topic_id={topicid}'.format(pt = passage_table, docno = '1'+_id[1:], topicid = t_id))

		elif len(_id) == 6:
			c.execute('SELECT topic_id, docno, rating FROM {pt} WHERE docno={docno} AND topic_id={topicid}'.format(pt = passage_table, docno = _id.zfill(7), topicid = t_id))
		
		#c.execute('SELECT topic_id, docno, rating FROM {pt} WHERE passage_id={pi}'.format(pt = passage_table, pi = 4949))
		id_rate = c.fetchall()

		conn.commit()

		conn.close()

		id_rewards.append(id_rate)

	return id_rewards


def get_reward(doc_ids, t_id):
	datas = get_reward_record(doc_ids, t_id)
	print('datas', datas)
	best_reward = 0
	best_id = ""
	rel_n = 0
	rel_id = []
	if not datas == []:
		rel_n = 0
		rel_id = []
		rewards = []
		for data in datas:
			r = 0
			if not data==[] :
				rel_n = rel_n+1
				rel_id.append(data[0][1])
				for s in data:
					r = r+s[2]
			else:
				rel_n = 0
				rel_id = []
			rewards.append(r)
		print('reward: ', rewards)
		best_reward = max(rewards)
		ind_best = rewards.index(max(rewards))
		if not datas[ind_best] == []:
			best_id = datas[ind_best][0][1]
		
		
		
	return best_reward, best_id, rel_n, rel_id



def load_corpus(dictionary):

	''' param dictionary: folder path where corpus xml files are 

		return: 

	'''

	for i,filename in enumerate(os.listdir(dictionary)):

		if filename.endswith('.xml'):

			with open(dictionary+'/'+filename, 'rb') as xml_content:

				x = xmltodict.parse(xml_content.read())

				j = json.dumps(x)

				__es__.index(index='nyt_corpus',doc_type='xml',id=i,body=x)




def get_query(query_tuple):  #[query_str,topic_id,weight]

	query_str = query_tuple[0]

	weight = query_tuple[2]

	terms = []

	for n,word in enumerate(query_str.split()):

		#term = "{'match':{'body':{wo},'boost':{we}\}\}".format(wo = word, we = weight[n])
		#term = {"term" : {"body":word}}
		#term = {'match':{'body':word}}
		term = {"multi_match":{"query":word,"fields":[], "boost":weight[n]}}#,"boost":weight[n]}}
		#t = str(term)
		# if terms=='':
		# 	terms = terms+t
		# else:
		# 	terms = terms+','+t
		# print(terms)
		terms.append(term)
        #query_to_es = "'query':{'bool':{'should':['{t}']\}\}".format(t = terms)
	query_to_es = {'query':{'bool':{'should':terms}}}

	return query_to_es





def es_search(query_tuple):

	''' param query_tuple: [query,topic_id,query_weight] 

		return: five most high ranked docs retrieved by search engine

	'''

	doc_ids=[]



	query = get_query(query_tuple)

	res = __es__.search(index='nyt_corpus',body=query, size=5)
	#res = __es__.search(index="nyt_corpus", body={"query": {"match": {"body": "fox"}}}, size=5)
	# res = __es__.get(index="nyt_corpus", doc_type='xml', id="1")
	# print(res['_source']['nitf']['head']['docdata']['doc-id']['@id-string'])

	passages = []
	#count = 0
	for doc in res['hits']['hits']:
		doc_id = doc['_source']['nitf']['head']['docdata']['doc-id']['@id-string']
		#if len(doc_id) == 6:
			#return 'OOV',''
		
		#count+=1
		try:
			doc_content = doc['_source']['nitf']['body']['body.content']['block']
		
			_id = doc['_id']
			document = ''
			for passage in doc_content:
				p = passage['p']
				for paragraph in p:
					document = document+str(paragraph)
			doc_ids.append(doc_id)
			passages.append(document)
		except:
			doc_content = 'OOV'
			passages.append(doc_content)
	
	return doc_ids, passages, _id



#------------------query update strategy---------
def get_doc_tfidf(doc,docs,query=None):

	stop_words = stopwords.words('english')

	tokens = doc.split()

	counts = {}

	pattern = [',','.','?','!','%','\'','\"','(',')','[',']','{','}']

	for tok in tokens:

		if tok not in stop_words and tok not in pattern:

			if tok not in counts.keys():

				counts[tok] = 1

			else:

				counts[tok] +=1

	meaning = list(counts.keys())

	df_counts = dict(zip(meaning,[0]*len(meaning)))
	
	for d in docs:
		for word in set(d.split()):

			if word in counts.keys():

				df_counts[word] += 1
				

	tfidfs = []

	for m in meaning:

		tfidf = counts[m]*math.log((5.5-df_counts[m])/(df_counts[m]+0.5))

		tfidfs.append(tfidf)

	if query == None:

		max_f = max(tfidfs)

		max_words = []

		for n,w in enumerate(tfidfs):

			if w == max_f:

				max_words.append(meaning[n])

		return random.choice(max_words),meaning,tfidfs

	
	else:

		query_tok = query.split()

		query_tfidf = [0] * len(query_tok)

		for n,q in enumerate(query_tok):

			if q in meaning:

				query_tfidf[n] = counts[q]*math.log((5.5-df_counts[q])/(df_counts[q]+0.5))

		return query_tfidf


#--------------


def find_max(value,li):
    result = []
    for n,i in enumerate(li) :
        if i == value:
            result.append(n)
    return result


def take_actions(action_agent):

	next_rewards = []
	next_query = []

	actions = ['add','remove','weight','stop']

	addRes,q = action_agent.add()

	next_rewards.append(addRes)
	next_query.append(q)

	remove_result,q = action_agent.remove()
	next_rewards.append(remove_result)
	next_query.append(q)

	weight_result,q = action_agent.weight()
	next_rewards.append(weight_result)
	next_query.append(q)

	next_result,q = action_agent.stop()
	next_rewards.append(next_result)
	next_query.append(q)

	max_value = max(next_rewards)
	maxes = find_max(max_value,next_rewards)
    
	best_action = []
	new_query = []

	for i in maxes:
		best_action.append(actions[i])
		new_query.append(next_query[i])

	return best_action, new_query, next_rewards




def learning(agent,topic_list):

	actions = ['add','remove','weight','stop']

	topic_rewards = []
	avg_rewards = []

	for n,topic in enumerate(topic_list):
		print('--topic-- ' +str(n)+' : '+topic[0])

		stop = False

		agent.init_actions(actions)

		context = {'docs_seen':0,'last_rel':0,'rel_seen':0,

			'add':0,'remove':0,'weight':0}

		docs_seen_id = []

		rel_seen_id = []

		query = topic

		topic_reward = 0


		step_n = 1
		while not stop:
			print('------step '+str(step_n)+' -----------')
	
			docs_id,docs_contents,_id = es_search(query)

			for i in docs_id:

				if i not in docs_seen_id:

					docs_seen_id.append(i)

					context['docs_seen']+=1

			current_reward,doc_id,rel_n,rel_id = get_reward(docs_id,"'{tp}'".format(tp = topic[1]))
			
			try:
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

				next_best_action,new_query,next_rewards = take_actions(action_agent)

				print('gold action: ', next_best_action)

				context_features = list(context.values())

				pred = agent.recommend(context_features,actions)
				print('predict action: ', pred)

				reward = next_rewards[actions.index(pred)]


				#method_to_call = getattr(action_agent,pred)

				#reward,_ = method_to_call()
				print('+++executed reward: ', reward)

				agent.update(reward)


				for a in actions:

					if a == pred and a!='stop':

						context[a] += 1


				for m,a in enumerate(next_best_action):

					if pred == a:
						query = new_query[m]
						print("new query: ", query)
			

				topic_reward += reward
				avg_topic_reward = topic_reward/step_n
				print('take: '+str(pred)+', reward: ' + str(reward)," topic reward: ", topic_reward, 'avg_topic_reward: ', avg_topic_reward)

				step_n += 1

				if pred == 'stop':

					stop = True

					topic_rewards.append(topic_reward)
					avg_rewards.append(avg_topic_reward)
			except:
				step_n += 1
				if step_n>100:
					break
				else:
					pass


	return [topic_rewards,avg_rewards]


class ActionsAgent: 


	def __init__(self,state):
        
		self.doc = state['doc']

		self.docs = state['docs']

		self.topic = state['topic'] #topic: [str,id,weight]

		self.query = state['query'] #query: [str,weight]

		self.next_topic = state['next_topic']



	def add(self):

		query = self.query[0]

		weight = self.query[1]

		#mean = sum(weight)/len(weight)

		add_word,_,_ = get_doc_tfidf(self.doc,self.docs)

		query = query + ' ' +add_word
		print('added query: ',query)

		new_weight = weight + [0.5]

		new_query_tuple = [query,self.topic[1],new_weight]

		next_ids,_,_ = es_search(new_query_tuple)

		reward,_,_,_ = get_reward(next_ids,"'{tp}'".format(tp = self.topic[1]))

		return reward,new_query_tuple



	def remove(self):

		query = self.query[0]

		weight = self.query[1]

		query_list = query.split()

		tfidf_list = get_doc_tfidf(self.doc,self.docs,query.lower())
		
		min_index = tfidf_list.index(min(tfidf_list))
	
		new_weight = weight[:min_index]+weight[min_index:]

		min_word = query_list[min_index]

		query_list.remove(min_word)

		removed_query = ''

		for i in query_list:

			removed_query = removed_query + i +' '

		new_query_tuple = (removed_query[:-1],self.topic[1],new_weight)
		print('removed query: ', new_query_tuple)

		next_ids,_,_ = es_search(new_query_tuple)

		reward,_,_,_ = get_reward(next_ids,"'{tp}'".format(tp = self.topic[1]))

		return reward,new_query_tuple




	def weight(self):

		query = self.query[0]

		current_weight = self.query[1]

		doc = self.doc

		docs = self.docs

		_,word_list,tfidf_list = get_doc_tfidf(doc,docs)

		tfidf_dict = dict(zip(word_list,tfidf_list))

		least_rel = sorted(tfidf_dict.items(), key=lambda kv: kv[1])[:20]

		most_rel = sorted(tfidf_dict.items(), key=lambda kv: kv[1])[-20:]

		new_weight = current_weight

		#print('query tokenize: ', word_tokenize(query))

		for i,q in enumerate(query.split()):
				
			#for n,mean in enumerate(word_list):
			for n,(a,b) in enumerate(most_rel):

				if q == a:
					new_weight[i] = current_weight[i]-0.2*((20-n)/20.0)


					#new_weight[i] = round(current_weight[i]+math.exp(tfidf_list[n]),3)
	

			for n,(a,b) in enumerate(least_rel):

				if q == a:
					#print('hit min')

					new_weight[i] = current_weight[i]-0.2*((20-n)/20.0)

		print('new_weight: ', new_weight)

		new_query_tuple = [query,self.topic[1],new_weight]

		next_ids,_,_ = es_search(new_query_tuple)

		reward,_,_,_ = get_reward(next_ids,"'{tp}'".format(tp = self.topic[1]))

		return reward,new_query_tuple
        
	def stop(self):
		next_query_tuple = self.next_topic
		print(next_query_tuple[0])
		next_ids,_,_ = es_search(next_query_tuple)
		
		reward,_,_,_ = get_reward(next_ids,"'{tp}'".format(tp = next_query_tuple[1]))
		return reward,next_query_tuple





class LinUCB:

	def __init__ (self,hparams):



		self.alpha = hparams['explore_rate']

		self.d = hparams['context_dim']

		self.r1 = 1.0

		self.r0 = -16.0

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

			r = math.sqrt(reward)

		self.Aa[self.a_max] += np.dot(self.x,self.xT)

		self.ba[self.a_max] += r * self.x

		self.AaI[self.a_max] = np.linalg.inv(self.Aa[self.a_max])

		self.theta[self.a_max] = np.dot(self.AaI[self.a_max],self.ba[self.a_max])





	def recommend(self,context_features,actions):

		xaT = np.array([context_features])

		xa = np.transpose(xaT)

		AaI_array = []
		for action in actions:
			AaI_array.append(self.AaI[action])
		AaI_tmp = np.array(AaI_array)
		
		theta_array = []
		for action in actions:
			theta_array.append(self.theta[action])
		theta_tmp = np.array(theta_array)

		action_max = actions[np.argmax(np.dot(xaT,theta_tmp)+ np.multiply(self.alpha , np.sqrt(np.dot(np.dot(xaT,AaI_tmp),xa))))]
		
		self.x = xa

		self.xT = xaT

		self.a_max = action_max

		return self.a_max


def main():


	hparams = {'context_dim':6,

		'explore_rate':0.25,

		'learning_rate':0.01,
	'iterations':50}


	topics = getTopics()

	es = Elasticsearch(['https://user:secret@localhost:443'])
	es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


	# folder_path = 'nyt_corpus'

	# if platform.system() == 'Darwin':

	#     load_corpus(folder_path)

	# else:
	#     if platform.system() == 'Windows':

	#         load_corpus('.\\trec-dd-jig\\all_doc')

	agent = LinUCB(hparams)
	result_list = []

	for i in range(hparams['iterations']):
		print('=================iteration '+str(i)+' ============================')
		results = learning(agent,topics)
		result_list.append(results)
		plot_analyze(result_list)


    #plot_analyze(result_list)

	





if __name__ == '__main__':

	#parser = argparse.ArgumentParser()

	#parser.add_argument("folder_path", type=str,default=)
    logging.basicConfig(level=logging.ERROR)
    main()