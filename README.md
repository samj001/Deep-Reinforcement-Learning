# COSC689-finalproject
# Dynamic search system via deep reinforcement learning on NYT corpus

Abstract  
-------

In this paper we present a contextual bandit ap- proach to dynamic search tasks. It is imple- mented on the New York Times Corpus and evaluated with CUBE tests and precision met- rics. Our approach relies on a search engine and a user agent, creating an interacting pro- cess of the information retrieval task. The re- inforcement learning method, which is based on each time of interactions, allows the system to optimize the searching path to find the most relevant information from the corpus.  
	

1 Introduction 
-------

Dynamic search is a kind of problem where the goal is to find the overall optimal return through a certain path among the searching pool. In the pro- cess, the search agent is faced with many branches to take a try and see how it goes at each time step, and it depends on limited knowledge to make de- cisions.
  
The primary challenging of dynamic search problems is to balance the exploration and ex- ploitation rate of the search agent, that is, whether the agent relies on current knowledge to explore already known good paths, or it spends in trying something new which may potentially be better than all known options.
Compared with basic multi-armed Bandit algo- rithm, the Contextual Bandit algorithm includes context features each time it makes predictions of which way to go. As a result, such an algorithm could make more reasonable decisions on explo- ration or exploitation contradiction. 
  
Our task discussed in this paper is that, given a bunch of topics, finding the most relevant docu- ment in a 20-year New York Times corpus for each topic. We build our reinforcement learning model on a search engine and a user agent. We use an open source searching tool called Elasticsearch to retrieve relevant documents from the corpus. With each query fed, Elasticsearch would give five high- est scoring documents. We then input them to a user agent1, which is designed to imitate a user’s behavior when faced with a group of searching re- sults. From the user agent, we can get the feed- back of the most relevant document and its rele- vance score, which scales from 0 (not relevant) to 4 (highly relevant). This score is used to instruct the model to adjust the query structure and per- form following query requests, which will be de- tailed in section three. Ideally, the model would converge to the optima and be able to predict an optimal search path at the end of certain times of iterations of all topics.  
  
We use the New York Time corpus for train- ing the model and perform predictions. Topics to search in the task involves various fields, instanced as dental implants, Who Outed Valerie Plame?, Montserrat eruption effects, etc.  

  
2 Task
-------
  
Now that we already have a hypothetically opti- mal search engine and user agent, we want to opti- mize what we feed to the search engine to get more relevant results. Hence, we formulate our task of dynamic search resolution as a set of decisions of what action to take on the current query structure for every state the model step on during iterations.
  
At each state, the model would predict an action to reformulate the query, which will yield some new searching results, also new a relevance score given by the user agent. We use the same struc- ture as that in LinUCB algorithm (Li et al.2010). In LinUCB algorithm, the action is selected by whose reward linearly combined with a contextual feature vector and model parameter matrix is maximized. Based on the predicted action and the user agent score, the model then updates the parameters by increasing or punishing the value correspond- ing to the predicted action.
  
  
3 Model
------
  
We aim to learn a policy to reformulate the query based on the searching history of each topic.
  
## 3.1 Query Reformulation Strategy

We set a list of actions with four potential options at each state: add (a word to the query), remove (a word from the query), weigh (every word in the query) and stop the search for the current topic and start the next one.
  
At each time step to decide an action, what our model can observe includes: five relevant documents retrieved by the search engine, the most relevant one among the five and its score provided by the user agent, as well as the current topic and the next one. Four potential actions will reformulate the query in following ways:
  
### Add    
Add action will add a new word to the current query. The word is selected from the most relevant document of the state. For each word in this document, we can calculate a tf-idf value (following the BM25 definition), which has positive correlation with the term frequency, as well as a negative correlation with its document frequency in all five relevant files, indicating its weight and uniqueness in the selected top-one file.
  
### Remove     
As described before, we also calculate the tf-idf value for each word in the most relevant file, based on which we can get the tf-idf value of each word in the query, and then remove the one with the smallest value from the query.
  
### Weigh     
The search engine (ElasticSearch) actually takes the query in the way that each word in it carries a weight. All words in the topic start with equal weight as 1.0. The weigh action changes the weight of each word. Still by calculating the tf-idf value of each word in the most relevant file, we can get a list of the most uniquely frequent words in this file and a list of the opposite. We set the size of both lists as 20. For the word in the query, if it is in the top-20 list, we increase its weight in the scale of its rank in top 20. On contrary, if the word is in the last-20 list, we decrease its weight in the scale of its rank in last 20. Eventually a query with same words but adjusted weights is returned.
  
### Stop       
The stop action simply stops the search of current topic and starts the new topic iteration if it’s not the last one on the topic list.
  
## 3.2 Reward Representation
  
For an action taken, the model can then observe five relevant files output from the search engine and one most relevance score from the user agent, which is use to update the corresponding weight parameter of this action. Instead of simply using the accumulated score as the reward in this paper (Yang2017), we set a minus value as the reward for those actions which yield no relevance score, to punish their parameters more. For actions leading to positive relevance scores, we also multiple their score by a rate range from 0 to 1 as the reward, except when the model predicts exactly the right action.
  
It should be noted that the stop action should be selected when none of ”add”, ”remove” and ”weigh” actions could yield a better score than starting searching a new topic.
  
Based on such reward representation and query reformulation strategy, if the model predicts an ac- tion the same as the gold label, which is the one enabling the next search results to scored the high- est from the user agent, then the query text updates as the actions says.
  
## 3.3 Context Feature  
  
The context feature grabs features related to the whole searching history. We follow the setting in former papers and set a context feature with di- mension of six. Features contain: the number of documents that search engine has seen, the num- ber of relevant documents of last iteration, the number of relevant documents that search engine has seen, times that action ”add” has taken, times that action ”remove” has taken, and times that ac- tion ”weigh” has taken.
The context feature is supposed to update dur- ing the path searching of each topic.
  
4 Experiments and Results
-------
  
The experiment is implemented on a topic list with sixty query-like topics based on the New York Times corpus. A ground-truth database reveals information including subtopics of each topic, doc- uments and passages related to each subtopic and their related rates, which serves exactly as the ”user agent” in our system and indicates how good are search results every time Elasticsearch picks up some documents.
  
![image](https://github.com/samj001/COSC689-finalproject/blob/master/image/figure1.png)
=-Figure 1-=
  
We start learning the model with a learning rate of 0.01, explore rate of 0.25 and a six-dimension of context feature. The model keeps predicting and updating weights until the ”stop” scores the highest among the four. Figure 1 shows an ex- ample of model’s procedures when executing the algorithm.
  
The first line shows a list of initial search results of the first topic and their feedback from the user agent. Here the score of each document is summed up by all subtopics related passages’ rates. Four possible actions and their corresponding reformu- lated queries and scores follow. We can see that the ”add” action reformulates the query by adding the word ”Ms.”, which is selected from document three according to the tf-idf value. Also, the ”re- move” action removes the word ”paintings” from the query, the ”weigh” action lowers the weight of ”Klimt”, ”paintings”, ”Maria” and ”Altmann”, and the ”stop” action gets much lower score so the search of this topic keeps going on.
  
From the model running, we find that it usu- ally stops around ten to two steps’ updating. The ”add” action is called most frequently. It some- times keeps picking up the same word added to the query, however, so the query text can become quite redundant. The ”weigh” action can very of- ten have impacts on the query, but it is very seldom predicted and picked by the model.
  
When searching different topics, we notice that the model can get very different reward scores, from thirty to over four hundred. But the total re- ward, or even average rewards per step seems to rely heavy on the topic itself. From the database we can find some topics just have more related documents and are supposed to rated higher than others.
  
5 Analysis
---------
  
Our model is consist of two parts, the search en- gine and the learning model, which both con- tribute to the overall performance of the search task. ElasticSearch is a high level application and we find it not very handy to use, so our revising mainly focuses on the learning model itself.
  
We notice several problems of our model and think of some solutions.
  
### Query drift   
Among the four actions, the ”add” action is very often selected by our model, and sometimes it even keeps picking up the same word added to the query. Hence, to maintain original information in the query to some extent and avoid the query drifted to added words too much, we set a lower weight for added words. We tried both 0.7 and 0.5 scenarios and found that 0.5 makes more sense.
  
### Avoid noise   
When doing ”add” and ”remove” actions, we rely on the tf-idf value of words in the user-agent-selected best document to make decision on which word to come and go. The tf-idf value, however, could be a highly noisy factor about the document’s focus information, especially when we only get the inverse document frequency value from those five document picked by ElasticSearch. To make some improvements, we use a random selecting mechanism among several words with top-ranked tf-idf values. Also, removing stop words and tokenization are performed before counting to lower the chance that the model would select meaningless words.
  
### Encourage fluctuation   
We find some of our topics to search in New York Times corpus can get quite fair results just by ElasticSearch even at the beginning of one iteration. We also saw that it’s not a rare case that one high-scored document keeps being selected. The reason could be that there are some texts in the collection quite close to the query. One or two high-scored documents keep showing up could make sense since they’re valid result and should be among the output of the search. However it also eliminates the chance to see more documents. Storing already selected high scored documents and removing them from the collection might be a solution.
  
Also, we found that because there are only four options their weight matrix are actually in a quite small scale, the expected rewards at a time step for two or even three possible actions are sometimes very similar, which makes them all gold labels at the point. Under this circumstances, we let the gold label be the one close to the model’s prediction, which enables the next step of search to use a reformulated query, so that it’s more likely to find more new information.
  
### ElasticSearch  
We use ElasticSearch to index all documents from New York Times corpus and do the search and feed the result to the user agent. However, we find a very weird, purely technical is- sue when joining these two parts. The thing is, all document index in the user agent database (gen- erated from jig-truth ground file) are strings of seven-digit number, in other words, padded with zero(s) at the beginning of the string if they are smaller number. For document keys in Elastic- Search however, they are all strings of real num- bers (with no zeros padded at the beginning). It seems that this exact difference prevents us get feedback from the jig-database by document in- dex output by ElasticSearch. We have literally looked up and tried every kind of converting or other methods but still can’t figure out this issue. As a result, we couldn’t able to work on docu- ments with index less than 1,000,000, which is about half of the corpus. We really wish to get help and find a solution to this.
  
### other future improvements   
Our model follows the former paper and uses a six-dimensional fea- ture vector to capture information about how far and how good the searching is going on for now. However, we are also thinking about that this con- textual can be enhanced to a more wide range, for example, vectorize the
  
6 Conclusion
-----
  
In this project, we implement a contextual bandit search for a dynamic search task. We searched sixty topics in 1987-2007 year New York Times corpus. We finds that the contextual information about search history actually enables the model to stop at a point and start a new query without the need of manually set up a stop threshold, and it could still get high scored results. However, we also find some limits of exploitation range of our model.
  
References
-------
  
[1] Angela Yang and Grace Hui Yang. 2017. A Contex- tual Bandit Approach to Dynamic Search. In Pro- ceedings of the ACM SIGIR International Confer- ence on Theory of Information Retrieval (ICTIR ’17). ACM, New York, NY, USA, 301-304. DOI: https://doi.org/10.1145/3121050.3121101
  
[2] Lihong Li, Wei Chu, John Langford, Robert E. Schapire. 2010. A Contextual-Bandit Approach to personalized News Article Recommandation.


