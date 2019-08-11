# COSC689-finalproject
# Dynamic search system via deep reinforcement learning on NYT corpus

Abstract
	In this paper we present a contextual bandit ap- proach to dynamic search tasks. It is imple- mented on the New York Times Corpus and evaluated with CUBE tests and precision met- rics. Our approach relies on a search engine and a user agent, creating an interacting pro- cess of the information retrieval task. The re- inforcement learning method, which is based on each time of interactions, allows the system to optimize the searching path to find the most relevant information from the corpus.  
	

1 Introduction
	Dynamic search is a kind of problem where the goal is to find the overall optimal return through a certain path among the searching pool. In the pro- cess, the search agent is faced with many branches to take a try and see how it goes at each time step, and it depends on limited knowledge to make de- cisions.
	The primary challenging of dynamic search problems is to balance the exploration and ex- ploitation rate of the search agent, that is, whether the agent relies on current knowledge to explore already known good paths, or it spends in trying something new which may potentially be better than all known options.
	Compared with basic multi-armed Bandit algo- rithm, the Contextual Bandit algorithm includes context features each time it makes predictions of which way to go. As a result, such an algorithm could make more reasonable decisions on explo- ration or exploitation contradiction.
	Our task discussed in this paper is that, given a bunch of topics, finding the most relevant docu- ment in a 20-year New York Times corpus for each topic. We build our reinforcement learning model on a search engine and a user agent. We use an open source searching tool called Elasticsearch to retrieve relevant documents from the corpus. With each query fed, Elasticsearch would give five high- est scoring documents. We then input them to a user agent1, which is designed to imitate a user’s behavior when faced with a group of searching re- sults. From the user agent, we can get the feed- back of the most relevant document and its rele- vance score, which scales from 0 (not relevant) to 4 (highly relevant). This score is used to instruct the model to adjust the query structure and per- form following query requests, which will be de- tailed in section three. Ideally, the model would converge to the optima and be able to predict an optimal search path at the end of certain times of iterations of all topics.
	We use the New York Time corpus for train- ing the model and perform predictions. Topics to search in the task involves various fields, instanced as dental implants, Who Outed Valerie Plame?, Montserrat eruption effects, etc.  
	




