

def get_doc_tfidf(doc,docs,query=None):

	stop_words = stopword.words('english')

	tokens = word_tokenize(doc)

	counts = {}

	for tok in tokens:
		if tok not in stop_words:
			if tok not in counts.keys():
				counts[tok] = 1
			else:
				counts[tok] +=1

	meaning = list(counts.keys())

	df_counts = dict(zip(meaning,[0]*len(meaning)))
	
	for d in docs:
		for word in word_tokenize(d):
			if word in counts.keys():
				df_counts += 1

	tfidfs = []
	for m in meaning:
		tfidf = counts[m]*math.log((5-df_counts[m])/df_counts[m]+0.5)

		tfidfs.append(tfidf)

	if query == None:

		max_f = max(tfidfs)

		max_words = []
		for n,w in enumerate(tfidfs):
			if w == max_words:

				max_words.append(meaning[n])

		return random.choice(max_words)


	else:
		query_tok = word_tokenize(query)
		query_tfidf = [0] * len(query_tok)
		for n,q in enumerate(query_tok):
			if q in meaning:
				query_tfidf[n] = counts[q]*math.log((5-df_counts[q])/df_counts[q]+0.5)
		return query_tfidf

