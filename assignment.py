import nltk
from nltk.util import ngrams
import numpy as np
from math import log
from decimal import *
import markovgen


getcontext().prec=10

def ngramModel(text,n):
	return ngrams(text,n)


def mleEstimator(n,length_corpus,ngram_model,helper_ngram1=None,helper_ngram2=None):
	unigram_count={}
	unigram_stat={}
	bigram_count={}
	trigram_count={}
	bigram_stat={}
	trigram_stat={}
	condprob={}
	
	
	if(n==1):
		ngram_model1=ngram_model
	elif(n==2):
		ngram_model1=helper_ngram1
		ngram_model2=ngram_model
	else:
		ngram_model1=helper_ngram1
		ngram_model2=helper_ngram2
		ngram_model3=ngram_model

	for ngram in ngram_model1:
		if ngram in unigram_count:
			unigram_count[ngram]+=1
		else:
			unigram_count[ngram]=1
		#unigram_count[ngram]=corpus_ngrams.count(ngram)
		unigram_stat[ngram]=(Decimal((unigram_count[ngram]))/Decimal((length_corpus)))
	
	if(n>=2):
		for bigram in ngram_model2:
			if bigram in bigram_count:
				bigram_count[bigram]+=1
			else:
				bigram_count[bigram]=1
			tup=(str(bigram[0]))
			tup1=tuple([tup])
			bigram_stat[bigram]=(Decimal((bigram_count[bigram]))/Decimal((unigram_count[tup1])))
		if(n==3):
			for trigram in ngram_model3:
				if trigram in trigram_count:
					trigram_count[trigram]+=1
				else:
					trigram_count[trigram]=1
				tup=str((trigram[0]))
				tup1=str((trigram[1]))
				tup2=tuple([tup]+[tup1])
				trigram_stat[trigram]=(Decimal((trigram_count[trigram]))/Decimal(bigram_count[tup2]))

	if(n==1):
		return unigram_count,unigram_stat
	elif(n==2):
		return bigram_count,bigram_stat
	else:
		return trigram_count,trigram_stat	

def crossEntropy(ngram_model,mle_prob):

	entropy=Decimal(0.0)
	temp_value=Decimal(1.0)
	for gram in ngram_model:
		log_value=Decimal(log(mle_prob[gram],2))
		prob_value=Decimal(mle_prob[gram])
		temp_value=log_value*prob_value
		entropy+=temp_value
		#entropy+=Decimal((Decimal(mle_prob[gram]))*(Decimal(log(mle_prob[gram],2))))
	#entropy=Decimal(entropy/Decimal(length_corpus))
	return -entropy

def mleEstimator_smoothing(train_model1,unigram_count,ngram_model,mle_prob_train,n,train_model2=None,bigram_count=None):
	
	mle_prob=mle_prob_train
	smooth_mle={}
	#print bigram_count	
	V_model=set(ngram_model)
	V=len(V_model)
	if(n==1):
		N=len(train_model1)
		for gram in ngram_model:
			if(gram not in mle_prob_train):
				mle_prob[gram]=0
			new_n=Decimal(N*mle_prob[gram])+Decimal(1.0)
			smooth_mle[gram]=Decimal(new_n/Decimal(N+V))
		return smooth_mle
	elif(n==2):

		for gram in ngram_model:
			tup=str(gram[0])
			tup1=tuple([tup])
			if(gram not in mle_prob_train):
				mle_prob[gram]=Decimal(0.0)
			if(tup1 not in unigram_count):
				unigram_count[tup1]=Decimal(1.0)
			new_n=Decimal(unigram_count[tup1]*mle_prob[gram])+Decimal(1.0)
			smooth_mle[gram]=Decimal(new_n/Decimal(unigram_count[tup1]+V))
			#smooth_mle[gram]=smooth_mle[gram]*Decimal(unigram_count[tup1])
		return smooth_mle
	else:
		for gram in ngram_model:
			tup=str(gram[0])
			tupp=str(gram[1])
			tup1=tuple([tup]+[tupp])
			if(gram not in mle_prob_train):
				mle_prob[gram]=Decimal(0.0)
			if(tup1 not in bigram_count):
				bigram_count[tup1]=Decimal(1.0)
			new_n=Decimal(bigram_count[tup1]*mle_prob[gram])+Decimal(1.0)
			#print "before"
			smooth_mle[gram]=Decimal(new_n/Decimal(bigram_count[tup1]+V))
			#print "after"
			#smooth_mle[gram]=smooth_mle[gram]*Decimal(unigram_count[tup1])
		#print smooth_mle
		return smooth_mle
	return None;

def authorIdentification(cross_entropy):
	print("The cross entropy values for each of the models: \n")
	print(cross_entropy)
	minimum=cross_entropy[0]
	index=0
	for every in cross_entropy:
		if(every<minimum):
			minimum=every
			index=cross_entropy.index(every)
	if(index==0):
		return "Corpus 3"
	elif(index==1):
		return "Corpus 2"
	else:
		return "Corpus 1"



def generateText(corpus1,corpus2,corpus3):
	c1=' '.join(corpus1)
	text_file = open("Output.txt", "w")
	text_file.write(c1)
	text_file.close()
	_file=open("Output.txt")
	markov = markovgen.Markov(_file)
	print("For corpus 1")
	print(markov.generate_markov_text())
	c1=' '.join(corpus2)
	text_file = open("Output1.txt", "w")
	text_file.write(c1)
	text_file.close()
	_file=open("Output1.txt")
	markov = markovgen.Markov(_file)
	print("For corpus 2")
	print(markov.generate_markov_text())
	c1=' '.join(corpus3)
	text_file = open("Output2.txt", "w")
	text_file.write(c1)
	text_file.close()
	_file=open("Output2.txt")
	markov = markovgen.Markov(_file)
	print("For corpus 3")
	print(markov.generate_markov_text())


def main():

	c1=list(nltk.corpus.gutenberg.words('austen-emma.txt'))
	c2=list(nltk.corpus.gutenberg.words('edgeworth-parents.txt'))
	c3=list(nltk.corpus.gutenberg.words('whitman-leaves.txt'))

	c1_test=c1[50:250]
	c2_test=c2[250:450]
	c3_test=c3[50:250]
	c1=c1[:50]+c1[250:]
	c2=c2[:250]+c2[450:]
	c3=c3[:50]+c3[250:]

	test=input('enter the test data set needed : 1,2,3 ' )
	if(test==1):
		test_corpus=c1_test
	elif(test==2):
		test_corpus=c2_test
	else:
		test_corpus=c3_test


	train_corpus=[c1]+[c2]+[c3]
	

	i=-1
	train_unigram=list()
	train_bigram=list()
	train_trigram=list()
	train_unigram_count=list()
	train_bigram_count=list()
	train_trigram_count=list()
	train_unigram_stat=list()
	train_bigram_stat=list()
	train_trigram_stat=list()

	
	for corpus in train_corpus:
		i=i+1
		#hello=[list(ngramModel(corpus,1))]
		train_unigram.insert(i,list(ngramModel(corpus,1)))
		train_bigram.insert(i,list(ngramModel(corpus,2)))
		train_trigram.insert(i,list(ngramModel(corpus,3)))
		a,b=mleEstimator(1,len(corpus),train_unigram[i])
		train_unigram_count.insert(i,a)
		train_unigram_stat.insert(i,b)
		c,d=mleEstimator(2,len(corpus),train_bigram[i],helper_ngram1=train_unigram[i])
		train_bigram_count.insert(i,c)
		train_bigram_stat.insert(i,d)
		e,f=mleEstimator(3,len(corpus),train_trigram[i],helper_ngram1=train_unigram[i],helper_ngram2=train_bigram[i])
		train_trigram_count.insert(i,e)
		train_trigram_stat.insert(i,f)


	#for the test_corpus
	test_unigram=list(ngramModel(test_corpus,1))
	test_bigram=list(ngramModel(test_corpus,2))
	test_trigram=list(ngramModel(test_corpus,3))

	test_smoothing_unigram=list()
	test_smoothing_bigram=list()
	test_smoothing_trigram=list()
	test_cross_entropy_unigram=list()
	test_cross_entropy_bigram=list()
	test_cross_entropy_trigram=list()

	for x in range(0,3):
		test_smoothing_unigram.insert(x,mleEstimator_smoothing(train_unigram[x],train_unigram_count[x],test_unigram,train_unigram_stat[x],1))
		test_smoothing_bigram.insert(x,mleEstimator_smoothing(train_unigram[x],train_unigram_count[x],test_bigram,train_bigram_stat[x],2))
		test_smoothing_trigram.insert(x,mleEstimator_smoothing(train_unigram[x],train_unigram_count[x],test_trigram,train_trigram_stat[x],3,train_model2=train_bigram[x],bigram_count=train_bigram_count[x]))
		test_cross_entropy_unigram.insert(x,[float(crossEntropy(test_unigram,test_smoothing_unigram[x]))])
		test_cross_entropy_bigram.insert(x,[float(crossEntropy(test_bigram,test_smoothing_bigram[x]))])
		test_cross_entropy_trigram.insert(x,[float(crossEntropy(test_trigram,test_smoothing_trigram[x]))])


	option=input('Enter choices for\n1)Author Identification\n2)Display Cross Entropy\n3)Generate Text\n4)Exit')

	while(option!=4):
		if(option==1):
			print(authorIdentification(test_cross_entropy_trigram))

			a=mleEstimator_smoothing(train_unigram[2],train_unigram_count[2],test_trigram,train_trigram_stat[2],3,train_model2=train_bigram[2],bigram_count=train_bigram_count[2])
			b=[float(crossEntropy(test_trigram,test_smoothing_trigram[2]))]
			print(b)
			print(test_cross_entropy_trigram)
			print(test_cross_entropy_bigram)
			print(test_cross_entropy_unigram)

		elif(option==2):
			print("1)Cross entropy for UNIGRAM model:\n")
			for x in range(0,3):
				print(test_cross_entropy_unigram[2-x])
			print("2)Cross entropy for BIGRAM model:\n")
			for x in range(0,3):
				print(test_cross_entropy_bigram[2-x])
			print("3)Cross entropy for TRIGRAM model:\n")
			for x in range(0,3):
				print(test_cross_entropy_trigram[2-x])
				
		else:
			generateText(c1,c2,c3)
		option=input('Enter choices for\n1)Author Identification\n2)Display Cross Entropy\n3)Generate Text\n4)Exit')

if __name__=="__main__":
	main()





