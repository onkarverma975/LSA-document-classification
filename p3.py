from collections import Counter
from shutil import copyfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import _pickle as pickle
import re
import path
import os
import numpy as np
import sys
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


train_data=[]
test_data=[]
class_train_flag=[]
class_test_flag=[]
num_files=[]
num_classes = 0
vocabulary = set()
path_test = '../q2data/test'
path_train = '../q2data/train'

def read_freq_file(path):
	with open(path, errors='ignore') as f1:
		shop = f1.read()
		regex = r'\b\w+\b'
		list1=re.findall(regex,shop)
		list1 = [x for x in list1 if not any(c.isdigit() for c in x)]
		c=Counter(x.strip() for x in list1)
		return c
def create_test_data():
	global num_classes
	global num_files
	global train_data
	global test_data
	global class_train_flag
	global class_test_flag
	
	num_classes = len([f for f in os.listdir(os.path.join(path_train)) ])
	num_files = [0]*num_classes
	for class_no in range(0,num_classes):
		for f in os.listdir(os.path.join(path_test,str(class_no).zfill(1))):
			# print(f)
			os.remove(os.path.join(path_test,str(class_no).zfill(1), f))


		num_files[class_no] = len([f for f in os.listdir(os.path.join(path_train,str(class_no).zfill(1))) \
				if os.path.isfile(os.path.join(path_train,str(class_no).zfill(1), f))])
				
		for file_no in range(int(num_files[class_no]*0.8)+1,num_files[class_no]+1):
			copyfile(os.path.join(path_train,str(class_no).zfill(1),str(file_no).zfill(3)+'.txt'),\
			os.path.join(path_test,str(class_no).zfill(1),str(file_no).zfill(3)+'.txt'))


		for f in os.listdir(os.path.join(path_test,str(class_no).zfill(1))):
			test_data.append(read_freq_file(os.path.join(path_test,str(class_no).zfill(1),f)))
			class_test_flag.append(class_no)


		for file_no in range(1,int(num_files[class_no]*0.8)+1):
			train_data.append(read_freq_file(os.path.join(path_train,str(class_no).zfill(1),str(file_no).zfill(3)+'.txt')))
			class_train_flag.append(class_no)
def build_lexicon(train_data):
	lexicon = set()
	for doc in train_data:
		lexicon.update([x for x in doc])
	filtered_words = [word for word in lexicon if word not in stopwords.words('english')]

	return filtered_words
def tf(term, document):
	if term in document:
		return document[term]
	else:
		return 0
def find_dim(threshold, Vt , s):
	
	num_dim=0
	# pc = U*np.diag(s)
	# pc = pc[:,::-1]
	# explained_variance = np.var(pc, axis=0)
	# full_variance = np.var(tf_idf_matrix,axis=0)
	# explained_variance_ratio = explained_variance / full_variance.sum()
	# print(explained_variance_ratio.cumsum())
	cumsum = (s/s.sum()).cumsum()
	# cumsum = explained_variance_ratio.cumsum()
	# print(cumsum)

	# print(threshold)
	for i in range(0,len(s)):
		num_dim+=1
		if cumsum[i]>threshold:
			break
	# num_dim = cumsum[np.where(cumsum>threshold)].shape[1]
	print('Number of Dimensions for threshold-> ',threshold, '\t',num_dim)
	return Vt[0:num_dim,:].T
	
	# B = U[:,0:num_dim]*((np.diag(s))[0:num_dim,0:num_dim])
	# B_test = test_tfidf*V[0:num_dim,:].T

	# print (s)
	# print (V)
def sign(num):
	if np.asscalar(num) >0:
		return 1
	else:
		return -1
def OVAPerceptronTraining(x_train, y_train):
	num_obs = len(x_train)
	a,num_feat = x_train[0].shape
	output=[]
	for i in range(0,num_classes):
		output.append(np.zeros(num_feat))
	c1=0
	c2=0
	for i in range(0,num_obs):
		actual = y_train[i]
		pred = 0
		prod_max = np.asscalar(np.dot(x_train[i],output[0].T))

		for class_no in range(0,num_classes):
			prod = np.asscalar(np.dot(x_train[i],output[class_no].T))

			if prod > prod_max:
				pred = class_no
				prod_max = prod
	
		if pred != actual:
			output[actual] = output[actual]+x_train[i]
			output[pred] = output[pred]-x_train[i]
			c1+=1
		else:
			c2+=1
	print(c1,c2)
	return output
def OVAPerceptronTesting(output, x_test, y_test):
	correct_v=0
	incorrect_v=0
	for ind in range(0,len(x_test)):
		pred = 0
		actual = y_test[ind]
		prod_max=0
		w = output[0]

		prod_max=np.asscalar(np.dot(x_test[ind],w.T))

		for class_number in range(0,num_classes):
			w = output[class_number]
			prod=np.asscalar(np.dot(x_test[ind],w.T))

			if prod > prod_max:
				pred = class_number
				prod_max = prod

		if pred == actual:
			correct_v+=1
		else:
			incorrect_v+=1
	print(correct_v, incorrect_v)
	return float(float(correct_v)/float(incorrect_v+correct_v))
def cosine_similarity( x_train, y_train, x_test, y_test):
	correct_v=0
	incorrect_v=0
	for test_doc in range(0,len(x_test)):
		cosines=[]
		for train_doc in range(0,len(x_train)):
			a=x_train[train_doc]
			b=x_test[test_doc]
			moda = np.linalg.norm(a)
			modb = np.linalg.norm(b)
			cosines.append((y_train[train_doc], np.dot(a,b.T)/moda/modb))
		top_10 = sorted(cosines, key=lambda x: x[1], reverse = True)[0:10]
		# print (top_10)
		count=[0]*num_classes
		for i in top_10:
			count[i[0]]+=1
		pred = count.index(max(count))
		if pred == y_test[test_doc]:
			correct_v+=1
		else:
			incorrect_v+=1
	return float(float(correct_v)/float(incorrect_v+correct_v))
def perform_tfidf(train_data, test_data, vocabulary):

	doc_term_matrix = []
	for doc in train_data+test_data:
	    tf_vector = [tf(word, doc) for word in vocabulary]
	    doc_term_matrix.append(tf_vector)

	doc_term_matrix = np.array(doc_term_matrix)

	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(doc_term_matrix)
	tf_idf_matrix = np.matrix(tfidf.transform(doc_term_matrix).toarray())
	
	tf_idf_matrix-=np.matrix((np.mean(tf_idf_matrix, 0)))

	U, s, V = np.linalg.svd( tf_idf_matrix , full_matrices=False)
	

	return s, V
	# return s,V [0:num_dim,:].T	
def perform_tfidf_validation(train_data, test_data, vocabulary):
	doc_term_matrix = []
	for doc in train_data:
	    tf_vector = [tf(word, doc) for word in vocabulary]
	    doc_term_matrix.append(tf_vector)

	doc_term_matrix = np.array(doc_term_matrix)
	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(doc_term_matrix)
	doc_term_matrix = doc_term_matrix[0:len(train_data)]
	tf_idf_matrix = np.matrix(tfidf.transform(doc_term_matrix).toarray())
	tf_idf_matrix-=np.matrix((np.mean(tf_idf_matrix, 0)))
	return tf_idf_matrix, tfidf
def perform_transform(test_data, tfidf, vocabulary):
	
	doc_term_matrix = []
	for doc in test_data:
	    tf_vector = [tf(word, doc) for word in vocabulary]
	    doc_term_matrix.append(tf_vector)

	doc_term_matrix = np.array(doc_term_matrix)
	tf_idf_matrix = np.matrix(tfidf.transform(doc_term_matrix).toarray())
	tf_idf_matrix-=np.matrix((np.mean(tf_idf_matrix, 0)))
	return tf_idf_matrix
def dimension_reduction(V, Matrix):
	return Matrix*V
def perform_tfidf_query(train_data):
	vocabulary = build_lexicon(train_data)
	doc_term_matrix = []
	for doc in train_data:
	    tf_vector = [tf(word, doc) for word in vocabulary]
	    doc_term_matrix.append(tf_vector)

	doc_term_matrix = np.array(doc_term_matrix)
	tfidf = TfidfTransformer(norm="l2")
	tfidf.fit(doc_term_matrix)
	tf_idf_matrix = np.matrix(tfidf.transform(doc_term_matrix).toarray())
	tf_idf_matrix-=np.matrix((np.mean(tf_idf_matrix, 0)))
	U, s, V = np.linalg.svd( tf_idf_matrix , full_matrices=False)
	return tf_idf_matrix, tfidf, vocabulary, V, s
	# return s,V [0:num_dim,:].T	
def cosine_similarity_query(x_train, y_train, x_test):
	for test_doc in range(0,len(x_test)):
		cosines=[]
		for train_doc in range(0,len(x_train)):
			a=x_train[train_doc]
			b=x_test[test_doc]
			moda = np.linalg.norm(a)
			modb = np.linalg.norm(b)
			cosines.append((y_train[train_doc], np.dot(a,b.T)/moda/modb))
		top_10 = sorted(cosines, key=lambda x: x[1], reverse = True)[0:10]
		count=[0]*num_classes
		for i in top_10:
			count[i[0]]+=1
		pred = count.index(max(count))
		return pred
	return 0
def create_test_data_query():
	global num_classes
	global num_files
	global train_data
	global test_data
	global class_train_flag
	global class_test_flag
	num_classes = len([f for f in os.listdir(os.path.join(path_train)) ])
	num_files = [0]*num_classes
	for class_no in range(0,num_classes):
		num_files[class_no] = len([f for f in os.listdir(os.path.join(path_train,str(class_no).zfill(1))) \
				if os.path.isfile(os.path.join(path_train,str(class_no).zfill(1), f))])
		for file_no in range(1,int(num_files[class_no])+1):
			train_data.append(read_freq_file(os.path.join(path_train,str(class_no).zfill(1),str(file_no).zfill(3)+'.txt')))
			class_train_flag.append(class_no)
# if __name__ == "__main__":

if sys.argv[1]=='1':
	if len(sys.argv)>=3:
		path_train = sys.argv[2]

	if len(sys.argv)>=4:
		path_test = sys.argv[3]
	create_test_data()
	vocabulary = build_lexicon(train_data)
	s, Vt = perform_tfidf(train_data, test_data, vocabulary)
	B, tfidf = perform_tfidf_validation(train_data, test_data, vocabulary)
	B_test = perform_transform(test_data, tfidf, vocabulary)
	with open("Save_SVD.data",'wb') as fp:
	    pickle.dump(s,fp)
	    pickle.dump(Vt,fp)
	    pickle.dump(B,fp)
	    pickle.dump(B_test,fp)
elif sys.argv[1]=='2':
	print("Threshold\tNum Dim")
	with open("Save_SVD.data",'rb') as fp:
	    s=pickle.load(fp)
	    Vt=pickle.load(fp)
	for threshold in range(50,110,5):
		find_dim(float(threshold/100),Vt , s)
elif sys.argv[1]=='3':
	if len(sys.argv)>=3:
		path_train = sys.argv[2]

	if len(sys.argv)>=4:
		path_test = sys.argv[3]
	create_test_data()
	vocabulary = build_lexicon(train_data)
	with open("Save_SVD.data",'rb') as fp:
	    s=pickle.load(fp)
	    Vt=pickle.load(fp)
	    B=pickle.load(fp)
	    B_test=pickle.load(fp)
	
	X_plot=[]
	Y_plotp=[]
	Y_plotc=[]
	print('threshold\taccuracy_perceptron\taccuracy_cosine')
	for threshold in range(60,101,5):
		V = find_dim(float(threshold/100),Vt , s)

		B_transformed = dimension_reduction(V, B)
		B_test_transformed = dimension_reduction(V, B_test)
		x_train=[]
		y_train=[]
		for ind in range(0,len(B_transformed)):
			# x_train.append(np.append(np.squeeze(B_transformed[ind]),np.ones((1,1)), axis=1))
			x_train.append(np.squeeze(B_transformed[ind]))
			y_train.append(class_train_flag[ind])

		model = OVAPerceptronTraining(x_train, y_train)

		x_test=[]
		y_test=[]
		for ind in range(0,len(B_test_transformed)):
			# x_test.append(np.append(np.squeeze(B_test_transformed[ind]),np.ones((1,1)), axis=1))
			x_test.append(np.squeeze(B_test_transformed[ind]))
			# x_test.append(np.squeeze(B_test_transformed[ind]))
			y_test.append(class_test_flag[ind])

		accuracy_perceptron = OVAPerceptronTesting(model, x_test, y_test)
		accuracy_cosine = cosine_similarity(x_train, y_train, x_test, y_test)
		print(threshold,'\t', accuracy_perceptron,'\t', accuracy_cosine)

		X_plot.append(threshold)
		Y_plotp.append(accuracy_perceptron*100)
		Y_plotc.append(accuracy_cosine*100)
	plt.plot(X_plot, Y_plotp, color = 'blue', label='Perceptron Accuracy')
	plt.plot(X_plot, Y_plotc, color = 'black', label='Cosine Sim Accuracy')

	for i in range(0,len(X_plot)):
		plt.plot(X_plot[i], Y_plotp[i], 'ro')

	for i in range(0,len(X_plot)):
		plt.plot(X_plot[i], Y_plotc[i], 'ro')

	plt.axis([50, 110, 60, 110])
	plt.legend(loc='best') 
	plt.xlabel('Threshold (in %)', fontsize=18)
	plt.ylabel('Accuracy', fontsize=18)
	plt.show()
elif sys.argv[1]=='4':
	if len(sys.argv)>=3:
		path_train = sys.argv[2]

	if len(sys.argv)>=4:
		path_test = sys.argv[3]

	if len(sys.argv)>=5:
		class_already = int(sys.argv[4])

	create_test_data_query()
	query_file = read_freq_file(path_test)
	tfidf_matrix, tfidf, vocabulary, Vt, s = perform_tfidf_query(train_data)
	query_tfidf = perform_transform([query_file], tfidf, vocabulary)

	V = find_dim(float(95/100),Vt , s)

	train_data = dimension_reduction(V,tfidf_matrix)
	query_tfidf = dimension_reduction(V,query_tfidf)

	predicted_class = cosine_similarity_query(train_data, class_train_flag, query_tfidf)
	print("Predicted Class-> ",predicted_class,'\n',"Query Class-> ", class_already)