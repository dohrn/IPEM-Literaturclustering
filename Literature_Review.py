import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import numpy as np
import csv
import re

from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from pandas import ExcelWriter
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin_min

def get_user_input():

	whichfield = input("Welches Feld soll betrachtet werden? [Title = 0, Abstract = 1, Keywords = 2] :    ")
	include_abstract = False
	include_author_keywords = False
	include_string = True
	#include_abstract = input("Soll der Abstract berücksichtigt werden? [True/False]   ")
	if whichfield == 1:
		include_abstract = True
	elif whichfield == 2:
		include_author_keywords = True


	#include_string = bool(input("Soll der String im Clustering berücksichtigt werden? [True/False]   "))

	# if include_string == "True":
	# 	include_string = True
	# else:
	# 	include_string = False

	# include_author_keywords = bool(input("Sollen die Author Keywords berücksichtigt werden? [True/False]   "))

	# if include_author_keywords == "True":
	# 	include_author_keywords = True
	# else:
	# 	include_author_keywords  = False

	#print(include_abstract, include_string, include_author_keywords)

	database_name = str(input("Name der Datenbank   "))
	#Bitte den Suchstring in Anführungszeichen setzen "(Production AND Machine Learning)"
	suchstring = "0"
	#suchstring = str(input("Wie lautet der Suchstring? ohne [\" und *]   "))
	cluster_multiplyer = 1
	#cluster_multiplyer = float(input("Cluster-Multiplayer:   "))
	kmax = int(input("kmax für Ellenbogenmethode: "))
	explained_variance = float(input("Explained Variance für SVD [Standard = 0.7]:  "))
	nb_topwords = 5
	return include_abstract, include_string, include_author_keywords, database_name, suchstring, cluster_multiplyer, explained_variance, nb_topwords, kmax

#Extrahiere Wörter aus Suchstring

def extract_searchstring(suchstring):

	searchstring = suchstring.replace("(","")
	searchstring = searchstring.replace(")","")
	searchstring = re.split(' AND | OR | ',searchstring)
	searchstring = [x.lower() for x in searchstring]
	orig_searchstring = {"Suchstring":suchstring}
	orig_searchstring = pd.DataFrame(orig_searchstring, index=[0])

	return searchstring, orig_searchstring

def keyw_titles(database_name):

	try:
		database = "Datenbanken/{}.csv".format(database_name)
		lit = pd.read_csv(database, encoding="utf8", error_bad_lines=False, warn_bad_lines=False)
	except:
		database = "Datenbanken/{}.xlsx".format(database_name)
		lit = pd.read_excel(database, encoding="utf8")

	if include_author_keywords == True:
		lit["Keywords_Title"] = lit["Author Keywords"]
		#lit.loc[pd.notnull(lit["Author Keywords"]), "Keywords_Title"] = lit["Author Keywords"] #+ " " + lit["Document Title"]
		#lit.loc[pd.isnull(lit["Author Keywords"]), "Keywords_Title"] = lit["Document Title"]
	elif include_abstract == True:
		lit["Keywords_Title"] = lit["Abstract"]
		#lit.loc[pd.notnull(lit["Abstract"]), "Keywords_Title"] = lit["Abstract"]
	else:
		lit["Keywords_Title"] = lit["Document Title"]


	return lit

def tokenizing_stopw(searchstring,include_string):

	stopwords_eng = nltk.corpus.stopwords.words('english')
	stopwords_ger = nltk.corpus.stopwords.words("german")
	print(include_string)
	if include_string == False:
		stopwords = stopwords_eng + stopwords_ger + searchstring
	else:
		stopwords = stopwords_eng + stopwords_ger

	lit["Document Title 1"] = lit["Keywords_Title"].astype(str)
	lit["Document Title 1"] = lit["Document Title 1"].str.lower()
	lit["Document Title 1"] = lit["Document Title 1"].apply(word_tokenize)
	lit["Document Title 1"] = lit["Document Title 1"].apply(lambda x: [item for item in x if item not in stopwords])
	lit["Document Title 1"] = lit["Document Title 1"].str.join(" ")

	fr = len(lit["Document Title 1"])

	lit.drop_duplicates(subset="Document Title 1", inplace=True)
	tx = len(lit["Document Title 1"])
	#print(lit["Document Title 1"])
	print("Von {} auf {} Titel".format(fr, tx))



def stemming(lit):
	#titles = literature.loc[:,"Title"].values
	#titles = [item for item in titles]

	keywords = lit.loc[:,"Document Title 1"].values.astype('U')
	keywords = [item for item in keywords]

	stemmer = SnowballStemmer("english")
	stemmer2 = SnowballStemmer("german")


	x_titles = [nltk.word_tokenize(x) for x in keywords]

	singles = [[stemmer.stem(word) for word in y] for y in x_titles]
	singles = [[stemmer2.stem(word) for word in y] for y in x_titles]

	singles = [' '.join(x) for x in singles]

	return singles



def vectorizing(singles):

	vectorizer_1 = TfidfVectorizer()
	x_titles = vectorizer_1.fit_transform(singles)
	feature_names = pd.DataFrame(vectorizer_1.get_feature_names())

	return vectorizer_1, x_titles, feature_names

def dim_reduce(x_titles, explained_variance):

	svd = TruncatedSVD(n_components=len(lit), n_iter=10, random_state=42)
	principalComponents = svd.fit_transform(x_titles)

	principalDf = pd.DataFrame(data=principalComponents)
	explained_variance_list = np.cumsum(svd.explained_variance_ratio_)
	n_comp = next(i for i,v in enumerate(explained_variance_list) if v > explained_variance)
	svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
	principalComponents = svd.fit_transform(x_titles)
	principalDf = pd.DataFrame(data=principalComponents)
	print(principalComponents.shape[1])
	return principalDf


def plot_silhouette(x, kmax):
	sil = []

	# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
	for k in range(2, kmax+1):
		kmeans = KMeans(n_clusters = k).fit(x)
		labels = kmeans.labels_
		sil.append(silhouette_score(x, labels, metric = 'euclidean'))

	plt.plot(sil)
	plt.ylabel("Silhouette Score")
	plt.savefig("Silhouette_score")
	return sil

def plot_ellbow(principalDf, kmax):
	sse = []
	K = range(1,kmax+1)
	for k in K:
		kmeanModel = KMeans(n_clusters=k, random_state=5)
		kmeanModel.fit(principalDf)
		sse.append(kmeanModel.inertia_)

	plt.figure(figsize=(16,8))
	plt.plot(K, sse, 'bx-')
	plt.xlabel('k')
	plt.ylabel('Inertia')
	plt.title('The Elbow Method showing the optimal k')
	plt.savefig("Ergebnisse/Ellbow_{}.png".format(database_name))
	plt.close()

def cluster(principalDf, lit):
	clusters = int(cluster_multiplyer * (len(lit) / 2) ** 0.5)
	#clusters = 2
	model_titles = KMeans(n_clusters=clusters, random_state=5)
	y_kmeans = model_titles.fit_predict(principalDf)
	centroids = model_titles.cluster_centers_
	print("centroids: {} \n inertia: {} \n iterations: {} \n".format(centroids, model_titles.labels_, model_titles.inertia_, model_titles.n_iter_))
	closest, distances = pairwise_distances_argmin_min(model_titles.cluster_centers_, principalDf)
	print(pd.DataFrame(list(zip(range(1,len(closest)+1), closest,  list(lit.iloc[closest, 1]), distances)), columns=["Cluster", "Closest_to_cent","Title", "Euc_dist_to_cent"]))

	tf_idf_norm = normalize(x_titles)
	tf_idf_array = tf_idf_norm.toarray()

	lit["Cluster"] = list(y_kmeans)
	print("Anzahl Cluster: {}".format(clusters))
	return y_kmeans, centroids, tf_idf_norm, tf_idf_array, clusters



def get_top_features_cluster(tf_idf_array, prediction, n_feats):
	labels = np.unique(prediction)
	dfs = []
	for label in labels:
		id_temp = np.where(prediction==label)
		x_means = np.mean(tf_idf_array[id_temp], axis = 0)
		sorted_means = np.argsort(x_means)[::-1][:n_feats]
		features = vectorizer_1.get_feature_names()
		best_features = [(features[i], x_means[i], label) for i in sorted_means]
		df = pd.DataFrame(best_features, columns = ['features', 'score', "cluster"])
		dfs.append(df)
	return dfs


def get_clusters(dfs):
	counts = lit["Cluster"].value_counts()
	for i in dfs:
		i["score_sum"] = sum(i["score"])
		i["cluster_size"] =  counts[i["cluster"][0]]
	topwords = pd.concat(dfs, axis=0)
	grouped_cluster = lit.groupby('Cluster').size().nlargest(clusters)
	biggest_cluster = grouped_cluster.index
	return biggest_cluster, grouped_cluster, topwords, counts



def plot_2d(centroids, dfs, biggest_cluster, counts):

	svd = TruncatedSVD(n_components=2, n_iter=7, random_state=42)
	principalComponents = svd.fit_transform(centroids)

	fig = plt.figure(figsize=(30,15))
	for i in biggest_cluster:
		plt.scatter(principalComponents[i, 0], principalComponents[i, 1],s = 10 * counts[i], label = 'Cluster {} {} titles {}'.format(i, counts[i],dfs[i]["features"].values), alpha=0.5,)


	plt.rcParams.update({'font.size': 15})
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlabel("PCA1")
	plt.ylabel("PCA2")

	plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
	plt.subplots_adjust(right=0.5)
	if include_string == True:
		plt.savefig("Ergebnisse/Cluster_Visualisierung_with_string_{}.png".format(database_name))
	else:
		plt.savefig("Ergebnisse/Cluster_Visualisierung_without_string_{}.png".format(database_name))
	plt.close()




def save_to_excel(database_name, include_string):
	if include_string == True:
		writer = ExcelWriter('Ergebnisse/Literature_clustered_with_string_{}.xlsx'.format(database_name))
	else:
		writer = ExcelWriter('Ergebnisse/Literature_clustered_without_string_{}.xlsx'.format(database_name))

	literature = lit.loc[:,["Document Title", "Author Keywords", "Authors", "Publication_Year", "Abstract", "Cluster"]]
	literature.to_excel(writer,'Literatur')
	topwords.to_excel(writer, "Topwords")
	grouped_cluster.to_excel(writer, "Biggest Cluster")
	orig_searchstring.to_excel(writer, "Suchstring")
	feature_names.to_excel(writer, "Feature Names")
	writer.save()
	print("Literatur-Datenbank liegt unter Literature_clustered_{}.xlsx \nDie Visualisierung der Cluster liegt unter Cluster_Visualisierung_{}.png".format(database_name, database_name))

# In[11]:

if __name__ == '__main__':
	include_abstract, include_string, include_author_keywords, database_name, suchstring, cluster_multiplyer, explained_variance, nb_topwords, kmax =  get_user_input()
	searchstring, orig_searchstring = extract_searchstring(suchstring)
	lit = keyw_titles(database_name)
	tokenizing_stopw(searchstring, include_string)
	print(include_string, include_abstract, include_author_keywords)
	singles = stemming(lit)
	vectorizer_1, x_titles, feature_names = vectorizing(singles)
	principalDF = dim_reduce(x_titles, explained_variance)
	plot_ellbow(principalDF, kmax)
	y_kmeans, centroids, tf_idf_norm, tf_idf_array, clusters= cluster(principalDF, lit)
	dfs = get_top_features_cluster(tf_idf_array, y_kmeans, nb_topwords)
	biggest_cluster, grouped_cluster, topwords, counts = get_clusters(dfs)

	plot_2d(centroids, dfs, biggest_cluster, counts)
	save_to_excel(database_name, include_string)