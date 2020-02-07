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
from sklearn.cluster import AgglomerativeClustering

plt.rcParams.update({'font.size': 22})


def get_user_input():

	whichfield = int(input("Welches Feld soll betrachtet werden? [Title = 0, Abstract = 1, Keywords = 2] :    "))
	include_abstract = False
	include_author_keywords = False
	include_string = int(input("Soll der Searchstring im BoW erhalten bleiben? [Ja = 1, Nein = 0]"))

	if include_string == 0:
		include_string = False
	else:
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
	suchstring =  str(input("Bitte Suchstring eingeben: "))
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
	#print(searchstring)
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

def tokenizing_stopw(searchstring,include_string, lit):
	print(lit)
	stopwords_eng = nltk.corpus.stopwords.words('english')
	stopwords_ger = nltk.corpus.stopwords.words("german")
	if include_string == False:
		stopwords = stopwords_eng + stopwords_ger + searchstring
	else:
		stopwords = stopwords_eng + stopwords_ger
	#print(include_string, stopwords)
	fr = len(lit)
	lit.dropna(inplace=True, subset=["Document Title"])
	lit["Document Title duplicates"] = lit["Document Title"].astype(str)
	lit["Document Title duplicates"] = lit["Document Title duplicates"].str.lower().apply(word_tokenize).str.join(" ")

	lit["Document Title 1"] = lit["Keywords_Title"].astype(str)
	lit["Document Title 1"] = lit["Document Title 1"].str.lower().apply(word_tokenize).apply(lambda x: [
		item for item in x if item not in stopwords and item.isalpha()]).str.join(" ")
	#print(lit)


	lit.drop_duplicates(subset="Document Title duplicates", inplace=True)
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

	vectorizer_1 = TfidfVectorizer(min_df = 0.01, max_df = 1, ngram_range=(1,2))
	x_titles = vectorizer_1.fit_transform(singles)
	feature_names = pd.DataFrame(vectorizer_1.get_feature_names())

	return vectorizer_1, x_titles, feature_names

def dim_reduce(x_titles, explained_variance, lit, database_name, include_string):
	svd = TruncatedSVD(n_components=x_titles.get_shape()[1]-1, n_iter=10, random_state=42)
	#print("features",x_titles.get_shape()[1])
	principalComponents = svd.fit_transform(x_titles)

	#print(svd.explained_variance_ratio_.sum())

	principalDf = pd.DataFrame(data=principalComponents)
	explained_variance_list = np.cumsum(svd.explained_variance_ratio_)
	n_comp = next(i for i,v in enumerate(explained_variance_list) if v > explained_variance)
	plt.figure(figsize=(20,10))
	plt.yticks(np.arange(0, 1, 0.1))
	plt.grid()
	plt.plot(explained_variance_list)

	plt.axvline(x=n_comp, ymin=0, ymax=explained_variance, color="b")
	axes = plt.gca()
	axes.plot([0, n_comp], [explained_variance, explained_variance], color="Blue")
	axes.annotate(n_comp,
            xy=(n_comp, 0))
	axes.set_ylim([0,1])
	axes.set_xlim(left=0)
	plt.xlabel('Components')
	plt.ylabel('Explained Variance')
	# if include_string == True:
	# 	plt.title('Explained Variance with number of components with searchstring included')
	# else:
	# 	plt.title('Explained Variance with number of components with searchstring excluded')

	plt.savefig("Ergebnisse/Explained_variance_with_string_{}_{}.png".format(include_string, database_name))
	plt.close()
	svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
	principalComponents = svd.fit_transform(x_titles)
	components = pd.DataFrame(zip(svd.components_, svd.explained_variance_, svd.explained_variance_ratio_, svd.singular_values_), columns=["Components", "Explained Variance", "Explained Variance Ratio", "Singular Values"])
	principalDf = pd.DataFrame(data=principalComponents)
	#print(principalComponents.shape[1])
	return principalDf, components


def plot_silhouette_ellbow(principalDf, kmax, include_string, clusters):
	# from yellowbrick.cluster import SilhouetteVisualizer
	# from yellowbrick.datasets import load_nfl

	# # Instantiate the clustering model and visualizer
	# model = KMeans(kmax, random_state=42)

	# plt.figure(figsize=(50,50))
	# axes = plt.gca()
	# visualizer = SilhouetteVisualizer(model, colors='yellowbrick', ax=axes)
	# visualizer.fit(x)        # Fit the data to the visualizer
	# visualizer.show()
	# plt.close()
	ssc = []
	sse = []
	K = range(2,kmax+1,1)
	for k in K:
		kmeanModel = KMeans(n_clusters=k, random_state=5)
		kmeanModel.fit(principalDf)
		sse.append(kmeanModel.inertia_)
		preds = kmeanModel.fit_predict(principalDf)
		centers = kmeanModel.cluster_centers_
		score = silhouette_score(principalDf, preds, metric='euclidean')
		ssc.append(score)

	plt.figure(figsize=(10, 10))
	plt.grid()
	plt.plot(K, ssc, 'bx-')
	axes = plt.gca()
	axes.plot([0, clusters], [ssc[clusters-2], ssc[clusters-2]], color="Green")
	axes.plot([clusters, clusters], [0, ssc[clusters-2]], color="Green")
	axes.annotate(clusters,
           xy=(clusters, 0))
	axes.annotate("{0:.2f}".format(ssc[clusters-2]),
            xy=(0, ssc[clusters-2]))
	axes.set_xlim(left=0)
	axes.set_ylim(bottom=0)
	start, end = axes.get_xlim()
	axes.xaxis.set_ticks(np.arange(start, end, 3))
	plt.xlabel('k')
	plt.ylabel('SSC')
	# if include_string == True:
	# 	plt.title('The Silhouette Score showing the optimal k with Searchstring included')
	# else:
	# 	plt.title('The Silhouette Score showing the optimal k with Searchstring excluded')
	plt.savefig("Ergebnisse/Silhouette_with_string_{}_{}.png".format(include_string, database_name))
	plt.close()

	plt.figure(figsize=(10, 10))
	plt.grid()
	plt.plot(K, sse, 'bx-')
	axes = plt.gca()
	axes.plot([0, clusters], [sse[clusters-2], sse[clusters-2]], color="Green")
	axes.plot([clusters, clusters], [0, sse[clusters-2]], color="Green")
	axes.annotate(clusters,
		xy=(clusters, 0))
	axes.annotate("{0:.2f}".format(sse[clusters-2]),
            xy=(0, sse[clusters-2]))
	axes.set_xlim(left=0)
	axes.set_ylim(bottom=0)
	start, end = axes.get_xlim()
	axes.xaxis.set_ticks(np.arange(start, end, 3))
	plt.xlabel('k')
	plt.ylabel('SSE')

	# if include_string == True:
	# 	plt.title("The Elbow Method showing the optimal k with searchstring included")
	# else:
	# 	plt.title("The Elbow Method showing the optimal k with searchstring excluded")
	plt.savefig("Ergebnisse/Ellbow_with_string_{}_{}.png".format(include_string, database_name))
	plt.close()


# def plot_ellbow(principalDf, kmax):
# 	sse = []
# 	K = range(2,kmax+1, 1)
# 	for k in K:
# 		kmeanModel = KMeans(n_clusters=k, random_state=5)
# 		kmeanModel.fit(principalDf)
# 		sse.append(kmeanModel.inertia_)

# 	plt.figure(figsize=(20, 10))
# 	plt.grid()
# 	plt.plot(K, sse, 'bx-')

# 	plt.xlabel('k')
# 	plt.ylabel('Inertia')
# 	plt.title('The Elbow Method showing the optimal k')
# 	plt.savefig("Ergebnisse/Ellbow_with_string_{}_{}.png".format(include_string, database_name))
# 	plt.close()

def agg_clustering(principalDf):
	import scipy.cluster.hierarchy as shc

	plt.figure(figsize=(10, 7))
	plt.title("Customer Dendograms")
	dend = shc.dendrogram(shc.linkage(principalDf, method='ward'))
	plt.savefig("dendogram")
	plt.close()
	clusters = int(cluster_multiplyer * (len(lit) / 2) ** 0.5)
	model_titles =  AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')
	y_kmeans = model_titles.fit_predict(principalDf)
	lit["Cluster_agg"] = list(y_kmeans)


def cluster(principalDf, lit):
	#clusters = int(cluster_multiplyer * (len(lit) / 2) ** 0.5)
	clusters = 13
	model_titles = KMeans(n_clusters=clusters, random_state=5)
	y_kmeans = model_titles.fit_predict(principalDf)
	centroids = model_titles.cluster_centers_
	centroids = [[round(j,2) for j in i] for i in centroids]

	#print("centroids: {} \n inertia: {} \n iterations: {} \n".format(centroids, model_titles.labels_, model_titles.inertia_, model_titles.n_iter_))
	closest, distances = pairwise_distances_argmin_min(model_titles.cluster_centers_, principalDf)
	evaluation_metrics = pd.DataFrame(list(zip(range(0,len(closest)), closest,  list(lit.iloc[closest, 1]), [round(i, 2) for i in distances], centroids)), columns=["Cluster", "Closest_to_cent","Title", "Euc_dist_to_cent", "Centroid"])
	#print(model_titles.inertia_)
	tf_idf_norm = normalize(x_titles)
	tf_idf_array = tf_idf_norm.toarray()

	lit["Cluster"] = list(y_kmeans)
	print("Anzahl Cluster: {}".format(clusters))
	return y_kmeans, centroids, tf_idf_norm, tf_idf_array, clusters, evaluation_metrics



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
		plt.scatter(principalComponents[i, 0], principalComponents[i, 1],s = 100 * counts[i], label = 'Cluster {} {} titles {}'.format(i, counts[i],dfs[i]["features"].values), alpha=0.5,)

	plt.xlabel("Component 1")
	plt.ylabel("Component 2")
	plt.grid()

	plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
	plt.subplots_adjust(right=0.5)
	if include_string == True:
		plt.savefig("Ergebnisse/Cluster_Visualisierung_with_string_{}.png".format(database_name))
	else:
		plt.savefig("Ergebnisse/Cluster_Visualisierung_without_string_{}.png".format(database_name))
	plt.close()




def save_to_excel(database_name, include_string, evaluation_metrics, components):
	if include_string == True:
		writer = ExcelWriter('Ergebnisse/Literature_clustered_with_string_{}.xlsx'.format(database_name))
	else:
		writer = ExcelWriter('Ergebnisse/Literature_clustered_without_string_{}.xlsx'.format(database_name))

	literature = lit.loc[:,["Document Title", "Author Keywords", "Authors", "Publication_Year", "Abstract", "Cluster"]]
	literature.to_excel(writer,'Literature')
	topwords.to_excel(writer, "Topwords")
	grouped_cluster.to_excel(writer, "Biggest Cluster")
	orig_searchstring.to_excel(writer, "Searchstring and Params")
	feature_names.to_excel(writer, "Feature Names")
	evaluation_metrics.to_excel(writer, "Evaluation Metrics")
	components.to_excel(writer, "SVD-Components")
	writer.save()
	print("Literatur-Datenbank liegt unter Literature_clustered_{}.xlsx \nDie Visualisierung der Cluster liegt unter Cluster_Visualisierung_{}.png".format(database_name, database_name))

# In[11]:

if __name__ == '__main__':
	include_abstract, include_string, include_author_keywords, database_name, suchstring, cluster_multiplyer, explained_variance, nb_topwords, kmax =  get_user_input()
	searchstring, orig_searchstring = extract_searchstring(suchstring)
	lit = keyw_titles(database_name)
	tokenizing_stopw(searchstring, include_string, lit)
	print(include_string, include_abstract, include_author_keywords)
	singles = stemming(lit)
	vectorizer_1, x_titles, feature_names = vectorizing(singles)
	principalDF, components = dim_reduce(x_titles, explained_variance, lit, database_name, include_string)
	agg_clustering(principalDF)
	y_kmeans, centroids, tf_idf_norm, tf_idf_array, clusters, evaluation_metrics = cluster(principalDF, lit)
	plot_silhouette_ellbow(principalDF, kmax, include_string, clusters)

	dfs = get_top_features_cluster(tf_idf_array, y_kmeans, nb_topwords)
	biggest_cluster, grouped_cluster, topwords, counts = get_clusters(dfs)

	plot_2d(centroids, dfs, biggest_cluster, counts)
	save_to_excel(database_name, include_string, evaluation_metrics, components)