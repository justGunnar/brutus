import pdb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster                 import KMeans

names = []
with open('../training_data/c_items') as items:
  for item in items:
    names.append(item.rstrip())

vectorizer = TfidfVectorizer(
  stop_words=['pk', 'yr', 'ml', 'oz', 'can', '750', '750ml', '6pk', '2014', '7yr', '2013', '2015',
              'box', '12', '06', '16', '16oz', 'dtx', '4cn', '06pk', '21st', 'amendment', 'amend'],
  max_df=0.7,
  ngram_range=(1, 2)
)
X = vectorizer.fit_transform(names)

true_k = 9
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print "Cluster %d:" % i,
    for ind in order_centroids[i, :10]:
        print ' %s' % terms[ind],
    print
