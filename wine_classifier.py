import numpy as np

from pyspark                         import SparkContext
from pyspark.sql                     import Row
from splearn.rdd                     import DictRDD
from splearn.rdd                     import ArrayRDD
from splearn.pipeline                import SparkPipeline
from splearn.linear_model            import SparkSGDClassifier
from splearn.feature_extraction.text import SparkCountVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer

sc = SparkContext()

# text_file   = sc.textFile("s3://yogi-training-data/wines")
# wines = text_file.collect()[:50] # TODO: the next 2 line are a waste, fix it
# w_rdd = sc.parallelize(wines)

# w_rdd.cache()

# labels_file = sc.textFile("s3://yogi-training-data/wine_categories")
# categories = labels_file.collect()[:50] # TODO: the next 2 line are a waste, fix it
# c_rdd = sc.parallelize(categories)

# c_rdd.cache()

# rdd = w_rdd.zip(c_rdd)

# full_rdd = DictRDD(rdd, columns=('X', 'y'), dtype=[np.ndarray, np.ndarray])

X = ['monkey is brown', 'bear is also brown', 'apple has a stem', 'orange has a rind']
y = ['animal', 'animal', 'fruit', 'fruit']
X_rdd = sc.parallelize(X)
y_rdd = sc.parallelize(y)
Z = DictRDD((X_rdd, y_rdd), columns=('X', 'y'))

clf = SparkSGDClassifier()
clf.fit_tansform(Z)

# pipeline = SparkPipeline((
#     ('vect', SparkCountVectorizer()),
#     ('tfidf', SparkTfidfTransformer()),
#     ('clf', SparkSGDClassifier())
# ))

# pipeline.fit(Z, clf__classes=Z.get('y'))
