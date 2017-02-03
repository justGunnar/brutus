import random

import numpy as np

from splearn.rdd import DictRDD
from splearn.feature_extraction.text import SparkHashingVectorizer
from splearn.feature_extraction.text import SparkTfidfTransformer
from splearn.svm import SparkLinearSVC
from splearn.pipeline import SparkPipeline

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from pyspark import SparkContext


from splearn.linear_model import SparkSGDClassifier

sc = SparkContext()

text_file   = sc.textFile("s3://yogi-training-data/wines")
wines = text_file.collect() # TODO: the next 2 line are a waste, fix it
w_rdd = sc.parallelize(wines, 4)
# w_rdd.cache()

labels_file = sc.textFile("s3://yogi-training-data/wine_categories")
categories = labels_file.collect() # TODO: the next 2 line are a waste, fix it
c_rdd = sc.parallelize(categories, 4)
# c_rdd.cache()

rdd = w_rdd.zip(c_rdd)

full_rdd = DictRDD(rdd, columns=('X', 'y'), dtype=[np.ndarray, np.ndarray])

dist_pipeline = SparkPipeline((
    ('vect', SparkHashingVectorizer()),
    ('tfidf', SparkTfidfTransformer()),
    ('clf', SparkSGDClassifier())
))

dist_pipeline.fit(full_rdd, clf__classes=np.unique(categories))

y_pred_dist = dist_pipeline.predict(Z[:, 'X'])