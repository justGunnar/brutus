import os
import numpy
import time
import fileinput
import readline
import code
import argparse
import pdb

from pandas                          import DataFrame
from itertools                       import izip
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.linear_model            import SGDClassifier
from sklearn.pipeline                import Pipeline
from sklearn.model_selection         import KFold

class GuzzlrClassifier:

    def __init__(self, text_file, labels_file, downscale_denominator = False):
        df = _build_data_frame(text_file, labels_file, downscale_denominator)
        self._pipeline = _build_pipeline(df)

    def classify(strings):
        """Predict the lable of a set of strings

        strings -- array of strings to classify
        """
        return self._pipeline.predict(strings)

    def _build_data_frame(self, text_file, labels_file, downscale_denominator):
        """Build and return a data frame

        text_file             -- the text file to use to train the classifier
        labels_file           -- the text file to use as labels
        downscale_denominator -- what we'd like to divide the dataset length by
        """

        rows = []
        with open(text_file) as text, open(labels_file) as labels:
            for string, label in izip(text, labels):
                rows.append({
                    'text': string.rstrip(),
                    'labels': label.rstrip()
                })

        if downscale_denominator:
            df = DataFrame(rows[0:len(rows)/downscale_denominator])
        else:
            df DataFrame(rows)

        self._strings = df['text'].values
        self._labels  = df['labels'].values

        return df.reindex(numpy.random.permutation(df.index))


    def _build_pipeline(data_frame):
        """Build and return the classification pipeline

        data_frame -- the data to build the classifier with
        """
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', SGDClassifier())
        ])

        pipeline.fit(data['text'].values, data['class'].values)

        return pipeline


