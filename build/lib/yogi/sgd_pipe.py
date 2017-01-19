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

class SGDPipe:

    def __init__(self, text_file, labels_file, downscale_denominator = False):
        df = self._build_data_frame(text_file, labels_file, downscale_denominator)
        self._pipeline = self._build_pipeline(df)

    def classify(self, strings, include_meta_data = False):
        """Predict the label of a set of strings

        strings -- array of strings to classify
        """
        if include_meta_data:
            probabilities = self._pipeline.predict_proba(strings)
            worst_to_best = numpy.argsort(probabilities, axis=1)
            best_index = worst_to_best[0][-1]
            return (self._pipeline.classes_[best_index], probabilities[0][best_index])
        else:
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
            df = DataFrame(rows)

        self._strings = df['text'].values
        self._labels  = df['labels'].values

        return df.reindex(numpy.random.permutation(df.index))


    def _build_pipeline(self, data_frame):
        """Build and return the classification pipeline

        data_frame -- the data to build the classifier with
        """
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf_transformer', TfidfTransformer()),
            ('classifier', SGDClassifier(loss='modified_huber'))
        ])

        pipeline.fit(data_frame['text'].values, data_frame['labels'].values)

        return pipeline


