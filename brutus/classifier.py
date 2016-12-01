###########################
#
# Introducing: Brutus
#
# This is a first pass script that builds a
# Stochastic Gradient Descent estimator. It leverages
# a count vectorizer, and a term frequency-inverse document
# frequency reansformer in order to classify raw items into
# either brand or catalog item buckets. In validate mode, the
# script will run k-fold cross validation, and print out the
# results. In the defualt mode, the classifier will be trained,
# and you will be dropped into an interactive shell.
#
###########################
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

###########################
#
# Build command line args
#
###########################
parser = argparse.ArgumentParser()
parser.add_argument(
    '-l',
    '--label_set',
    help='brands OR catalog_items; the set of labels to match items against',
    type=str,
    required=True
)
parser.add_argument(
    '-f',
    '--full',
    help='whether or not to work on the full dataset',
    type=bool,
    default=False
)
parser.add_argument(
    '-v',
    '--validate',
    help='''whether or not to run in validate mode.
            use -k to set the number of permutes to test''',
    type=bool,
    default=False
)
parser.add_argument(
    '-k',
    '--k_folds',
    help='the number of permutations to test if in validate mode',
    type=int,
    default=6
)
args = vars(parser.parse_args())

###########################
#
# Build up the data frame
# NOTE: Input files must have the same length
#       and be in the current directory
#
###########################
rows = []
with open('items') as items, open(args['label_set']) as catalog_items:
    for item, catalog_item in izip(items, catalog_items):
        rows.append({
            'text': item.rstrip(),
            'class': catalog_item.rstrip()
        })

if args['full']:
    data = DataFrame(rows)
else:
    data = DataFrame(rows[0:len(rows)/500])

# randomize the dataset to avoid bias
data = data.reindex(numpy.random.permutation(data.index))

###########################
#
# Build the pipeline
#
###########################
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf_transformer', TfidfTransformer()),
    ('classifier', SGDClassifier())
])

if args['validate']:
    ###########################
    #
    # Validate the model
    #
    ###########################
    kf = KFold(n_splits=args['k_folds'])
    for train_indices, test_indices in kf.split(data):
        train_text   = data.iloc[train_indices]['text'].values
        train_labels = data.iloc[train_indices]['class'].values

        test_text   = data.iloc[test_indices]['text'].values
        test_labels = data.iloc[test_indices]['class'].values

        pipeline.fit(train_text, train_labels)
        predictions = pipeline.predict(test_text)

        count_correct = 0
        for i, prediction in enumerate(predictions):
            if prediction == test_labels[i]:
                count_correct += 1

        print('Correct:', count_correct)
        print('Total:', len(predictions))
        print('Percentage:', int((float(count_correct) / len(predictions)) * 100))
        print("\n")

else:
    pipeline.fit(data['text'].values, data['class'].values)

    ###########################
    #
    # Add an interactive shell
    #
    ###########################
    vars = globals().copy()
    vars.update(locals())
    shell = code.InteractiveConsole(vars)
    shell.interact()

    # prediction example:
    # pipeline.predict(['SKYY VODKA 750mL', 'MACALLAN 12yr'])
