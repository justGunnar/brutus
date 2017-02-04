"""
A simple text classification pipeline that recognizes "spark" from
input text. This is to show how to create and configure a Spark ML
pipeline in Python. Run with:

  bin/spark-submit examples/src/main/python/ml/simple_text_classification_pipeline.py
"""
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row, SQLContext

# DONT FORGET THAT PREDICTIONS COME OUT AS FLOATS, THIS IS IMPORTANT WHEN RETRIEVING
# THE STRING THAT REPRESENTS THE PREDICTION

sc = SparkContext(appName="SimpleTextClassificationPipeline")
sqlCtx = SQLContext(sc)

text_file   = sc.textFile("s3://yogi-training-data/wines")
labels_file = sc.textFile("s3://yogi-training-data/wine_categories")

texts  = text_file.collect()[:600]
labels = labels_file.collect()[:600]
unique_labels = list(set(labels)) # unique

zipped = []
for i in range(0, len(texts)):
  zipped.append(
    (i, texts[i], unique_labels.index(labels[i]))
  )

# Prepare training documents, which are labeled.
LabeledDocument = Row("id", "text", "label")
training = sc.parallelize(zipped).map(lambda x: LabeledDocument(*x)).toDF()

# Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = RandomForestClassifier()
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled.
Document = Row("id", "text")
test = sc.parallelize([(4, "Ice Wine"),
                       (5, "Stone Something Else Wine"),
                       (6, "Bud Light 6pk"),
                       (7, "Standing Stone Ice Wine 1 Bottle")]) \
    .map(lambda x: Document(*x)).toDF()

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "prediction")
for row in selected.collect():
    print row

sc.stop()
