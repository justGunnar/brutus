from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

training_data_path = '../training_data/'

print 'building beer brand classifier'
brand_pipeline = SGDPipe(
    training_data_path + 'beers',
    training_data_path + 'beer_brands'
)
joblib.dump(brand_pipeline, '../pkl_jar/beer_brand_pipeline.pkl')