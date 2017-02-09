from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

training_data_path = '../training_data/'

print 'building tlc classifier'
brand_pipeline = SGDPipe(
    training_data_path + 'tlc_items',
    training_data_path + 'top_level_category'
)
joblib.dump(brand_pipeline, '../pkl_jar/tlc_pipeline.pkl')