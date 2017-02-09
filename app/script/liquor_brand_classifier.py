from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

training_data_path = '../training_data/'

print 'building liquor brand classifier'
brand_pipeline = SGDPipe(
    training_data_path + 'liquors',
    training_data_path + 'liquor_brands'
)
joblib.dump(brand_pipeline, '../pkl_jar/liquor_brand_pipeline.pkl')