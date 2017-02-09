from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

training_data_path = '../training_data/'

print 'building wine brand classifier'
brand_pipeline = SGDPipe(
    training_data_path + 'wines',
    training_data_path + 'wine_brands'
)
joblib.dump(brand_pipeline, '../pkl_jar/wine_brand_pipeline.pkl')