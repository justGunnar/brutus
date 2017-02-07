import time

from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

training_data_path = '../training_data/'

print 'building brand pipeline'
start = time.time()
brand_pipeline = SGDPipe(
    training_data_path + 'items',
    training_data_path + 'brands'
)
end = time.time()
joblib.dump(brand_pipeline, '../pkl_jar/full_brand_pipeline.pkl')
print end - start