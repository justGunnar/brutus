###########################
#
# Yogi
#
###########################
import json

from flask               import Flask
from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

###########################
#
# Startup
#
###########################
pkl_jar_path = 'pkl-jar/'
print 'loading brand pipeline'
brand_pipeline = joblib.load(pkl_jar_path + 'brand_pipeline.pkl')

print 'loading beer pipeline'
beer_pipeline = joblib.load(pkl_jar_path + 'beer_pipeline.pkl')

print 'loading wine pipeline'
wine_pipeline = joblib.load(pkl_jar_path + 'wine_pipeline.pkl')

print 'loading liquor pipeline'
liquor_pipeline = joblib.load(pkl_jar_path + 'liquor_pipeline.pkl')

print 'ready'

###########################
#
# Endpoints
#
###########################
app = Flask('yogi')

@app.route('/', methods=['GET'])
def home():
    return 'This is the homepage.'

@app.route('/brand/<to_predict>', methods=['GET'])
def brand(to_predict):
    return brand_pipeline.classify([to_predict])[0]

@app.route('/brands', methods=['GET'])
def brands():
    brands      = brand_pipeline._labels
    uniq_brands = set(brands) # uniq!
    return json.dumps(list(uniq_brands))

@app.route('/beer_category/<to_predict>', methods=['GET'])
def beer_category(to_predict):
    return beer_pipeline.classify([to_predict])[0]

@app.route('/wine_category/<to_predict>', methods=['GET'])
def wine_category(to_predict):
    return wine_pipeline.classify([to_predict])[0]

@app.route('/liquor_category/<to_predict>', methods=['GET'])
def liquor_category(to_predict):
    return liquor_pipeline.classify([to_predict])[0]
