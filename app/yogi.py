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
pkl_jar_path = 'pkl_jar/'
print 'loading top level category pipeline'
top_level_category_pipeline = joblib.load(pkl_jar_path + 'tlc_pipeline.pkl') # TODO: Extra vs Extras??

print 'loading beer brand pipeline'
beer_brand_pipeline = joblib.load(pkl_jar_path + 'beer_brand_pipeline.pkl')

print 'loading wine brand pipeline'
wine_brand_pipeline = joblib.load(pkl_jar_path + 'wine_brand_pipeline.pkl')

print 'loading liquor brand pipeline'
liquor_brand_pipeline = joblib.load(pkl_jar_path + 'liquor_brand_pipeline.pkl')

print 'loading beer leaf category pipeline'
beer_pipeline = joblib.load(pkl_jar_path + 'beer_pipeline.pkl')

print 'loading wine leaf category pipeline'
wine_pipeline = joblib.load(pkl_jar_path + 'wine_pipeline.pkl')

print 'loading liquor leaf category pipeline'
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

@app.route('/top_level_category/<path:to_predict>', methods=['GET'])
def top_level_category(to_predict):
    return top_level_category_pipeline.classify([to_predict])[0]

@app.route('/beer_brand/<path:to_predict>', methods=['GET'])
def beer_brand(to_predict):
    return beer_brand_pipeline.classify([to_predict])[0]

@app.route('/wine_brand/<path:to_predict>', methods=['GET'])
def wine_brand(to_predict):
    return wine_brand_pipeline.classify([to_predict])[0]

@app.route('/liquor_brand/<path:to_predict>', methods=['GET'])
def liquor_brand(to_predict):
    return liquor_brand_pipeline.classify([to_predict])[0]

@app.route('/beer_category/<path:to_predict>', methods=['GET'])
def beer_category(to_predict):
    return beer_pipeline.classify([to_predict])[0]

@app.route('/wine_category/<path:to_predict>', methods=['GET'])
def wine_category(to_predict):
    return wine_pipeline.classify([to_predict])[0]

@app.route('/liquor_category/<path:to_predict>', methods=['GET'])
def liquor_category(to_predict):
    return liquor_pipeline.classify([to_predict])[0]

@app.route('/brands', methods=['GET'])
def brands():
    brands      = brand_pipeline._labels
    uniq_brands = set(brands) # uniq!
    return json.dumps(list(uniq_brands))
