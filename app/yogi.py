###########################
#
# Yogi
#
###########################
import json

from flask    import Flask
from sgd_pipe import SGDPipe

###########################
#
# Startup
#
###########################
print 'building brand pipeline'
brand_pipeline = SGDPipe(
    'training_data/items',
    'training_data/brands',
    200
)

print 'building wine pipeline'
wine_pipeline  = SGDPipe(
    'training_data/wines',
    'training_data/wine_categories'
)

print 'building liquor pipeline'
liquor_pipeline  = SGDPipe(
    'training_data/liquors',
    'training_data/liquor_categories'
)

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

@app.route('/wine_category/<to_predict>', methods=['GET'])
def wine_category(to_predict):
    return wine_pipeline.classify([to_predict])[0]

@app.route('/liquor_category/<to_predict>', methods=['GET'])
def liquor_category(to_predict):
    return liquor_pipeline.classify([to_predict])[0]
