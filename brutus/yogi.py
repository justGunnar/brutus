###########################
#
# Yogi
#
###########################
from flask    import Flask
from sgd_pipe import SGDPipe

###########################
#
# Startup
#
###########################
print 'building brand pipeline'
brand_pipeline = SGDPipe(
    '../training_data/items',
    '../training_data/brands',
    500
)

print 'building wine pipeline'
wine_pipeline  = SGDPipe(
    '../training_data/wines',
    '../training_data/wine_categories',
    100
)

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

@app.route('/wine_category/<to_predict>', methods=['GET'])
def wine_category(to_predict):
    return wine_pipeline.classify([to_predict])[0]
