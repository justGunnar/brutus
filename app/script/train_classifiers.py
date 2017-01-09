from lib.sgd_pipe      import SGDPipe
from sklearn.externals import joblib

training_data_path = '../training_data/'

print 'building brand pipeline'
brand_pipeline = SGDPipe(
    training_data_path + 'items',
    training_data_path + 'brands',
    200
)
joblib.dump(brand_pipeline, '../pkl_jar/brand_pipeline.pkl')

print 'building beer pipeline'
beer_pipeline  = SGDPipe(
    training_data_path + 'beers',
    training_data_path + 'beer_categories'
)
joblib.dump(beer_pipeline, '../pkl_jar/beer_pipeline.pkl')

print 'building wine pipeline'
wine_pipeline  = SGDPipe(
    training_data_path + 'wines',
    training_data_path + 'wine_categories'
)
joblib.dump(wine_pipeline, '../pkl_jar/wine_pipeline.pkl')

print 'building liquor pipeline'
liquor_pipeline  = SGDPipe(
    training_data_path + 'liquors',
    training_data_path + 'liquor_categories'
)
joblib.dump(liquor_pipeline, '../pkl_jar/liquor_pipeline.pkl')