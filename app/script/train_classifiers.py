from yogi_utils.sgd_pipe import SGDPipe
from sklearn.externals   import joblib

training_data_path = '../training_data/'

print 'building top level category classifier'
tlc_pipeline = SGDPipe(
    training_data_path + 'tlc_items',
    training_data_path + 'top_level_category'
)
joblib.dump(tlc_pipeline, '../pkl_jar/tlc_pipeline.pkl')

print 'building beer brand pipeline'
beer_brand_pipeline = SGDPipe(
    training_data_path + 'beers',
    training_data_path + 'beer_brands'
)
joblib.dump(beer_brand_pipeline, '../pkl_jar/beer_brand_pipeline.pkl')

print 'building wine brand pipeline'
wine_brand_pipeline = SGDPipe(
    training_data_path + 'wines',
    training_data_path + 'wine_brands'
)
joblib.dump(wine_brand_pipeline, '../pkl_jar/wine_brand_pipeline.pkl')

print 'building liquor brand pipeline'
liquor_brand_pipeline = SGDPipe(
    training_data_path + 'liquors',
    training_data_path + 'liquor_brands'
)
joblib.dump(liquor_brand_pipeline, '../pkl_jar/liquor_brand_pipeline.pkl')

print 'building beer leaf category pipeline'
beer_pipeline  = SGDPipe(
    training_data_path + 'beers',
    training_data_path + 'beer_categories'
)
joblib.dump(beer_pipeline, '../pkl_jar/beer_pipeline.pkl')

print 'building wine leaf category pipeline'
wine_pipeline  = SGDPipe(
    training_data_path + 'wines',
    training_data_path + 'wine_categories'
)
joblib.dump(wine_pipeline, '../pkl_jar/wine_pipeline.pkl')

print 'building liquor leaf category pipeline'
liquor_pipeline  = SGDPipe(
    training_data_path + 'liquors',
    training_data_path + 'liquor_categories'
)
joblib.dump(liquor_pipeline, '../pkl_jar/liquor_pipeline.pkl')