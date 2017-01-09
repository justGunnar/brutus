import pdb
import numpy
import pandas
import MySQLdb
import MySQLdb.cursors
import recordlinkage as rl

#from recordlinkage.datasets import load_febrl4
# dfA, dfB = load_febrl4()

db  = MySQLdb.connect(
    host="localhost",
    user="root",
    db="drizly",
    charset="utf8"
)

items = pandas.read_sql("""
    SELECT
        id,
        master_item_id as m_id,
        name,
        price,
        upc,
        container_qty,
        container_type,
        volume,
        volume_units,
        abv,
        vintage
    FROM
        items
    WHERE
        master_item_id IS NOT NULL
    LIMIT
        1000
""", db)
master_items = pandas.read_sql("""
    SELECT
        id,
        name,
        container_qty,
        container_type,
        volume,
        volume_units,
        abv,
        vintage
    FROM
        master_items
""", db)

# Indexation step
pcl = rl.Pairs(items, master_items)
pairs = pcl.sortedneighbourhood('name')

# Comparison step
compare_cl = rl.Compare(pairs, items, master_items)

compare_cl.string('name', 'name', method='jarowinkler', threshold=0.85, name='name')
compare_cl.exact('container_qty', 'container_qty', name='container_qty')
compare_cl.exact('container_type', 'container_type', name='container_type')
compare_cl.exact('volume', 'volume', name='volume')
compare_cl.exact('volume_units', 'volume_units', name='volume_units')
compare_cl.exact('abv', 'abv', name='abv')
compare_cl.exact('vintage', 'vintage', name='vintage')

# Classification step
matches = compare_cl.vectors[compare_cl.vectors.sum(axis=1) > 3]
pdb.set_trace()
print(len(matches))
