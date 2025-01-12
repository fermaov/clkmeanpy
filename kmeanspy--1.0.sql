DROP FUNCTION IF EXISTS load_table_py(text);

CREATE OR REPLACE FUNCTION load_table_py(
	sql text)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

plpy.execute('drop table if exists clustering.cl_data')
plpy.execute('create table clustering.cl_data as ' + sql)
rs = plpy.execute('SELECT count(*) as total FROM clustering.cl_data')
return 'total records loaded: ' + str(rs[0]['total'])

$BODY$;

ALTER FUNCTION load_table_py(text)
    OWNER TO postgres;
	
DROP FUNCTION IF EXISTS load_file_py(text, text);

CREATE OR REPLACE FUNCTION load_file_py(
	url text,
	delim text)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import pandas as pd

df = pd.read_csv(url)
column_dtypes = df.dtypes
column_data_types = column_dtypes.to_dict()
head = ''
for column_name, data_type in column_data_types.items():
	head = head + ', ' + column_name + ' '
	if data_type == 'object':
		head = head +  'text'
	elif data_type == 'int64':
		head = head +  'integer'
	elif data_type == 'float64':
		head = head +  'double precision'
	elif data_type == 'bool':
		head = head +  'boolean'		
	else:
		head = head +  'text,'
head = 'CREATE TABLE clustering.cl_data (' + head[1:] + ')'
plpy.execute('drop table if exists clustering.cl_data')
plpy.execute(head)
sql = "COPY clustering.cl_data FROM '" + url + "' WITH CSV HEADER DELIMITER '" + delim + "' QUOTE " + '\'"\''
plpy.execute(sql)
rs = plpy.execute('SELECT count(*) as total FROM clustering.cl_data')
return 'total records loaded: ' + str(rs[0]['total'])

$BODY$;

ALTER FUNCTION load_file_py(text, text)
    OWNER TO postgres;
	
	
DROP FUNCTION IF EXISTS preprocessing_py();

CREATE OR REPLACE FUNCTION preprocessing_py(
	)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import os
import pandas as pd
import numpy as np
import pickle
import tempfile
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rs = plpy.execute('SELECT * FROM clustering.cl_data')
dat = pd.DataFrame(rs[:])

nom_columns_num = dat.select_dtypes('number').columns
nom_columns_cat = dat.select_dtypes('object').columns

scaler = StandardScaler()
dat_encoded = pd.DataFrame()
if len(nom_columns_num) > 0:  
  dat_encoded = pd.DataFrame(scaler.fit_transform(dat[nom_columns_num]), columns=nom_columns_num)
 
encoder = OneHotEncoder(sparse_output=False)
if len(nom_columns_cat) > 0:  
  cat_encoded = encoder.fit_transform(dat[nom_columns_cat])
  dat_encoded[encoder.get_feature_names_out(nom_columns_cat)] = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(nom_columns_cat))

head = ''
with open('scaler.pickle', 'wb') as f:
	pickle.dump(scaler, f)
	
with open('encoder.pickle', 'wb') as f:
	pickle.dump(encoder, f)

for column in dat_encoded.columns:
	head = head + ', "' + column + '" double precision'
	
head = 'CREATE TABLE clustering.cl_data_pre (' + head[1:] + ')'
		
plpy.execute('drop table if exists clustering.cl_data_pre')
plpy.execute(head)

temp_dir = os.path.expanduser(tempfile.gettempdir())
dat_encoded.to_csv(os.path.join(temp_dir,'clustering.cl_data_pre.csv'),sep=';',index=False)
plpy.execute("COPY clustering.cl_data_pre FROM '" + os.path.join(temp_dir,'clustering.cl_data_pre.csv') + "' WITH DELIMITER ';' HEADER")

rs = plpy.execute('SELECT count(*) as total FROM clustering.cl_data_pre')
return 'Processed records: ' + str(rs[0]['total'])

$BODY$;

ALTER FUNCTION preprocessing_py()
    OWNER TO postgres;
	
	
DROP FUNCTION IF EXISTS kmeans_py(integer, text, integer, text, integer, double precision, integer);

CREATE OR REPLACE FUNCTION kmeans_py(
	n_clusters integer,
	init text DEFAULT 'k-means++'::text,
	n_init integer DEFAULT '-1'::integer,
	algorithm text DEFAULT 'lloyd'::text,
	max_iter integer DEFAULT 300,
	tol double precision DEFAULT 0.0001,
	random_state integer DEFAULT '-1'::integer)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import os
import pandas as pd
import numpy as np
import tempfile
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import datetime

rs = plpy.execute('SELECT * FROM clustering.cl_data_pre')
dat_encoded = pd.DataFrame(rs[:])

cad_n_init = n_init
if n_init == -1:
	cad_n_init = 'auto'	
	
cad_random_state = random_state	
if random_state == -1:
	cad_random_state = None

milliseconds1 = int(datetime.datetime.now().timestamp() * 1000)
k_means = KMeans(n_clusters=n_clusters,init=init,n_init=cad_n_init,algorithm=algorithm,max_iter=max_iter,tol=tol,random_state=cad_random_state)
k_means.fit(dat_encoded)
milliseconds2 = int(datetime.datetime.now().timestamp() * 1000)
timefit = (milliseconds2-milliseconds1)/1000
	
with open('kmeans.pickle', 'wb') as f:
	pickle.dump(k_means, f)

kmean_data = 'inertia: ' + str(round(k_means.inertia_,2)) + ', n_iter: ' + str(k_means.n_iter_) + ', time_fit: ' + str(timefit)
return kmean_data

$BODY$;

ALTER FUNCTION kmeans_py(integer, text, integer, text, integer, double precision, integer)
    OWNER TO postgres;
	

DROP FUNCTION IF EXISTS result_py();

CREATE OR REPLACE FUNCTION result_py(
	)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import pickle
import os
import tempfile
import pandas as pd

with open('kmeans.pickle', 'rb') as f:
  k_means = pickle.load(f)
  
rs = plpy.execute('SELECT * FROM clustering.cl_data')
dat = pd.DataFrame(rs[:])

dat['cluster'] = k_means.labels_
plpy.execute('drop table if exists clustering.cl_result')
plpy.execute('create table clustering.cl_result as SELECT * FROM clustering.cl_data limit 0')
plpy.execute('ALTER TABLE IF EXISTS clustering.cl_result ADD COLUMN cluster integer')
temp_dir = os.path.expanduser(tempfile.gettempdir())
temp_dir = os.path.expanduser(tempfile.gettempdir())
dat.to_csv(os.path.join(temp_dir,'clustering.cl_result.csv'),sep=';',index=False)
plpy.execute("COPY clustering.cl_result FROM '" + os.path.join(temp_dir,'clustering.cl_result.csv') + "' WITH DELIMITER ';' HEADER")

return 'results in table: clustering.cl_result'

$BODY$;

ALTER FUNCTION result_py()
    OWNER TO postgres;
	
	
DROP FUNCTION IF EXISTS centroids_py();

CREATE OR REPLACE FUNCTION centroids_py(
	)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import pickle
import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

with open('scaler.pickle', 'rb') as f:
  scaler = pickle.load(f)
  
with open('encoder.pickle', 'rb') as f:
  encoder = pickle.load(f)
  
with open('kmeans.pickle', 'rb') as f:
  k_means = pickle.load(f)
	
rs = plpy.execute('SELECT * FROM clustering.cl_data')
dat = pd.DataFrame(rs[:])

nom_columns_num = dat.select_dtypes('number').columns
nom_columns_cat = dat.select_dtypes('object').columns

centroids = pd.DataFrame()
if len(nom_columns_num) > 0:
  centroids = pd.DataFrame(scaler.inverse_transform(k_means.cluster_centers_[:, :nom_columns_num.size]), columns=nom_columns_num)

if len(nom_columns_cat) > 0:
  centroids[nom_columns_cat] = pd.DataFrame(encoder.inverse_transform(k_means.cluster_centers_[:, nom_columns_num.size:]), columns=nom_columns_cat)

sql_tb = ''
for col in nom_columns_num:
	sql_tb = sql_tb + col + ' double precision,'
for col in nom_columns_cat:
	sql_tb = sql_tb + col + ' text,'
	
plpy.execute('drop table if exists clustering.cl_centroids')
sql_tb = 'CREATE TABLE clustering.cl_centroids(' + sql_tb[:-1] + ')'
centroids['cluster'] = range(len(centroids))
plpy.execute(sql_tb)
plpy.execute('ALTER TABLE IF EXISTS clustering.cl_centroids ADD COLUMN cluster integer')

temp_dir = os.path.expanduser(tempfile.gettempdir())
centroids.to_csv(os.path.join(temp_dir,'clustering.cl_centroids.csv'),sep=';',index=False)
plpy.execute("COPY clustering.cl_centroids FROM '" + os.path.join(temp_dir,'clustering.cl_centroids.csv') + "' WITH DELIMITER ';' HEADER")

return 'results in table: clustering.cl_centroids'

$BODY$;

ALTER FUNCTION centroids_py()
    OWNER TO postgres;
	
	
DROP FUNCTION IF EXISTS summary_py();

CREATE OR REPLACE FUNCTION summary_py(
	)
    RETURNS TABLE(cluster integer, count integer, percentage double precision) 
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
    ROWS 1000

AS $BODY$

import pickle
import os
import tempfile
import pandas as pd

with open('kmeans.pickle', 'rb') as f:
  k_means = pickle.load(f)
n = len(k_means.cluster_centers_)

labels = pd.DataFrame()
labels['cluster'] = pd.DataFrame(k_means.labels_)

plpy.execute('drop table if exists clustering.cl_summary')
plpy.execute('CREATE TABLE clustering.cl_summary(cluster integer,count integer,percentage double precision)')

summary = pd.DataFrame()
summary['cluster'] = range(n)
summary['count'] = labels.groupby(['cluster'])['cluster'].count()
total = summary['count'].sum()
summary['percentage'] = summary['count'] / total * 100
summary['percentage'] = summary['percentage'].round(2)

temp_dir = os.path.expanduser(tempfile.gettempdir())
summary.to_csv(os.path.join(temp_dir,'clustering.cl_summary.csv'),sep=';',index=False)
plpy.execute("COPY clustering.cl_summary FROM '" + os.path.join(temp_dir,'clustering.cl_summary.csv') + "' WITH DELIMITER ';' HEADER")
rs = plpy.execute('select * from clustering.cl_summary')
return rs

$BODY$;

ALTER FUNCTION summary_py()
    OWNER TO postgres;
	
	
DROP FUNCTION IF EXISTS inertia_py();

CREATE OR REPLACE FUNCTION inertia_py(
	)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import pickle

with open('kmeans.pickle', 'rb') as f:
  k_means = pickle.load(f)
  
kmean_data = 'inertia: ' + str(round(k_means.inertia_,2)) + ', n_iter: ' + str(k_means.n_iter_)
return kmean_data

$BODY$;

ALTER FUNCTION inertia_py()
    OWNER TO postgres;
	

DROP FUNCTION IF EXISTS elbow_py(integer, text, integer, text, integer, double precision, integer);

CREATE OR REPLACE FUNCTION elbow_py(
	n_max integer DEFAULT 10,
	init text DEFAULT 'k-means++'::text,
	n_init integer DEFAULT '-1'::integer,
	algorithm text DEFAULT 'lloyd'::text,
	max_iter integer DEFAULT 300,
	tol double precision DEFAULT 0.0001,
	random_state integer DEFAULT '-1'::integer)
    RETURNS text
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

cad_n_init = n_init
if n_init == -1:
	cad_n_init = 'auto'	
	
cad_random_state = random_state	
if random_state == -1:
	cad_random_state = None
	
rs = plpy.execute('SELECT * FROM clustering.cl_data_pre')
dat_encoded = pd.DataFrame(rs[:])
plpy.execute('drop table if exists clustering.cl_elbow')
plpy.execute('CREATE TABLE clustering.cl_elbow(k integer,inertia double precision)')

sqlstr = 'INSERT INTO clustering.cl_elbow VALUES '
for i in range(1, n_max + 1, 1):
	k_means = KMeans(n_clusters=i,init=init,n_init=cad_n_init,algorithm=algorithm,max_iter=max_iter,tol=tol,random_state=cad_random_state)
	k_means.fit(dat_encoded)
	sqlstr = sqlstr + '(' + str(i) + ',' + str(round(k_means.inertia_,2)) + '),'
plpy.execute(sqlstr[:-1])
return 'results in table: clustering.cl_elbow'

$BODY$;

ALTER FUNCTION elbow_py(integer, text, integer, text, integer, double precision, integer)
    OWNER TO postgres;
	
	
DROP FUNCTION IF EXISTS silhouette_py(integer, text, integer, text, integer, double precision, integer);

CREATE OR REPLACE FUNCTION silhouette_py(
	n_max integer DEFAULT 10,
	init text DEFAULT 'k-means++'::text,
	n_init integer DEFAULT '-1'::integer,
	algorithm text DEFAULT 'lloyd'::text,
	max_iter integer DEFAULT 300,
	tol double precision DEFAULT 0.0001,
	random_state integer DEFAULT '-1'::integer)
    RETURNS integer
    LANGUAGE 'plpython3u'
    COST 100
    VOLATILE PARALLEL UNSAFE
AS $BODY$

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cad_n_init = n_init
if n_init == -1:
	cad_n_init = 'auto'	
	
cad_random_state = random_state	
if random_state == -1:
	cad_random_state = None

rs = plpy.execute('SELECT * FROM clustering.cl_data_pre')
dat_encoded = pd.DataFrame(rs[:])
plpy.execute('drop table if exists clustering.cl_silhouette')
plpy.execute('CREATE TABLE clustering.cl_silhouette(k integer,silhouette_index double precision)')

sqlstr = 'INSERT INTO clustering.cl_silhouette VALUES '

score = 0
score_max = 0
k_value = 2
for i in range(2, n_max + 1, 1):
	k_means = KMeans(n_clusters=i,init=init,n_init=cad_n_init,algorithm=algorithm,max_iter=max_iter,tol=tol,random_state=cad_random_state)
	k_means.fit(dat_encoded)
	score = silhouette_score(dat_encoded, k_means.labels_)
	sqlstr = sqlstr + '(' + str(i) + ',' + str(score) + '),'
	if score > score_max:
		score_max = score
		k_value = i		
plpy.execute(sqlstr[:-1])
return k_value

$BODY$;

ALTER FUNCTION silhouette_py(integer, text, integer, text, integer, double precision, integer)
    OWNER TO postgres;
