# Customer-segmentation-and-analytics
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:27:25 2020

@author: Faheem
"""

#%% Loading required modules
from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import cluster
from order_cluster import order_cluster
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering

#%%
# Reading data set 
#service_data=pd.read_csv('CDNOW_master.txt', delimiter='\s+' ) #or
tx_3m=pd.read_csv('CDNOW_master.txt', delim_whitespace=True ,header=None, names=['CustomerID', 'InvoiceDate', 'Quantity', 'Transaction_in_dollars'])

tx_3m['UnitPrice']= tx_3m['Transaction_in_dollars']/tx_3m['Quantity']

#%%
#create tx_user for assigning clustering
tx_user = pd.DataFrame(tx_3m['CustomerID'].unique())
tx_user.columns = ['CustomerID']
#tx_3m['InvoiceDate'] = pd.to_datetime(tx_3m['InvoiceDate'], errors='coerce')
A= list(tx_3m['InvoiceDate'])
df = pd.DataFrame({'InvoiceDate': A})
B = df[['InvoiceDate']].applymap(str).applymap(lambda s: "{}-{}-{}".format(s[4:6],s[6:], s[0:4]))
tx_3m = tx_3m.drop(['InvoiceDate'], axis = 1) 
tx_3m['InvoiceDate']=B
tx_3m['InvoiceDate']= pd.to_datetime(tx_3m['InvoiceDate']) 

#%%
#calculate recency score
tx_max_purchase = tx_3m.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

#%%
#calcuate frequency score
tx_frequency = tx_3m.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
#
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)

#%%
#calcuate revenue score
tx_3m['Revenue'] = (tx_3m['UnitPrice'] * tx_3m['Quantity'])
tx_revenue = tx_3m.groupby('CustomerID').Revenue.sum().reset_index()
tx_revenue['Revenue']/=tx_frequency['Frequency']
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

#%%
dendrogram = dendrogram(linkage(tx_user[list(['Recency','Frequency','Revenue'])], method='ward'))
#%%
#overall scoring
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 

#%%
# Using clustering approach
kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_user[list(['Recency','Frequency','Revenue'])])
tx_user['Cluster'] = kmeans.predict(tx_user[list(['Recency','Frequency','Revenue'])])

#%%
# Using clustering approach on normalized data
x=tx_user[list(['Recency','Frequency','Revenue'])]
X = StandardScaler().fit_transform(x)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
tx_user['ClusterStandard'] = kmeans.predict(X)
