import numpy as np
import pandas as pd
import hazm
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
from pyclustering.cluster import kmedoids
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import re
df = pd.read_csv('E:/DS/DATA/nk.csv')
df1=df.iloc[:1000,2:4]
df2=df.iloc[:1000,11]
D=pd.concat((df1,df2),1)
D.dropna()
D1=D[D['title_en']=='MO']
D1=D1.dropna()
D1['disadvantages'] = D1['disadvantages'].agg(lambda x:re.sub('[^\w\s]','',x))
D1['disadvantages'] = D1['disadvantages'].agg(lambda x:re.sub('r','',x))
vocab=D1['disadvantages'].str.split(expand=True).stack().value_counts()
vocab=vocab[vocab>=2]
H=D1['disadvantages'].agg(lambda x:[word for word in x.split() if word in vocab])
H_comment=H.agg(lambda x:' '.join(x))
H1=H_comment.dropna()
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(H1)
SS=pd.DataFrame(X.toarray())
D1.index=np.arange(0,78)
complete_data=pd.concat((D1.iloc[:,1],SS),1)
n_clusters=5
kmeans=KMeans(n_clusters=n_clusters, random_state=0).fit(complete_data)
labels=kmeans.labels_
average_silhouette=silhouette_score(complete_data,labels)
#0.7159779927928461
label=pd.DataFrame(labels)
c=pd.concat((D1.iloc[:,1:3],label),1)
sum(c.iloc[:,2]==0)
#18
sum(c.iloc[:,2]==1)
#9
sum(c.iloc[:,2]==2)
#13
sum(c.iloc[:,2]==3)
#23
sum(c.iloc[:,2]==4)
#15


#c['disadvantages'][c.iloc[:,2]==3]

#باتري ضعيف و قابليتهاي کم

