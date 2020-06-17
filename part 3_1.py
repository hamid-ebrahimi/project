import numpy as np
import pandas as pd
import hazm
from sklearn.cluster import KMeans
from pyclustering.cluster import kmedoids
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import re
df = pd.read_csv('E:/DS/DATA/nk.csv')
df1=df.iloc[:1000,2:4]
df2=df.iloc[:1000,10]
D=pd.concat((df1,df2),1)
D.dropna()
D1=D[D['title_en']=='MO']
D1=D1.dropna()
D1['advantages'] = D1['advantages'].agg(lambda x:re.sub('[^\w\s]','',x))
D1['advantages'] = D1['advantages'].agg(lambda x:re.sub('r','',x))
vocab=D1['advantages'].str.split(expand=True).stack().value_counts()
M=[]
M.append('دوربین')
M.append('عالی')
M.append('خوب')
M.append('صفحه')
M.append('نمایش')
M.append('کیفیت')
M.append('حافظه')
M.append('قوی')
M.append('قیمت')
M.append('بالا')
M.append('زیبا')
M.append('طراحی')
M.append('کاربری')
M.append('کاربری')
M.append('رم')
M.append('رنگ')
M.append('کارت')
M.append('آپدیت')
M.append('بازی')
M.append('مناسب')
M.append('سرعت')
M.append('سلفی')
M.append('کاربردی')
M.append('پشتیبانی')
M.append('ظاهر')
M.append('ضد')
M.append('شارژ')
M.append('صدای')
M.append('سبک')
M.append('بدنه')
M.append('امکانات')
M.append('قوي')
M.append('زیبایی')


H=D1['advantages'].agg(lambda x:[word for word in x.split() if word in M])
H_comment=H.agg(lambda x:' '.join(x))
H1=H_comment.dropna()
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(H1)
SS=pd.DataFrame(X.toarray())
SS.shape
D1.index=np.arange(0,88)
complete_data=pd.concat((D1.iloc[:,1],SS),1)
n_clusters=5
kmeans=KMeans(n_clusters=n_clusters, random_state=0).fit(complete_data)
labels=kmeans.labels_
average_silhouette=silhouette_score(complete_data,labels)
label=pd.DataFrame(labels)
c=pd.concat((D1.iloc[:,1:3],label),1)

sum(c.iloc[:,2]==0)
#26
sum(c.iloc[:,2]==1)
#17
sum(c.iloc[:,2]==2)
#16
sum(c.iloc[:,2]==3)
#20
sum(c.iloc[:,2]==4)
#9
#بيشرين علت رضايت مربوط به صفحه نمايش و دوربين با کفيت ميباشد



