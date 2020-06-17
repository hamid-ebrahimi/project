import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyclustering.cluster import kmedoids
import pandas as pd
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import re
import hazm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from kmodes.kmodes import KModes
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance

df = pd.read_csv('E:/DS/DATA/nk.csv')
df1=df.iloc[:,[0,1]]
df2=df1.iloc[0:1000,:]
df2=df2.dropna()
stop=hazm.stopwords_list
stops=[stop()]

stops.append('میلی')
stops.append('لیتر')
stops.append('مدل')
stops.append('ميلي')
stops.append('حجم')
stops.append('سري')
stops.append('ليتر')
#stops.append('مردانه')
#stops.append('زنانه')
stops.append('مناسب')
stops.append('برای')
stops.append('و')
stops.append('طرح')
stops.append('به ')
stops.append('به')
stops.append('کد')
stops.append('ظرفیت')
stops.append('کننده')
stops.append('با')
stops.append('2')
stops.append('100')
stops.append('بی')
stops.append('مخصوص')
stops.append('1')
stops.append('دو')
stops.append('3')
stops.append('4')
stops.append('ال')
stops.append('ای')
stops.append('تی')
stops.append('سری')
#stops.append('ست')
stops.append('8')
stops.append('64')
stops.append('15')
stops.append('45')
stops.append('9')
stops.append('120')
stops.append('250')
stops.append('اس')
stops.append('های')
stops.append('5')
stops.append('20')
stops.append('131')
stops.append('30')
stops.append('360')
stops.append('12')
stops.append('7')
stops.append('25')
stops.append('همه')
stops.append('هوم')
stops.append('روی')
stops.append('مقدار')
stops.append('Super')
stops.append('انرجایزر')
stops.append('کانن')
stops.append('اثر')
stops.append('می ')
stops.append('تحت')
stops.append('18')
stops.append('انکر')
stops.append('اف')
stops.append('یو')
stops.append('هیوندای')
stops.append('نیا')
stops.append('10')
stops.append('چند')
stops.append('Plus')
stops.append('ان')
stops.append('طبیعی')
stops.append('لنوو')
stops.append('راحت')
stops.append('امپر')
stops.append('سن')
stops.append('پک')
stops.append('از')
stops.append('ایکس')
stops.append('آنر')
stops.append('دستی')
stops.append('وی')
stops.append('ام')
stops.append('کن')
stops.append('اچ')
stops.append('سر')
#stops.append('شیاومی')
#stops.append('ایسوس')
#stops.append('پاناسونیک')
stops.append('عددی')
stops.append('سه')
stops.append('ضد')
stops.append('حمل')
stops.append('جی')
#stops.append('شیائومی')
#stops.append('هوآوی')
stops.append('دی')
#stops.append('سامسونگ')
stops.append('بدون')
stops.append('راهی')
stops.append('مایدیا')
stops.append('اداری')
stops.append('آ')
stops.append('کامل')
stops.append('میدسان')
stops.append('زیر')
stops.append('پیاده')
stops.append('آوی')
stops.append('16')
stops.append('microSDXC')
stops.append('آی')
stops.append('Galaxy')
stops.append('Mi')
#stops.append('USB')
stops.append('قابل')
#stops.append('فیلیپس')
#stops.append('سونی')
#stops.append('بلوتوثی')
#stops.append('دیتا')
#stops.append('واون')
stops.append('تسکو')
stops.append('وان')
stops.append('دهنده')
#stops.append('گوشتی')
stops.append('مور')
stops.append('می')
stops.append('ساز')
stops.append('استریو')
#stops.append('فلر')
#stops.append('نوکیا')
#stops.append('وسترن')
stops.append('روموس')
#stops.append('آب')
#stops.append('چکشی')
stops.append('بعدی')
#stops.append('رونیکس')
#stops.append('کنوود')
stops.append('پشتی')
stops.append('ایکسسل')
stops.append('پارس')
#stops.append('آدیداس')
stops.append('Tab')
stops.append('سایز')
#stops.append('کاسیو')
stops.append('مایکروسافت')
#stops.append('microUSB')
stops.append('دبلیو')
#stops.append('اپل')
#stops.append('فندکی')
#stops.append('سیلیکون')
stops.append('پایونیر')
#stops.append('ریمل')
#stops.append('زبان')
stops.append('پاور')
stops.append('دل')
#stops.append('خزر')


df2['word']=''
df2['product']=''
df2['product']=df2['product_title'].agg(lambda x:x.split(" "))
for i in range(0,1000):
    df2['word'][i]=df2['product'][i][0:5]
    

df2['word']=df2['word'].agg(lambda x:' '.join(x))
S=df2['word'].agg(lambda x:[word for word in x.split() if word not in stops])
products=S.agg(lambda x:' '.join(x))
product_count=products.str.split(expand=True).stack().value_counts()
vocab=product_count[product_count>=6]

s1=set(vocab.index)
m=pd.Series(list(s1))
df3=df2.iloc[:,0:2]
allfeatures=np.zeros((df3.shape[0],m.shape[0]))
for i in np.arange(m.shape[0]):
    allfeatures[df3['product_title'].agg(lambda x:sum([y==m[i] for y in x.split()])>0),i]=1

df4=df3.iloc[:,0:1]
Complete_data=pd.concat([df4,pd.DataFrame(allfeatures)],1)



cm=Complete_data.values.tolist()
initial_medoids=[1,2,3,4,5,6,7,8,9]
#initial_medoids=[1,2,3,4,5]

#metric=distance.euclidean
metric = distance_metric(type_metric.EUCLIDEAN_SQUARE)
#metric = DistanceMetric.get_metric('')



kmedoids_instance = kmedoids.kmedoids(cm, initial_medoids, metric=metric)
kmedoids_instance.process()
cl=kmedoids_instance.get_clusters()


zero=pd.DataFrame(cl[0])
zero['label']=0
one=pd.DataFrame(cl[1])
one['label']=1
two=pd.DataFrame(cl[2])
two['label']=2
three=pd.DataFrame(cl[3])
three['label']=3
four=pd.DataFrame(cl[4])
four['label']=4
five=pd.DataFrame(cl[5])
five['label']=5
six=pd.DataFrame(cl[6])
six['label']=6
seven=pd.DataFrame(cl[7])
seven['label']=7
eight=pd.DataFrame(cl[8])
eight['label']=8
cluster_lablel=pd.concat([zero,one,two,three,four,five,six,seven,eight],0)
#cluster_lablel=pd.concat([zero,one,two,three,four])

cluster_lablel.index=np.arange(0,1000)
cluster_lablel.columns=['index','lable']


cluster_lablel=cluster_lablel.sort_values('index',ascending=True)
cluster_lablel.index=np.arange(0,1000)
df5=pd.concat((df3.iloc[:,0],cluster_lablel.iloc[:,1]),1)

df6=pd.concat((df5,df3.iloc[:,1]),  1)
df6[df5.iloc[:,1]==0]

















