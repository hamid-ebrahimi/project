import numpy as np
import pandas as pd
import hazm
import re
from sklearn.cluster import KMeans
from pyclustering.cluster import kmedoids
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

df = pd.read_csv('E:/DS/DATA/nk.csv')
df1=df.iloc[:,3]
#df2=df.iloc[:,7]
df3=df.iloc[:,9]
#d=pd.concat((df1,df2),1)
D=pd.concat((df1,df3),1)
#D['comment'].agg(lambda x: x.str.split(" "))
#D['comment']=D['comment'].agg(lambda x:' '.join(x)
D=D.dropna()
D['comment'] = D['comment'].agg(lambda x:re.sub('[^\w\s]','',x))
B=D.iloc[:1000,:]
stop=hazm.stopwords_list
stops=[stop()]
#vectorizer.fit_transform()

stops.append('و')
stops.append('که')
stops.append('از')
stops.append('fi')
stops.append('به')
stops.append('')
stops.append('به')
stops.append('این')
stops.append('من')
stops.append('با')
stops.append('هم')
stops.append('داره')
stops.append('در')
stops.append('براي')
stops.append('برای')
stops.append('استفاده')
stops.append('هست')
stops.append('بود')
stops.append('خریدم')
stops.append('میشه')
stops.append('های')
stops.append('یه')
stops.append('واقعا')
stops.append('کردم')
stops.append('میلی')
stops.append('لیتر')
stops.append('مدل')
stops.append('ميلي')
stops.append('حجم')
stops.append('سري')
stops.append('ليتر')
stops.append('مردانه')
stops.append('زنانه')
#stops.append('مناسب')
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
stops.append('ست')
stops.append('8')
stops.append('64')
stops.append('15')
stops.append('45')
stops.append('9')
stops.append('120')
stops.append('250')
stops.append('اس')
stops.append('های')
stops.append('رو')
stops.append('ولی')
stops.append('تو')
stops.append('تا')
stops.append('می')
stops.append('یک')
stops.append('فقط')
stops.append('قیمت')
stops.append('میکنم')
stops.append('خرید')
stops.append('نداره')
stops.append('است')
stops.append('دیجی')
stops.append('شده')
stops.append('کالا')
stops.append('اما')
stops.append('میکنه')


#stops.append('نیست')
stops.append('ها')
stops.append('کار')
stops.append('بعد')
stops.append('محصول')
stops.append('کنید')
stops.append('گوشی')
stops.append('روی')
stops.append('چون')
stops.append('باید')
stops.append('را')
stops.append('کل')
stops.append('دیگه')
stops.append('باشه')
stops.append('dh')
stops.append('يا')
stops.append('شارژ')
stops.append('یا ')
stops.append('اینکه')
stops.append('همه')
stops.append('شد ')
stops.append('رنگ')
stops.append('کنم')
stops.append('قیمتش')
stops.append('یا')
stops.append('سلام ')
stops.append('سلام')
stops.append('جنس')
stops.append('هر')
stops.append('شد ')
stops.append('شد')
stops.append('اگه')
stops.append('توی')
stops.append('چند')
stops.append('وقتی')
stops.append('دارم')


stops.append('ازش')
stops.append('کرد')
stops.append('اون')
stops.append('البته')
stops.append('دستگاه')
stops.append('ساعت')
stops.append('کم')
stops.append('کردن ')
stops.append('مثل')
stops.append('دوستان')

stops.append('اگر')
stops.append('کردن')
stops.append('بیشتر')
stops.append('یکی')
stops.append('الان')
stops.append('نه')
stops.append('زیاد')
stops.append('دستم')
stops.append('هستش')
stops.append('بار')
stops.append('رسید')
stops.append('میده')
stops.append('ی')
stops.append('تر')
stops.append('دست')


#stops.append('نمیشه')
stops.append('هستم')
stops.append('حتی')
stops.append('بودم')
stops.append('بازی')
stops.append('همین')

stops.append('داشت')
stops.append('روز')
stops.append('توجه')
stops.append('بشه')
stops.append('نمی')

stops.append('بسته')
stops.append('بوی')
stops.append('کنه')
stops.append('هیچ')
stops.append('ماه')
stops.append('گرفتم')
stops.append('کاملا')
stops.append('عکس')
stops.append('باز')
stops.append('کاملا')
stops.append('حتما')
stops.append('اول')
stops.append('ک')
stops.append('ام')
stops.append('سال')
stops.append('بگم')
stops.append('واسه')
stops.append('اندازه')
stops.append('پیش')
stops.append('خریدش')
stops.append('بدون')
stops.append('چیزی')
stops.append('دادم')
stops.append('نظرم')
stops.append('سه')
stops.append('صدای')
stops.append('ماشین')
stops.append('بودن')
stops.append('خود')
stops.append('دارد')
stops.append('داخل')
stops.append('پایین')
stops.append('برند')
stops.append('کلا')
stops.append('کمی')
stops.append('نصب')
stops.append('داشته')
stops.append('ساخت')
stops.append('داشتم')
stops.append('شما')
stops.append('آن')
stops.append('کرده')
stops.append('سفارش')
stops.append('طراحی')
stops.append('جا')
stops.append('فکر')
stops.append('صدا')
stops.append('کیف')
stops.append('آب')
stops.append('یکم ')
stops.append('یکم')
stops.append('یکم')
stops.append('صفحه')
stops.append('باهاش')
stops.append('حدود')
stops.append('شدم')
stops.append('سایز')
stops.append('نبود')
#stops.append('نرم')
stops.append('اینه')
stops.append('پس')
stops.append('کیفیتش')
stops.append('نمیکنه')
stops.append('شدن')
stops.append('سرعت')
stops.append('کفش')
stops.append('توصیه')
stops.append('روشن')
stops.append('پوست')
stops.append('جنسش')
stops.append('بندی')
stops.append('پخش')
stops.append('مورد')
stops.append('کابل')
stops.append('خودم')
stops.append('جای')
stops.append('مشکلی')
stops.append('زود')
stops.append('دسته')
stops.append('نکنید')
stops.append('حالت')
stops.append('خودش')
stops.append('امروز')
stops.append('صورت')
stops.append('باعث')
stops.append('بر')
stops.append('تنها')
stops.append('شک')
stops.append('اینو')
stops.append('سر')
stops.append('خریداری')
stops.append('نوشته')
stops.append('روش')
stops.append('اش')
stops.append('زیادی')
stops.append('دیدم')
stops.append('موقع ')
stops.append('موقع')
stops.append('باشید')
stops.append('هفته')
stops.append('بازار')

#stops.append('نمیکنم')
stops.append('دور')
stops.append('عنوان')
stops.append('باتری')
stops.append('تقریبا')
stops.append('وصل')
stops.append('عطر')
stops.append('بدنه')
stops.append('همون')
#stops.append('درست')
stops.append('تمام')
stops.append('دوربین')
stops.append('تهیه')
stops.append('درد')
stops.append('قسمت')
stops.append('هاش')
stops.append('میاد')
stops.append('زیر')
stops.append('شاید')
stops.append('یعنی')
stops.append('مدت')
stops.append('لحاظ')
stops.append('بیرون')
stops.append('دی')
stops.append('قرار')
stops.append('رنگش')
stops.append('دارن')
stops.append('پیدا')
stops.append('پر')
stops.append('قبل')
stops.append('برا')
stops.append('ماندگاری')
stops.append('انتخاب')
stops.append('ب')
stops.append('دیجیکالا')
stops.append('بو')
stops.append('ظاهر')
stops.append('نور')
stops.append('پول')
stops.append('زمان')
stops.append('جعبه')
stops.append('ندارد')
stops.append('بقیه')
stops.append('حالا')
stops.append('قدرت')
stops.append('کتاب')
stops.append('بوده')
stops.append('پاور')
stops.append('راه')
stops.append('صداش')
stops.append('قبلا')
stops.append('ساله')
stops.append('موتور')
stops.append('حرف')
stops.append('باشد')
stops.append('انجام')
stops.append('حال')
stops.append('چه')
stops.append('ایرانی')
stops.append('روغن')
stops.append('دکمه')
stops.append('هدفون')
stops.append('بهش')
stops.append('موجود')
stops.append('همیشه')
stops.append('بعضی')
stops.append('گوش')
stops.append('کرم')
stops.append('هزینه')
stops.append('داده')
stops.append('تصویر')
stops.append('بوش ')
stops.append('بوش')
stops.append('نشده')
stops.append('میخوره')
stops.append('تولید')
stops.append('بچه')
stops.append('نگه')
stops.append('بند')
stops.append('جواب')
stops.append('هستن')
stops.append('اصلی')
stops.append('میکردم')
stops.append('حس')
stops.append('باطری')
stops.append('متوجه')
stops.append('نمیخوره')
stops.append('شود')
stops.append('نیاز')
stops.append('سایت')
stops.append('نشون')
stops.append('دقیقه')
stops.append('برام')
stops.append('جی')
stops.append('کلی')
stops.append('نکته')
stops.append('بخاطر')
stops.append('فشار')
stops.append('میشد')
stops.append('پشیمون')
stops.append('چسب')
stops.append('دادن')
stops.append('خاطر')
stops.append('نوع')
stops.append('هایی')
stops.append('توش')
stops.append('شیشه')
stops.append('ما')
stops.append('پاک')
stops.append('کنین')
stops.append('کسی')
stops.append('۲')
stops.append('خط')
stops.append('اومد')
stops.append('قابلیت')
stops.append('امیدوارم')
stops.append('بنده')
stops.append('شبیه')
stops.append('نکردم')
stops.append('قاب')
stops.append('گرم')
stops.append('انگار')
stops.append('دوباره')
stops.append('طول')
stops.append('کوچیک')
stops.append('تره')
stops.append('دقیقا')
stops.append('شه')
stops.append('پشت')
stops.append('ماهه')
stops.append('مقایسه')
stops.append('مو')
stops.append('کوچک')
stops.append('وقت')
stops.append('دنبال')
stops.append('کنی')
stops.append('هستند')
stops.append('جدا')
stops.append('داد')
stops.append('ضد')
stops.append('دلیل')
stops.append('میگیره')
stops.append('میتونید')
stops.append('تست')
stops.append('ضمن')
stops.append('کاور ')
stops.append('کاور')
stops.append('داغ')
stops.append('کاور')
stops.append('عوض')
stops.append('چراغ')
stops.append('پایه')
stops.append('غیر')
stops.append('کارت')
stops.append('مشخصات')
stops.append('داخلش')
stops.append('امتحان')
stops.append('نداشته')
stops.append('اولین')
stops.append('هدیه')
stops.append('جمع')
stops.append('دارید')
stops.append('اونم')
stops.append('بلند')
stops.append('چشم')
stops.append('وزن')
stops.append('تحویل')
stops.append('بنظرم')
stops.append('چی')
stops.append('میزنه')
stops.append('نظرات')
stops.append('گلس')
stops.append('مهم')
stops.append('همراه')
stops.append('ترین')
stops.append('پلاستیک')
stops.append('خونه')
stops.append('نزدیک')
stops.append('مقدار')
stops.append('برنامه')
stops.append('خاصی')
stops.append('شدت')
stops.append('میره')
stops.append('ارسال')
stops.append('فرق')
stops.append('مشکی')
stops.append('چرا')
stops.append('۳')
stops.append('دارند')
stops.append('برابر')
stops.append('گیر')
stops.append('میدم')
stops.append('سامسونگ')
stops.append('هارد')
stops.append('براش')
stops.append('سفید')
stops.append('ش')
stops.append('پیچ')
stops.append('5')
stops.append('گفتن')
stops.append('سیستم')
stops.append('میگن')
stops.append('سی')
stops.append('تنظیم')
stops.append('ضربه')
stops.append('موس')
stops.append('درصد')
stops.append('احساس')
stops.append('نمایش')
stops.append('اسپیکر')
stops.append('پا')
stops.append('بالای')
stops.append('دوتا')
stops.append('میتونم')
stops.append('بالای')
stops.append('زیاده')
stops.append('درجه')
stops.append('نسبتا')
stops.append('زده')
stops.append('فروش')
stops.append('نیم')
stops.append('مثلا')
stops.append('کالای')
stops.append('دارای')
stops.append('همچنین')
stops.append('میشود')
stops.append('میکنید')
stops.append('کنار')
stops.append('لطفا')
stops.append('نیس')
stops.append('زدم')
stops.append('حرفه')
stops.append('بانک')
stops.append('هندزفری')
stops.append('قهوه')
stops.append('محافظ')
stops.append('خش')
stops.append('گزینه')
stops.append('پارچه')
stops.append('دوستانی')
stops.append('گفت')
stops.append('لپ')
stops.append('تاپ')
stops.append('قیمتی')
stops.append('تغییر')
stops.append('لباس')
stops.append('کسانی')
stops.append('جوش')
stops.append('بدستم')
stops.append('مشابه')
stops.append('بازم')
stops.append('تماس')
stops.append('حداقل')
stops.append('خورد')
stops.append('تعویض')
stops.append('اتصال')
stops.append('کسایی')
stops.append('خب')
stops.append('مودم')
stops.append('ایران')
stops.append('دیگر')
stops.append('لامپ')
stops.append('ده')
stops.append('خلاصه')
stops.append('تموم')
stops.append('فروشنده')
stops.append('هنگام')
stops.append('خالی')
stops.append('دوم')
stops.append('مختلف')
stops.append('دار')
stops.append('اصل')
stops.append('شکل')
stops.append('اب')
stops.append('چهار')
stops.append('خورده')
stops.append('فلش')
stops.append('دوخت')
stops.append('نازک')
stops.append('ایجاد')
stops.append('پوشش')
stops.append('تلویزیون')
stops.append('فیلم')
stops.append('عدد')
stops.append('ممکنه')
stops.append('بهتون')
stops.append('باد')
stops.append('کف')
stops.append('نقطه')
stops.append('عرض')
stops.append('صورتی')
stops.append('اثر')
stops.append('پلاستیکی')
stops.append('موهای')
stops.append('10')
stops.append('لب')
stops.append('نمونه')
stops.append('لبه')
stops.append('عکسش')
stops.append('زدن')
stops.append('۵')
stops.append('موبایل')
stops.append('بدن')
stops.append('گفته')
stops.append('آدم')
#stops.append('مواد')
#stops.append('افتاد')
stops.append('بابت')
stops.append('امکان')
stops.append('شارژش')
stops.append('توضیحات')
stops.append('سایزش')
stops.append('درب')
stops.append('طرفطرف')
stops.append('طرف')
stops.append('جهت')
stops.append('لازم')
stops.append('افزار')
stops.append('یکبار')
stops.append('زمانی')
stops.append('براتون')
stops.append('میز')
stops.append('کن')
stops.append('۱')
stops.append('کلام')
stops.append('دستی')
stops.append('تیغه')
stops.append('بیس')
stops.append('6')
stops.append('متر')
stops.append('نتیجه')
stops.append('نباید')
stops.append('گفتم')
stops.append('عین')
stops.append('داخلی')
stops.append('گوشیم')
stops.append('پی')
stops.append('کوله')
stops.append('کنند')
stops.append('چندین')
stops.append('خودتون')
stops.append('سطح')
stops.append('خنک')
stops.append('وجه')
stops.append('بدید')
stops.append('شامپو')
stops.append('بهم')
stops.append('ساختش')
stops.append('کلید ')
stops.append('کلید')
stops.append('مدتی')
stops.append('منو')
stops.append('سمت')
stops.append('سوخت')
stops.append('قبول')
stops.append('وارد')
stops.append('طولانی')
stops.append('چرب')
stops.append('قبلی')
stops.append('واقعی')
stops.append('سرویس')
stops.append('گرفت')
stops.append('جایی')
stops.append('شب')
stops.append('ادکلن')
stops.append('دستمال')
stops.append('دهی')
stops.append('گیری')
stops.append('ظرف')
stops.append('میگم')
stops.append('حل')
stops.append('انقدر')
stops.append('ان')
stops.append('ثانیه')
stops.append('اینقدر')
stops.append('طوری')
stops.append('عمر')
stops.append('بیش')
stops.append('برد')
stops.append('ماهی')
stops.append('میخواستم')
stops.append('کنترل')
stops.append('شکست')
stops.append('مجموع')
stops.append('میکند')
stops.append('کنیم')
stops.append('قرمز')
stops.append('خروجی')
stops.append('محل')
stops.append('چاپ')
stops.append('پاوربانک')
stops.append('ترک')
stops.append('پشتیبانی')
stops.append('زنگ')
stops.append('کنده')
stops.append('نگاه')
stops.append('رفت')
stops.append('نصف')
stops.append('گرفتن')
stops.append('جور')
stops.append('شروع')
stops.append('شماره')
stops.append('وگرنه')
stops.append('نمیاد')
stops.append('ظاهری')
stops.append('خودرو')
stops.append('سایر')
stops.append('سایر')
stops.append('میرسه')
stops.append('بلوتوث')
stops.append('میکرد')
stops.append('زمین')
stops.append('اتاق')
stops.append('میکنی')
stops.append('جز')
stops.append('میداره')
stops.append('میزان')
stops.append('میشن')
stops.append('صد')
stops.append('قدیمی')
stops.append('تفاوت')
stops.append('بعدش')
stops.append('نهایت')
stops.append('فن')
stops.append('رنگی')
stops.append('سفت')
stops.append('پوستم')
stops.append('باشین')
stops.append('فاصله')
stops.append('جلو')
stops.append('حساب')
stops.append('انتقال')
stops.append('ریش')
stops.append('رم')
stops.append('خدمت')
stops.append('تیره')
stops.append('گذاشتم')
stops.append('حافظه')
stops.append('برندهای')
stops.append('فعلا')
stops.append('قطعات')
stops.append('بزنید')
stops.append('بیاد')
stops.append('گرافیک')
stops.append('آبی')
stops.append('نقاط')
stops.append('چوب')
stops.append('خریدن')
stops.append('سلیقه')
stops.append('نصبش')
stops.append('فضای')
stops.append('قصد')
stops.append('نکرده')
stops.append('روزه')
stops.append('هوا')
stops.append('دیر')
stops.append('دیروز')
stops.append('کمک')
stops.append('اینجا')
stops.append('هزار')
stops.append('منم')
stops.append('اکثر')
stops.append('میشید')
stops.append('خلاف')
stops.append('فیلتر')
stops.append('زد')
stops.append('خواستم')
stops.append('خدمات')
stops.append('مال')
stops.append('عدم')
stops.append('فک')
stops.append('قفل')
stops.append('اومده')
stops.append('شستشو')
stops.append('چای')
stops.append('روزی')
stops.append('کاربرد')
stops.append('صفر')
stops.append('گاز')
stops.append('نورش')
stops.append('ابعاد')
stops.append('آخر')
stops.append('ابعاد')
stops.append('افزایش')
stops.append('بخوره')
stops.append('کنسول')
stops.append('سوراخ')
stops.append('کادو')
stops.append('اینترنت')
stops.append('مژه ')
stops.append('مژه')
stops.append('فلزی')
stops.append('بگیرم')
stops.append('لنز')
stops.append('سختی')
stops.append('وسایل')
stops.append('کارهای')
stops.append('دید')
stops.append('براق')
stops.append('ذکر')
stops.append('اینا')
stops.append('حرکت')
stops.append('واقع')
stops.append('مچ')
stops.append('برخلاف')
stops.append('قبولی')
stops.append('ارائه')
stops.append('گیره')
stops.append('لایه')
stops.append('لوازم')
stops.append('کردنش')
stops.append('رده')
stops.append('محیط')
stops.append('حدودا')
stops.append('موهام')
stops.append('بدین')
stops.append('گذاشتن')
stops.append('آنتن')
stops.append('نمایندگی')
stops.append('تیز')
stops.append('اسم')
stops.append('پودر')
stops.append('۸')
stops.append('انگشت')
stops.append('ریز')
stops.append('خیلی')
stops.append('نسبت')


S=D['comment'].agg(lambda x:[word for word in x.split() if word not in stops])

comment=S.agg(lambda x:' '.join(x))


#
notsatisfaction=[]
satisfaction=[]

#satisfaction

satisfaction.append('خوب')
satisfaction.append('خوبه')
satisfaction.append('عالی')
satisfaction.append('عالیه')
satisfaction.append('راضی')
satisfaction.append('مناسب')
satisfaction.append('ارزش')
satisfaction.append('راضیم')
satisfaction.append('فوق')
satisfaction.append('راحت')
satisfaction.append('خوبیه')
satisfaction.append('بخرید')
satisfaction.append('شگفت')
satisfaction.append('العاده')
satisfaction.append('انگیز')
satisfaction.append('خوش')
satisfaction.append('بهترین')
satisfaction.append('ممنون')
satisfaction.append('شیک')
satisfaction.append('مناسبه')
satisfaction.append('سبک')
satisfaction.append('سریع')
satisfaction.append('زیبایی ')
satisfaction.append('زیبایی')
satisfaction.append('دقت')
satisfaction.append('ویژه')
satisfaction.append('تمیز')
satisfaction.append('زیباست')
satisfaction.append('مناسبی')
satisfaction.append('تشکر')
satisfaction.append('جالب')
satisfaction.append('قوی')
satisfaction.append('محکم')
satisfaction.append('قشنگ')
satisfaction.append('قشنگه')
satisfaction.append('راحته')
satisfaction.append('خوشم')#
satisfaction.append('لذت')
satisfaction.append('دقیق')
satisfaction.append('ممنونم')
satisfaction.append('خاص')
satisfaction.append('بخرین')
satisfaction.append('مرسی')
satisfaction.append('خوشگله')
satisfaction.append('متوسط')
satisfaction.append('باکیفیت')
satisfaction.append('سبکه')
satisfaction.append('ارزون')
satisfaction.append('شیکه')
satisfaction.append('رضایت')
satisfaction.append('خوشبو')
satisfaction.append('مقاوم')
satisfaction.append('قشنگی')
satisfaction.append('سالم')
satisfaction.append('مفید')
satisfaction.append('جذاب')
satisfaction.append('کاربردیه')
satisfaction.append('موفق')
satisfaction.append('دستتون')
satisfaction.append('خوشگل')
satisfaction.append('جالبه')
satisfaction.append('العادس')
satisfaction.append('ذوق')
satisfaction.append('تنوع')
satisfaction.append('فوقالعاده')
satisfaction.append('العادست')
satisfaction.append('میارزه')
satisfaction.append('لطیف')
satisfaction.append('شیکی')
satisfaction.append('خوشش')
satisfaction.append('مقرون')
satisfaction.append('طبیعیه')
satisfaction.append('شکیل')
satisfaction.append('سپاس')
satisfaction.append('خوشمزه')
satisfaction.append('عاشق')
satisfaction.append('علاقه')
satisfaction.append('محشره')
satisfaction.append('خوشحالم')
satisfaction.append('بهترینه')
satisfaction.append('خوبش')
satisfaction.append('باتشکر')
satisfaction.append('خوبیش')
satisfaction.append('عاشقش')
satisfaction.append('مناسبیه')
satisfaction.append('خوشبختانه')
satisfaction.append('خوشحال')
satisfaction.append('عالیست')
satisfaction.append('راضیه')
satisfaction.append('خوشرنگ')
satisfaction.append('متشکرم')
satisfaction.append('بخریدش')
satisfaction.append('قدرتمند')
satisfaction.append('عالیی')
satisfaction.append('مقاومه')
satisfaction.append('باحاله')
satisfaction.append('متنوع')
satisfaction.append('بینظیره')
satisfaction.append('باکیفیتی')
satisfaction.append('خوشدست')
satisfaction.append('عالی')
satisfaction.append('عالیه')
satisfaction.append('حرف')
satisfaction.append('رضایت')
satisfaction.append('بودممنون')
satisfaction.append('بینظیره')




#not satisfaction
notsatisfaction.append('زمخت')
notsatisfaction.append('بد')
notsatisfaction.append('متاسفانه')
notsatisfaction.append('خراب')
notsatisfaction.append('ضعیف')
notsatisfaction.append('نخرید')
notsatisfaction.append('بدی')
notsatisfaction.append('بدرد')
notsatisfaction.append('سخت')
notsatisfaction.append('سنگین')
notsatisfaction.append('اذیت')
notsatisfaction.append('اشتباه')
notsatisfaction.append('ضعیفه')
notsatisfaction.append('افتضاح')
notsatisfaction.append('حیف')
notsatisfaction.append('ضعف')
notsatisfaction.append('ایراد')
notsatisfaction.append('پشیمونم')
#notsatisfaction.append('خسته')
notsatisfaction.append('گرونه')
notsatisfaction.append('هنگ')
notsatisfaction.append('افتضاحه')
notsatisfaction.append('سنگینه')
notsatisfaction.append('ضعیفی')
notsatisfaction.append('مشکلش')
notsatisfaction.append('الکی')
notsatisfaction.append('گول')
notsatisfaction.append('ضرر')
notsatisfaction.append('ناراضی')
notsatisfaction.append('بدترین')
notsatisfaction.append('ایرادش')
notsatisfaction.append('مشکلات')
notsatisfaction.append('مزیت')
notsatisfaction.append('متاسفم')
notsatisfaction.append('ضعفش')
notsatisfaction.append('پشیمان')
notsatisfaction.append('فاقد')
notsatisfaction.append('ضعفی')
notsatisfaction.append('زشت')
notsatisfaction.append('نخرین')
notsatisfaction.append('نامناسب')
notsatisfaction.append('اشغال')
notsatisfaction.append('پولتونو')
notsatisfaction.append('نمیارزه')
notsatisfaction.append('بدیش')
notsatisfaction.append('اشکال')
notsatisfaction.append('متأسفانه')
notsatisfaction.append('انداختمش')
notsatisfaction.append('زباله')
notsatisfaction.append('آشغال')
notsatisfaction.append('نامرغوب')
notsatisfaction.append('مسخره')
notsatisfaction.append('منفی')
notsatisfaction.append('بی')
notsatisfaction.append('نمیتونه')
notsatisfaction.append('دور')
notsatisfaction.append('گیر')
notsatisfaction.append('داغ')
notsatisfaction.append('زمانبره')



H=B['comment'].agg(lambda x:[word for word in x.split() if word in satisfaction])
comment1=H.agg(lambda x:' '.join(x))
H_count=comment1.str.split(expand=True).stack().value_counts()
s1=set(H_count.index)
m=pd.Series(list(s1))


F=B['comment'].agg(lambda x:[word for word in x.split() if word in notsatisfaction])
comment2=F.agg(lambda x:' '.join(x))
F_count=comment2.str.split(expand=True).stack().value_counts()
s2=set(F_count.index)
n=pd.Series(list(s2))



allfeatures1=np.zeros((B.shape[0],m.shape[0]))

allfeatures2=np.zeros((B.shape[0],n.shape[0]))
            

for i in np.arange(m.shape[0]):
	allfeatures1[B['comment'].agg(lambda x:sum([y==m[i] for y in x.split()])>0),i]=1


for j in np.arange(n.shape[0]):
	allfeatures2[B['comment'].agg(lambda x:sum([y==n[j] for y in x.split()])>0),j]=2
	

all1=pd.DataFrame(allfeatures1)
all2=pd.DataFrame(allfeatures2)
al=pd.concat((all1,all2),1)
complete_data=pd.concat((B.iloc[:,0],al),1)

n_clusters=2
kmeans=KMeans(n_clusters=n_clusters, random_state=0).fit(complete_data)
labels=kmeans.labels_
average_silhouette=silhouette_score(complete_data,labels)
average_silhouette
label=pd.DataFrame(labels)
c=pd.concat((B.iloc[:,0],label),1)
A=pd.concat((c,B.iloc[:,1]),     1)

A.iloc[:,1].loc[A.iloc[:,1]==1]='satisfy'
A.iloc[:,1].loc[A.iloc[:,1]==0]='notsatisfy'








































