# İş Problemi
# Bir e-ticaret şirketi satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.

# Task 1: 6 months of CLTV Prediction (Görev 1: 6 aylık CLTV Prediction)

# Make a 6-month CLTV prediction for 2010-2011 UK customers. (2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.)
# Interpret and evaluate your results. (Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.)
# ATTENTION! (DİKKAT!)
# It is expected that cltv prediction will be made, not the expected number of transaction for 6 months.(6 aylık expected number of transaction değil cltv prediction yapılmasını beklenmektedir.)
# In other words, continue by installing the BGNBD & GAMMA GAMMA models directly and enter 6 in the moon section for cltv prediction. (Yani direkt BGNBD & GAMMA GAMMA modellerini kurarak devam ediniz ve cltv prediction için ay bölümüne 6 giriniz.)

# if necessary
# !pip install lifetimes
# !pip install sqlalchemy

import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("dersler/hafta_3/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Preparation of lifetime data structure (Lifetime veri yapısının hazırlanması)

# recency: Time since last purchase. Weekly. (according to analysis day on cltv_df, user specific here) (Son satın alma üzerinden geçen zaman. Haftalık. (cltv_df'de analiz gününe göre, burada kullanıcı özelinde))
# T: Customer's age. Weekly. (how long before the analysis date the first purchase was made) (Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış))
# frequency: total number of repeat purchases (frequency>1) (tekrar eden toplam satın alma sayısı (frequency>1))
# monetary_value: average earnings per purchase (satın alma başına ortalama kazanç)

cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

# we specify the naming of the variables (değişkenlerin isimlendirini belirliyoruz)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# We calculate the monetary value as the average earnings per purchase (monetary değerinin satın alma başına ortalama kazanç olarak hesaplıyoruz)
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# We choose those with monetary values greater than zero. (monetary sıfırdan büyük olanların seçiyoruz)
cltv_df = cltv_df[cltv_df["monetary"] > 0]

# We convert recency and T to week value for BGNBD (BGNBD için recency ve T'nin hafta değerine çeviriyoruz)
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

# frequency must be greater than 1, we filter it to greater than 1. (frequency'nin 1'den büyük olması gerekmektiği için 1den büyük olarak filtreliyoruz.)
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# We Are Establishing the BG/NBD Model (BG/NBD Modelini Kuruyoruz)

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# We Establish the GAMMA-GAMMA Model (GAMMA-GAMMA Modelini Kuruyoruz)

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.sort_values("expected_average_profit", ascending=False).head(20)

# We calculate CLTV with BG-NBD and GG model (BG-NBD ve GG modeli ile CLTV'yi hesaplıyoruz)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head()
cltv_final.sort_values(by="clv", ascending=False)[10:30]

# Task 2: CLTV analysis of different time periods. (Görev 2: Farklı zaman periyotlarından oluşan CLTV analizi)
# Calculate 1-month and 12-month CLTV for 2010-2011 UK customers. (2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.)
# Analyze the top 10 people at 1 month CLTV and the 10 highest at 12 months. (1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.)
# Is there a difference? If so, why do you think it could be? (Fark var mı? Varsa sizce neden olabilir?)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index()

rfm_cltv1_final = cltv_df.merge(cltv, on="Customer ID", how="left")
rfm_cltv1_final.sort_values(by="clv", ascending=False).head()

########
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv = cltv.reset_index()

rfm_cltv12_final = cltv_df.merge(cltv, on="Customer ID", how="left")
rfm_cltv12_final.sort_values(by="clv", ascending=False).head(10)

# There are obvious differences in the clv values. This is due to the fact that the 1-month clv and 12-month clv values of the customers differ in the longer period.
# (clv değerlerinde bariz farklar görünüyor. bu da müşterilerin 1 aylık clv ile 12 aylık clv değerlerinin daha uzun periyotta farklı olmasından kaynaklanmaktadır.)

# Task 3: For 2010-2011 UK customers, divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the dataset.
# (Görev 3: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.)
# Make short 6-month action suggestions to the management for 2 groups you will choose from among 4 groups. (4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.)

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # months
                                   freq="W",  # T haftalık
                                   discount_rate=0.01)

cltv_final_6 = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final_6["cltv_segment"] = pd.qcut(cltv_final_6["clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.groupby("segment").agg({"count", "mean", "sum"})

# The expected average profit of segment A is highly differentiated compared to other segments. Efforts should be made to make this segment feel its privileges. (A segmentinin beklenen ortalama karı diğer segmentlere göre yüksek farkla ayrışıyor. Bu segmente ayrıcalıklarını hissettirecek çalışmalar yapılmalı.)
# With the campaigns to be made in the B segment, its potential can be increased. (B segmenti de yapılacak kampanyalarla potansiyelinin artırılması sağlanabilir.)
