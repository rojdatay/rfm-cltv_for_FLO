###############################################################
############# Customer Segmentation with RFM #################
##### CLTV Prediction with BG-NBD & Gamma-Gamma Models ########
###############################################################

###############################################################
# Business Problem
###############################################################

# 1. FLO wants to segment its customers and determine marketing strategies based on these segments.
# 2. For this purpose, customer behaviors will be defined, and groups will be formed based on these behavior patterns.
# 3. FLO wants to establish a roadmap for its sales and marketing activities.
# 4. In order for the company to plan for the medium and long term, it is essential to predict the potential value
# that existing customers will bring to the company in the future.

###############################################################
# Data Set Story
###############################################################

#The dataset consists of information derived from the past shopping behaviors of customers who made their last purchases in the years 2020-2021
#through the OmniChannel (both online and offline shopping).

#master_id: Unique customer number
#order_channel: The channel used for shopping (Android, iOS, Desktop, Mobile, Offline) related to the shopping platform.
#last_order_channel: The channel used for the last purchase.
#first_order_date: The date of the customer's first purchase.
#last_order_date: The date of the customer's last purchase.
#last_order_date_online: The date of the customer's last online purchase.
#last_order_date_offline: The date of the customer's last offline purchase.
#order_num_total_ever_online: The total number of purchases made by the customer online.
#order_num_total_ever_offline: The total number of purchases made by the customer offline.
#customer_value_total_ever_offline: The total amount spent by the customer on offline purchases.
#customer_value_total_ever_online: The total amount spent by the customer on online purchases.
#interested_in_categories_12: The list of categories in which the customer made purchases in the last 12 months.

###############################################################
########### Customer Segmentation with RFM ###################
###############################################################

# İmport Library
#############################

import datetime as dt
import pandas as pd
#!pip install lifetimes
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None

# import data
#############################

df_ = pd.read_csv("contents/CRM/Flo/datasets/flo_data_20k.csv")

df= df_.copy()

# Data Understanding
#############################

def describe_data(df):
    print("###################### First 5 Lines ###################")
    print(df.head())
    print("###################### Last 5 Lines ###################")
    print(df.tail())
    print("###################### Types ###################")
    print(df.dtypes)
    print("######################## Shape #########################")
    print(df.shape)
    print("######################### Info #########################")
    print(df.info())
    print("######################### N/A ##########################")
    print(df.isnull().sum())
    print("######################### Quantiles  ######################")
    print(df.describe().T)

describe_data(df)
"""
###################### First 5 Lines ###################
                              master_id order_channel last_order_channel first_order_date last_order_date last_order_date_online last_order_date_offline  order_num_total_ever_online  order_num_total_ever_offline  customer_value_total_ever_offline  customer_value_total_ever_online       interested_in_categories_12
0  cc294636-19f0-11eb-8d74-000d3a38a36f   Android App            Offline       2020-10-30      2021-02-26             2021-02-21              2021-02-26                         4.00                          1.00                             139.99                            799.38                           [KADIN]
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f   Android App             Mobile       2017-02-08      2021-02-16             2021-02-16              2020-01-10                        19.00                          2.00                             159.97                           1853.58  [ERKEK, COCUK, KADIN, AKTIFSPOR]
2  69b69676-1a40-11ea-941b-000d3a38a36f   Android App        Android App       2019-11-27      2020-11-27             2020-11-27              2019-12-01                         3.00                          2.00                             189.97                            395.35                    [ERKEK, KADIN]
3  1854e56c-491f-11eb-806e-000d3a38a36f   Android App        Android App       2021-01-06      2021-01-17             2021-01-17              2021-01-06                         1.00                          1.00                              39.99                             81.98               [AKTIFCOCUK, COCUK]
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       Desktop            Desktop       2019-08-03      2021-03-07             2021-03-07              2019-08-03                         1.00                          1.00                              49.99                            159.99                       [AKTIFSPOR]
###################### Last 5 Lines ###################
                                  master_id order_channel last_order_channel first_order_date last_order_date last_order_date_online last_order_date_offline  order_num_total_ever_online  order_num_total_ever_offline  customer_value_total_ever_offline  customer_value_total_ever_online interested_in_categories_12
19940  727e2b6e-ddd4-11e9-a848-000d3a38a36f   Android App            Offline       2019-09-21      2020-07-05             2020-06-05              2020-07-05                         1.00                          2.00                             289.98                            111.98          [ERKEK, AKTIFSPOR]
19941  25cd53d4-61bf-11ea-8dd8-000d3a38a36f       Desktop            Desktop       2020-03-01      2020-12-22             2020-12-22              2020-03-01                         1.00                          1.00                             150.48                            239.99                 [AKTIFSPOR]
19942  8aea4c2a-d6fc-11e9-93bc-000d3a38a36f       Ios App            Ios App       2019-09-11      2021-05-24             2021-05-24              2019-09-11                         2.00                          1.00                             139.98                            492.96                 [AKTIFSPOR]
19943  e50bb46c-ff30-11e9-a5e8-000d3a38a36f   Android App        Android App       2019-03-27      2021-02-13             2021-02-13              2021-01-08                         1.00                          5.00                             711.79                            297.98          [ERKEK, AKTIFSPOR]
19944  740998d2-b1f7-11e9-89fa-000d3a38a36f   Android App        Android App       2019-09-03      2020-06-06             2020-06-06              2019-09-03                         1.00                          1.00                              39.99                            221.98          [KADIN, AKTIFSPOR]
###################### Types ###################
master_id                             object
order_channel                         object
last_order_channel                    object
first_order_date                      object
last_order_date                       object
last_order_date_online                object
last_order_date_offline               object
order_num_total_ever_online          float64
order_num_total_ever_offline         float64
customer_value_total_ever_offline    float64
customer_value_total_ever_online     float64
interested_in_categories_12           object
dtype: object
######################## Shape #########################
(19945, 12)
######################### Info #########################
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19945 entries, 0 to 19944
Data columns (total 12 columns):
 #   Column                             Non-Null Count  Dtype  
---  ------                             --------------  -----  
 0   master_id                          19945 non-null  object 
 1   order_channel                      19945 non-null  object 
 2   last_order_channel                 19945 non-null  object 
 3   first_order_date                   19945 non-null  object 
 4   last_order_date                    19945 non-null  object 
 5   last_order_date_online             19945 non-null  object 
 6   last_order_date_offline            19945 non-null  object 
 7   order_num_total_ever_online        19945 non-null  float64
 8   order_num_total_ever_offline       19945 non-null  float64
 9   customer_value_total_ever_offline  19945 non-null  float64
 10  customer_value_total_ever_online   19945 non-null  float64
 11  interested_in_categories_12        19945 non-null  object 
dtypes: float64(4), object(8)
memory usage: 1.8+ MB
None
######################### N/A ##########################
master_id                            0
order_channel                        0
last_order_channel                   0
first_order_date                     0
last_order_date                      0
last_order_date_online               0
last_order_date_offline              0
order_num_total_ever_online          0
order_num_total_ever_offline         0
customer_value_total_ever_offline    0
customer_value_total_ever_online     0
interested_in_categories_12          0
dtype: int64
######################### Quantiles  ######################
                                     count   mean    std   min    25%    50%    75%      max
order_num_total_ever_online       19945.00   3.11   4.23  1.00   1.00   2.00   4.00   200.00
order_num_total_ever_offline      19945.00   1.91   2.06  1.00   1.00   1.00   2.00   109.00
customer_value_total_ever_offline 19945.00 253.92 301.53 10.00  99.99 179.98 319.97 18119.14
customer_value_total_ever_online  19945.00 497.32 832.60 12.99 149.98 286.46 578.44 45220.13
"""

# In this project, we will focus on the concept of OmniChannel, which refers to customers engaging in both online and offline shopping.
# To effectively analyze this behavior, we will introduce new variables to monitor the total number of purchases and total expenditure for each customer.
# These variables will offer valuable insights into the overall shopping activity and spending habits of customers across various channels.
# By taking into account both online and offline transactions, we aim to gain a comprehensive understanding of customer behavior and their significance to the business

df["total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# When we examine the variable types describe_data function, we can see that the variables representing dates are split into a numerical format.
# Let's convert them back to dates.

date_columns = [col for col in df.columns if "date" in col]
df[date_columns]= df[date_columns].apply(pd.to_datetime)
df.info()
"""
 #   Column                             Non-Null Count  Dtype         
---  ------                             --------------  -----         
 0   master_id                          19945 non-null  object        
 1   order_channel                      19945 non-null  object        
 2   last_order_channel                 19945 non-null  object        
 3   first_order_date                   19945 non-null  datetime64[ns]
 4   last_order_date                    19945 non-null  datetime64[ns]
 5   last_order_date_online             19945 non-null  datetime64[ns]
 6   last_order_date_offline            19945 non-null  datetime64[ns]
 7   order_num_total_ever_online        19945 non-null  float64       
 8   order_num_total_ever_offline       19945 non-null  float64       
 9   customer_value_total_ever_offline  19945 non-null  float64       
 10  customer_value_total_ever_online   19945 non-null  float64       
 11  interested_in_categories_12        19945 non-null  object        
 12  total_purchases                    19945 non-null  float64       
 13  customer_value                     19945 non-null  float64       
dtypes: datetime64[ns](4), float64(6), object(4)
"""

# Let's examine the distribution of the number of customers in shopping channels,
# the average number of products purchased, and the average expenditures.

df.groupby("order_channel").agg({"total_purchases": ["count", "sum"],
                                 "customer_value": ["count", "sum"]})
"""
              total_purchases          customer_value           
                        count      sum          count        sum
order_channel                                                   
Android App              9495 52269.00           9495 7819062.76
Desktop                  2735 10920.00           2735 1610321.46
Ios App                  2833 15351.00           2833 2525999.93
Mobile                   4882 21679.00           4882 3028183.16

"""

# Let's rank the top 10 customers who bring the highest revenue
df.groupby("master_id").agg({"customer_value": "sum"}).sort_values("customer_value", ascending=False).head(10)
"""
                                      customer_value
master_id                                           
5d1c466a-9cfd-11e9-9897-000d3a38a36f        45905.10
d5ef8058-a5c6-11e9-a2fc-000d3a38a36f        36818.29
73fd19aa-9e37-11e9-9897-000d3a38a36f        33918.10
7137a5c0-7aad-11ea-8f20-000d3a38a36f        31227.41
47a642fe-975b-11eb-8c2a-000d3a38a36f        20706.34
a4d534a2-5b1b-11eb-8dbd-000d3a38a36f        18443.57
d696c654-2633-11ea-8e1c-000d3a38a36f        16918.57
fef57ffa-aae6-11e9-a2fc-000d3a38a36f        12726.10
cba59206-9dd1-11e9-9897-000d3a38a36f        12282.24
fc0ce7a4-9d87-11e9-9897-000d3a38a36f        12103.15
"""

# Let's rank the top 10 customers with the highest number of orders.
df.groupby("master_id").agg({"total_purchases": "sum"}).sort_values(by="total_purchases", ascending=False).head(10)
"""
                                      total_purchases
master_id                                            
5d1c466a-9cfd-11e9-9897-000d3a38a36f           202.00
cba59206-9dd1-11e9-9897-000d3a38a36f           131.00
a57f4302-b1a8-11e9-89fa-000d3a38a36f           111.00
fdbe8304-a7ab-11e9-a2fc-000d3a38a36f            88.00
329968c6-a0e2-11e9-a2fc-000d3a38a36f            83.00
73fd19aa-9e37-11e9-9897-000d3a38a36f            82.00
44d032ee-a0d4-11e9-a2fc-000d3a38a36f            77.00
b27e241a-a901-11e9-a2fc-000d3a38a36f            75.00
d696c654-2633-11ea-8e1c-000d3a38a36f            70.00
a4d534a2-5b1b-11eb-8dbd-000d3a38a36f            70.00
"""



#  Calculation of RFM Metrics
####################################

# Recency : When was the customer's most recent purchase? # Today's Date - Last purchase date
# Frequency : How frequently has the customer been shopping?
# Monetary / (Monetary Value): How much money has the customer spent?

# Note: To calculate RFM Metrics we choose an analysis date two days after the latest last order date in the dataframe!

df["last_order_date"].max()
# Timestamp('2021-05-30 00:00:00')


today_date= df["last_order_date"].max() + dt.timedelta(days=2)
today_date  # --> Timestamp('2021-06-01 00:00:00')


rfm = pd.DataFrame()
rfm["master_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["total_purchases"]
rfm["monetary"] = df["customer_value"]

rfm.head()
"""
                              master_id  frequency  recency  monetary
0  cc294636-19f0-11eb-8d74-000d3a38a36f       5.00    95.00    939.37
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f      21.00   105.00   2013.55
2  69b69676-1a40-11ea-941b-000d3a38a36f       5.00   186.00    585.32
3  1854e56c-491f-11eb-806e-000d3a38a36f       2.00   135.00    121.97
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       2.00    86.00    209.98

"""

rfm= rfm[rfm["monetary"]> 0]  # Let's remove the ones with zero monetary value since money cannot be zero.

rfm.describe().T
"""
             count   mean    std   min    25%    50%    75%      max
frequency 19945.00   5.02   4.74  2.00   3.00   4.00   6.00   202.00
recency   19945.00 134.46 103.28  2.00  43.00 111.00 202.00   367.00
monetary  19945.00 751.24 895.40 44.98 339.98 545.27 897.78 45905.10
"""


# Calculation of RF Score
#################################

# Let's rank the scores, 5 being the highest, 1 being the lowest score.

rfm["recency_score"]= pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
rfm["monetary_score"]= pd.qcut(rfm["monetary"], 5, labels=[1,2,3,4,5])
rfm["frequency_score"]= pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]) ## method="first" --> order-based ranking


# Let's assign "recency_score" and "frequency_score" as a single variable named "RF_SCORE"

rfm["RF_SCORE"]= (rfm["recency_score"].astype(str)+ rfm["frequency_score"].astype(str))

rfm.head()
"""
                              master_id  frequency  recency  monetary recency_score monetary_score frequency_score RF_SCORE
0  cc294636-19f0-11eb-8d74-000d3a38a36f       5.00    95.00    939.37             3              4               4       34
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f      21.00   105.00   2013.55             3              5               5       35
2  69b69676-1a40-11ea-941b-000d3a38a36f       5.00   186.00    585.32             2              3               4       24
3  1854e56c-491f-11eb-806e-000d3a38a36f       2.00   135.00    121.97             3              1               1       31
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       2.00    86.00    209.98             3              1               1       31

"""



# Customer Segmentation
############################

seg_map = {
    r"[1-2][1-2]": "hibernating",
    r"[1-2][3-4]": "at_Risk",
    r"[1-2]5": "cant_loose",
    r"3[1-2]": "about_to_sleep",
    r"33": "need_attention",
    r"[3-4][4-5]": "loyal_customers",
    r"41": "promising",
    r"51": "new_customers",
    r"[4-5][2-3]": "potential_loyalists",
    r"5[4-5]": "champions"
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.head()
"""
                              master_id  frequency  recency  monetary recency_score monetary_score frequency_score RF_SCORE          segment
0  cc294636-19f0-11eb-8d74-000d3a38a36f       5.00    95.00    939.37             3              4               4       34  loyal_customers
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f      21.00   105.00   2013.55             3              5               5       35  loyal_customers
2  69b69676-1a40-11ea-941b-000d3a38a36f       5.00   186.00    585.32             2              3               4       24          at_Risk
3  1854e56c-491f-11eb-806e-000d3a38a36f       2.00   135.00    121.97             3              1               1       31   about_to_sleep
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       2.00    86.00    209.98             3              1               1       31   about_to_sleep

"""

# let's check to segment groups.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])
"""
                       mean count      mean count     mean count
segment                                                         
about_to_sleep       113.79  1629      2.40  1629   359.01  1629
at_Risk              241.61  3131      4.47  3131   646.61  3131
cant_loose           235.44  1200     10.70  1200  1474.47  1200
champions             17.11  1932      8.93  1932  1406.63  1932
hibernating          247.95  3604      2.39  3604   366.27  3604
loyal_customers       82.59  3361      8.37  3361  1216.82  3361
need_attention       113.83   823      3.73   823   562.14   823
new_customers         17.92   680      2.00   680   339.96   680
potential_loyalists   37.16  2938      3.30  2938   533.18  2938
promising             58.92   647      2.00   647   335.67   647
"""

# The target customer segments for the new women's shoe brand at FLO are planned to be loyal customers
# who primarily purchase from the women's category and are willing to pay higher prices.
# To identify these segments, RFM analysis and other customer behavior attributes can be utilized.
# By leveraging RFM analysis and considering customer interests, the brand can effectively communicate
# and market its products to these target customer segments.

rfm_final= rfm.merge(df, on="master_id", how="left")
rfm_final.head()
"""
Out[272]: 
                              master_id  frequency  recency  monetary recency_score monetary_score frequency_score RF_SCORE          segment order_channel last_order_channel first_order_date last_order_date last_order_date_online last_order_date_offline  order_num_total_ever_online  order_num_total_ever_offline  customer_value_total_ever_offline  customer_value_total_ever_online       interested_in_categories_12  total_purchases  customer_value
0  cc294636-19f0-11eb-8d74-000d3a38a36f       5.00    95.00    939.37             3              4               4       34  loyal_customers   Android App            Offline       2020-10-30      2021-02-26             2021-02-21              2021-02-26                         4.00                          1.00                             139.99                            799.38                           [KADIN]             5.00          939.37
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f      21.00   105.00   2013.55             3              5               5       35  loyal_customers   Android App             Mobile       2017-02-08      2021-02-16             2021-02-16              2020-01-10                        19.00                          2.00                             159.97                           1853.58  [ERKEK, COCUK, KADIN, AKTIFSPOR]            21.00         2013.55
2  69b69676-1a40-11ea-941b-000d3a38a36f       5.00   186.00    585.32             2              3               4       24          at_Risk   Android App        Android App       2019-11-27      2020-11-27             2020-11-27              2019-12-01                         3.00                          2.00                             189.97                            395.35                    [ERKEK, KADIN]             5.00          585.32
3  1854e56c-491f-11eb-806e-000d3a38a36f       2.00   135.00    121.97             3              1               1       31   about_to_sleep   Android App        Android App       2021-01-06      2021-01-17             2021-01-17              2021-01-06                         1.00                          1.00                              39.99                             81.98               [AKTIFCOCUK, COCUK]             2.00          121.97
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f       2.00    86.00    209.98             3              1               1       31   about_to_sleep       Desktop            Desktop       2019-08-03      2021-03-07             2021-03-07              2019-08-03                         1.00                          1.00                              49.99                            159.99                       [AKTIFSPOR]             2.00          209.98

"""



flo_kadin = rfm_final.loc[(rfm_final["segment"].isin(["champions", 'loyal_customers'])) &
                       (rfm_final["interested_in_categories_12"].str.contains("KADIN"))]

flo_kadin.shape # (2497, 22) we suggested 2497 customers
flo_kadin.to_csv("flo_kadin.csv")

# A discount of nearly 40% is planned for men's and children's products.
# Customers who are interested in these discounted categories and have been good customers
# in the past but haven't made purchases for a long time, newly acquired customers,
# and those who are considered "at risk" of being lost should be specifically targeted.
# The goal is to save the IDs of the eligible customers.


flo_40_discount = rfm_final.loc[(rfm_final["segment"].isin(["cant_loose", "about_to_sleep", 'new_customers'])) &
                    (rfm_final["interested_in_categories_12"].str.contains('COCUK', 'ERKEK'))]

flo_40_discount.shape  # (980, 22) we suggested 980 customers
flo_40_discount.to_csv("flo_40_discount")



###############################################################
##### CLTV Prediction with BG-NBD & Gamma-Gamma Models ########
###############################################################

# This part focuses on Customer Lifetime Value (CLTV) prediction using advanced models,
# specifically the BG-NBD and Gamma-Gamma models.
# Building upon the earlier RFM analysis, which segmented customers based on Recency, Frequency,
# and Monetary Value, this project aims to delve deeper into predicting future customer value.
# The BG-NBD model allows for the estimation of future purchasing probabilities and customer churn rates,
# while the Gamma-Gamma model enables analysis of customer transaction values.

# Data Prep
######################

df = df_.copy()
df.describe().T  # for check to outliers
"""
                                     count   mean    std   min    25%    50%    75%      max
order_num_total_ever_online       19945.00   3.11   4.23  1.00   1.00   2.00   4.00   200.00
order_num_total_ever_offline      19945.00   1.91   2.06  1.00   1.00   1.00   2.00   109.00
customer_value_total_ever_offline 19945.00 253.92 301.53 10.00  99.99 179.98 319.97 18119.14
customer_value_total_ever_online  19945.00 497.32 832.60 12.99 149.98 286.46 578.44 45220.13
"""

def outlier_thresholds(dataframe, variable):
    quartile1=dataframe[variable].quantile(0.25)
    quartile3=dataframe[variable].quantile(0.75)
    interquantile_range = quartile3-quartile1
    up_limit= quartile3+ 1.5*interquantile_range
    low_limit= quartile1 - 1.5*interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit= outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable]< low_limit), variable]= low_limit
    dataframe.loc[(dataframe[variable]> up_limit), variable]= up_limit

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

df.describe().T
"""
                                     count   mean    std   min    25%    50%    75%     max
order_num_total_ever_online       19945.00   2.72   2.25  1.00   1.00   2.00   4.00    8.50
order_num_total_ever_offline      19945.00   1.72   0.89  1.00   1.00   1.00   2.00    3.50
customer_value_total_ever_offline 19945.00 232.45 173.96 10.00  99.99 179.98 319.97  649.94
customer_value_total_ever_online  19945.00 416.86 352.91 12.99 149.98 286.46 578.44 1221.13
"""

df["total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

date_columns = [col for col in df.columns if "date" in col]
df[date_columns]= df[date_columns].apply(pd.to_datetime)

today_date= df["last_order_date"].max() + dt.timedelta(days=2)
today_date  # --> Timestamp('2021-06-01 00:00:00')

# Create a new cltv dataframe
#################################

# recency= time since last purchase, weekly
# T = custoer tenure, weekly
# Frequency= Total repeat purchases (frequency>1)
# monatary_value: total earnings per purchase

cltv_df= pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"]= ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["T_weekly"]= ((today_date- df["first_order_date"]).astype('timedelta64[D]'))/7
cltv_df["frequency"] = df["total_purchases"]
cltv_df["monetary_cltv_avg"] = df["customer_value"] / df["total_purchases"]

cltv_df.head()
"""
                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg
0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57       5.00             187.87
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86      10.50             131.53
2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86       5.00             117.06
3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86       2.00              60.98
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43       2.00             104.99
"""
cltv_df.describe().T
"""
                       count   mean   std   min    25%    50%    75%    max
recency_cltv_weekly 19945.00  95.26 74.59  0.00  50.43  76.57 109.43 433.43
T_weekly            19945.00 114.47 74.77  0.71  73.86  93.00 119.43 437.14
frequency           19945.00   4.44  2.45  2.00   3.00   4.00   5.50  12.00
monetary_cltv_avg   19945.00 149.91 64.08 22.49 106.66 139.12 178.86 765.55
"""

cltv_df= cltv_df[cltv_df["recency_cltv_weekly"]>1]
cltv_df.describe().T
"""
                       count   mean   std   min    25%    50%    75%    max
recency_cltv_weekly 19790.00  96.01 74.41  1.14  51.29  76.86 109.71 433.43
T_weekly            19790.00 115.20 74.59  1.86  74.43  93.43 119.57 437.14
frequency           19790.00   4.45  2.45  2.00   3.00   4.00   5.50  12.00
monetary_cltv_avg   19790.00 149.69 63.80 22.49 106.66 139.07 178.45 765.55
"""

# Predicting purchases using the BG/NBD model
################################################


bgf= BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# Let's predict expected purchases from customers in 3 months
cltv_df["exp_sales_3_month"]= bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                        cltv_df["frequency"],
                                                                                        cltv_df["recency_cltv_weekly"],
                                                                                        cltv_df["T_weekly"])

# Let's predict expected purchases from customers in 6 months
cltv_df["exp_sales_6_month"]= bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                                                        cltv_df["frequency"],
                                                                                        cltv_df["recency_cltv_weekly"],
                                                                                        cltv_df["T_weekly"])


# Predicting average value per transaction using the Gamma-Gamma model
######################################################################

cltv_df["frequency"] = cltv_df["frequency"].astype(int)  # the GammaGammaFitter run with int type

ggf= GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

cltv_df["expected_average_profit"]= ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary_cltv_avg"])
# let's calculate 6-month CLTV
cltv_6_month = ggf.customer_lifetime_value(bgf,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'],
                                               cltv_df['monetary_cltv_avg'],
                                               time=6,  # aylık değer için
                                               freq="W",  # frekans bigisi bu çalışmada hafta
                                               discount_rate=0.01)

cltv_6_month.reset_index()

cltv_df["cltv_6_month"] = cltv_6_month

cltv_df.head()
"""
                            customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  cltv_6_month
0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     30.57          5             187.87               0.82               1.64        332.41
1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    224.86         10             131.53               0.54               1.09        147.29
2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     78.86          5             117.06               0.61               1.22        155.38
3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     20.86          2              60.98               0.61               1.22         85.94
4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     95.43          2             104.99               0.39               0.78         93.70
"""

# Let's look at the 20 people with the highest cltv_6_month value

cltv_df.sort_values(by="cltv_6_month", ascending=False).head(20)
# Identify the top 20 customers who are prediction to generate the highest revenue within 6 months
"""
                                customer_id  recency_cltv_weekly  T_weekly  frequency  monetary_cltv_avg  exp_sales_3_month  expected_average_profit  exp_sales_6_month  cltv_6_month
6402   851de3b4-8f0c-11eb-8cb8-000d3a38a36f                 8.29      9.43          2             650.22               0.67                   696.51               1.33         972.42
9055   47a642fe-975b-11eb-8c2a-000d3a38a36f                 2.86      7.86          4             467.77               0.87                   484.17               1.75         886.66
15123  635b5e0a-a686-11eb-a6d3-000d3a299ebf                 2.00      5.14          3             447.11               0.79                   468.31               1.58         777.44
7173   8897f4a8-c793-11ea-b753-000d3a38a36f                40.14     45.71          3             597.14               0.59                   624.98               1.18         772.17
10792  f1f89712-84e5-11eb-8a3c-000d3a38a36f                 7.29     11.29          2             522.45               0.66                   560.08               1.31         770.25
14858  031b2954-6d28-11eb-99c4-000d3a38a36f                14.86     15.57          3             450.70               0.73                   472.05               1.45         720.05
14833  b09765ae-29a1-11eb-b280-000d3a38a36f                 1.57     27.86          2             552.81               0.58                   592.49               1.16         718.92
12828  0c24fc44-2ac8-11ea-9d27-000d3a38a36f                68.00     84.29          2             765.55               0.41                   819.67               0.83         709.98
4087   5ee23224-ad83-11ea-b736-000d3a38a36f                49.57     50.43          2             631.11               0.50                   676.10               1.00         707.01
18979  251ced36-a2fb-11eb-a692-000d3a38a36f                 2.00      5.86          2             449.96               0.69                   482.67               1.37         694.13
14209  413c8242-1386-11eb-8ffc-000d3a38a36f                27.71     32.14          2             544.44               0.56                   583.56               1.12         687.16
7171   77e66e92-31fa-11eb-860c-000d3a38a36f                16.86     26.29          5             374.21               0.84                   384.81               1.69         681.02
2090   529fc4f6-7adb-11eb-8460-000d3a38a36f                 2.71     13.14          3             416.59               0.74                   436.44               1.48         678.55
3418   64b689d8-2075-11eb-b361-000d3a38a36f                 2.86     29.14          3             468.54               0.66                   490.68               1.31         676.97
9738   3a27b334-dff4-11ea-acaa-000d3a38a36f                40.00     41.14          3             507.49               0.61                   531.36               1.21         675.99
16770  b46f33b8-a09a-11ea-9d0a-000d3a38a36f                14.71     52.71          2             600.51               0.49                   643.43               0.98         663.56
18997  41231c72-566a-11eb-9e65-000d3a38a36f                 2.57      4.57          7             257.30               1.20                   262.65               2.41         663.33
50     c109302c-f72c-11ea-a533-000d3a38a36f                13.00     37.00          3             482.65               0.62                   505.42               1.25         660.75
18977  380cabbe-da8a-11ea-a65b-000d3a38a36f                29.14     42.00          3             496.56               0.60                   519.94               1.21         657.81
7231   36863d7a-2646-11eb-8a9b-000d3a38a36f                20.43     28.57          3             453.32               0.66                   474.79               1.32         657.69

"""

# Let's segment customers based on 6-month revenue forecast

cltv_df["cltv_segment_6_month"]= pd.qcut(cltv_df["cltv_6_month"], 4, labels=["D", "C", "B", "A"])
cltv_df.groupby("cltv_segment_6_month").agg({"count", "mean", "sum"})

# We can take it with various actions specific to each segment.
"""
                     recency_cltv_weekly               T_weekly              frequency            monetary_cltv_avg              exp_sales_3_month            exp_sales_6_month            cltv_6_month             
                                     sum   mean count       sum   mean count       sum mean count               sum   mean count               sum mean count               sum mean count          sum   mean count
cltv_segment_6_month                                                                                                                                                                                                
D                              692790.14 140.01  4948 806899.57 163.08  4948     18392 3.72  4948         458558.53  92.68  4948           1975.63 0.40  4948           3951.26 0.80  4948    388269.05  78.47  4948
C                              502584.14 101.59  4947 600612.14 121.41  4947     21346 4.31  4947         628560.16 127.06  4947           2390.67 0.48  4947           4781.34 0.97  4947    640189.37 129.41  4947
B                              405114.43  81.89  4947 496452.43 100.35  4947     22804 4.61  4947         787922.87 159.27  4947           2671.99 0.54  4947           5343.98 1.08  4947    891104.74 180.13  4947
A                              299469.86  60.52  4948 375865.43  75.96  4948     24105 4.87  4948        1087297.33 219.74  4948           3093.01 0.63  4948           6186.02 1.25  4948   1407450.59 284.45  4948

"""

