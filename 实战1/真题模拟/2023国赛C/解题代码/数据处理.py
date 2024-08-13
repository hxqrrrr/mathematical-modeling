import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 获取当前脚本所在目录的绝对路径
current_path = os.path.dirname(__file__)

# 构建附件1.xlsx和附件4.xlsx的相对路径
path_file1 = os.path.join(current_path, '附件1.xlsx')
path_file2 = os.path.join(current_path,'附件2.xlsx')
path_file3 = os.path.join(current_path, '附件3.xlsx')
path_file4 = os.path.join(current_path,'附件4.xlsx')
path_file5 = os.path.join(current_path,'附件5.xlsx')

# 打印路径以进行调试
print(path_file1)
print(path_file2)
print(path_file3)
print(path_file4)
print(path_file5)

# 读取附件
df1 = pd.read_excel(path_file1)
df2 = pd.read_excel(path_file2)
df3 = pd.read_excel(path_file3)
df4 = pd.read_excel(path_file4)
df5 = pd.read_excel(path_file5)


# 打印数据框的前几行，查看数据
print(df1.head())
print(df2.head())
print(df3.head())
print(df4.head())
print(df5.head())

#df_merge是全数据，df_result1是单品数据，df_result2是分类数据
df_merge = pd.merge(df1,df2,on='单品编码',how='inner')
df_merge['总销售额']=df_merge['销量(千克)']*df_merge['销售单价(元/千克)']
df_result1 = df_merge.groupby('单品编码').agg({'销量(千克)':'sum','总销售额':'sum'}).reset_index()
df_result2 = df_merge.groupby('分类编码').agg({'销量(千克)':'sum','总销售额':'sum'}).reset_index()
df_merge = df_merge[df_merge['销量(千克)'] >= 0]

df_result2["单品数量"] = df1.groupby('分类编码').agg({'单品编码':'count'}).reset_index()['单品编码']
df_result2["平均单品销售量"] = df_result2['总销售额'] / df_result2['单品数量']

#时间
df_merge['销售日期'] = pd.to_datetime(df_merge['销售日期'])
# 按天统计每个品类销量
daily_sales_1 = df_merge.groupby(['单品名称', df_merge['销售日期'].dt.date])['销量(千克)'].sum().reset_index()
daily_sales_1.columns = ['单品名称', '销售日期', '总销量(千克)']

daily_sales_2 = df_merge.groupby(['分类名称', df_merge['销售日期'].dt.date])['销量(千克)'].sum().reset_index()
daily_sales_2.columns = ['分类名称', '销售日期', '总销量(千克)']
daily_sales_2.to_excel("分类销量按日.xlsx", index=False)

daily_sales_2_1 = daily_sales_2[daily_sales_2["分类名称"] == "水生根茎类"]
daily_sales_2_2 = daily_sales_2[daily_sales_2["分类名称"] == "花叶类"]
daily_sales_2_3 = daily_sales_2[daily_sales_2["分类名称"] == "食用菌"]
daily_sales_2_4 = daily_sales_2[daily_sales_2["分类名称"] == "辣椒类"]
daily_sales_2_5 = daily_sales_2[daily_sales_2["分类名称"] == "花菜类"]
daily_sales_2_6 = daily_sales_2[daily_sales_2["分类名称"] == "茄类"]

daily_sales_2_1.to_excel("分类销量按日_水生根茎类.xlsx", index=False)
daily_sales_2_2.to_excel("分类销量按日_花叶类.xlsx", index=False)
daily_sales_2_3.to_excel("分类销量按日_食用菌.xlsx", index=False)
daily_sales_2_4.to_excel("分类销量按日_辣椒类.xlsx", index=False)
daily_sales_2_5.to_excel("分类销量按日_花菜类.xlsx", index=False)
daily_sales_2_6.to_excel("分类销量按日_茄类.xlsx", index=False)

daily_sales_all = daily_sales_2_1.drop(columns = ["分类名称"])
daily_sales_all = daily_sales_2_1.drop(columns = ["总销量(千克)"])

daily_sales_all["水生根茎类"] = daily_sales_2_2["总销量(千克)"].reset_index(drop=True)
daily_sales_all["花叶类"] = daily_sales_2_2["总销量(千克)"].reset_index(drop=True)
daily_sales_all["食用菌"] = daily_sales_2_3["总销量(千克)"].reset_index(drop=True)
daily_sales_all["辣椒类"] = daily_sales_2_4["总销量(千克)"].reset_index(drop=True)
daily_sales_all["花菜类"] = daily_sales_2_5["总销量(千克)"].reset_index(drop=True)
daily_sales_all["茄类"] = daily_sales_2_6["总销量(千克)"].reset_index(drop=True)
daily_sales_all.to_excel("分类全销量按日.xlsx", index=False)

#单品总销量，日销量，最大值统计
daily_sales_item = daily_sales_1.groupby(['单品名称'])['总销量(千克)'].sum().reset_index()
daily_sales_item = pd.merge(daily_sales_item, daily_sales_1.groupby(['单品名称'])['总销量(千克)'].mean().reset_index(), on='单品名称', how='inner')
daily_sales_item = pd.merge(daily_sales_item, daily_sales_1.groupby(['单品名称'])['总销量(千克)'].max().reset_index(), on='单品名称', how='inner')
daily_sales_item.columns = ["单品名称","单品总销量","单品日均销量","单品日最大销量"]
daily_sales_item['单品总销量'] = daily_sales_item['单品总销量'].round(1)
daily_sales_item['单品日均销量'] = daily_sales_item['单品日均销量'].round(1)
daily_sales_item['单品日最大销量'] = daily_sales_item['单品日最大销量'].round(1)

daily_sales_item.to_excel("单品销量统计.xlsx", index=False)

# k-means聚类分析

features = ['单品总销量', '单品日均销量', '单品日最大销量']
X = daily_sales_item[features]

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置K值
K = 3

# 创建KMeans模型
kmeans = KMeans(n_clusters=K, random_state=42)

# 拟合数据并预测簇标签
daily_sales_item['Cluster'] = kmeans.fit_predict(X_scaled)

# 获取簇中心
centers = kmeans.cluster_centers_

# 将簇中心反标准化
centers = scaler.inverse_transform(centers)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x='单品总销量', y='单品日均销量', hue='Cluster', data=daily_sales_item, palette='viridis', s=100)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title('K-means 聚类算法结果')
plt.xlabel('单品总销量')
plt.ylabel('单品日均销量')
plt.legend(title='Cluster')
plt.show()


daily_sales_item.to_excel("单品销量统计.xlsx", index=False)
daily_sales_item_30 = daily_sales_item.drop(daily_sales_item[(daily_sales_item["Cluster"] == 0) & (daily_sales_item["单品日均销量"] < 8.8)].index)
daily_sales_item_30 = daily_sales_item_30.drop(daily_sales_item[(daily_sales_item["Cluster"] == 1) & (daily_sales_item["单品日均销量"] < 17.3)].index)
daily_sales_item_30 = daily_sales_item_30.reset_index(drop=True)
daily_sales_item_30 = daily_sales_item_30.sort_values(['Cluster', '单品日均销量'],ascending=True)
daily_sales_item_30.to_excel("单品销量统计_30.xlsx", index=False,)
df3.columns =["销售日期","单品编码","批发价格(元/千克)"]
daily_sales_item_1 = pd.merge(df3,df_merge,on=['单品编码','销售日期'],how='inner')

daily_sales_item_1.to_excel("单品日销量_批发价.xlsx", index=False)

