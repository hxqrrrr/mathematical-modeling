import os
import pandas as pd

# 获取当前脚本所在目录的绝对路径
current_path = os.path.dirname(__file__)

# 构建附件1.xlsx和附件4.xlsx的相对路径
path_file1 = os.path.join(current_path, '附件1.xlsx')
path_file2 = os.path.join(current_path,'附件2.xlsx')

# 打印路径以进行调试
print(path_file1)
print(path_file2)
# 读取附件1.xlsx
df1 = pd.read_excel(path_file1)
# 读取附件2.xlsx
df2 = pd.read_excel(path_file2)

# 打印数据框的前几行，查看数据
print(df1.head())
print(df2.head())

df_merge = pd.merge(df1,df2,on='单品编码',how='inner')
df_merge['总销售额']=df_merge['销量(千克)']*df_merge['销售单价(元/千克)']

df_result1 = df_merge.groupby('单品编码').agg({'销量(千克)':'sum','总销售额':'sum'}).reset_index()
df_result2 = df_merge.groupby('分类编码').agg({'销量(千克)':'sum','总销售额':'sum'}).reset_index()

print(df_result1.to_string())
print(df_result2.to_string())
