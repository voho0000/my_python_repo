# Dataframe SOP

[TOC]

[有目錄版HackMD筆記](https://hackmd.io/lprtq0KiRj2tRnIQoLLyXw?both)

## 步驟
1. 資料讀取+路徑處理
2. 檢視資料
3. column/row處理
     1. 檢視columns name    
     2. 改動類型或名稱
     3. 刪除不用columns
     4. 改變內容

## 資料讀取

```python
import glob
import os
import numpy as np
import pandas as pd
```

```python
#抓特定資料夾裡所有的檔案路徑
glob.glob('./summary/4/*')

#指讀取3-6資料，且最後要長特定模樣
for category in [4,5,6,7]:
path=glob.glob(('./summary/'+str(category)+'/*_clear_fi.xlsx'))
```

## 資料檢視
```python
type(data)
data.head()
data.tail()
data.shape
data.ndim
data.dtypes
data.describe()
data['column'].describe()
```
## set index
根據哪個column做index
```python
df2.set_index('NAME', inplace=True)
```

## 合併
- join:outer如果遇到空白資料填NA，inter則是刪除空白資料
- axis=0 直向合併，axix=1 橫向合併
`pd.concat(df_list,axis=1,join='outer')`

## column處理
### 改動
```python
#檢視現有columns
data.columns

# drop columns
## 單個
data=data.drop(columns='Unnamed: 0')
## 多個
data=data.drop(columns=['stable_frequent','_stable_frequent','標準','Pr > |t|','sample_n'])
# rename columns
data.rename(columns={'估計值':'value',
                     'bpv_cat':'BPV types',
                     'bpv_type':'BP types'}, 
                 inplace=True)

# 改變類型
data['Item Code'].astype(str)

# 取代內容
## 分開
data['sex'].replace(0, 'Female',inplace=True)
data['sex'].replace(1, 'Male',inplace=True)
## 一起改
data['sex'].replace([0,1],['Female','Male'],inplace=True)
data['sex'].replace({0:'Female',1:'Male'},inplace=True)
```
### 新增
```python
#根據條件
##單條件
data['sig']= np.where(data['significant']==1,'*','')
##多條件
conditions=[(data['p']<=0.001),
           (data['p']<=0.01),
           (data['p']<=0.05)]
choices = ['***','**','*']
data['sig_group']= np.select(conditions, choices, default='')
```

### 根據column name挑選
挑出名字中有POP開頭
```python
df2 = df2[[col for col in df2.columns if col.startswith('POP')]]
```

## row 處理
### 根據index挑選要的row
```python
df=pd.read_pickle(data_path)
# load and delete specific data
delete_index=pd.read_pickle(deleted_data_path).index
df=df.loc[~df.index.isin(delete_index)]
```
