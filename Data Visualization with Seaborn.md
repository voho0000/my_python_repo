
# Data Visualization with Seaborn 

[TOC]

[此筆記Hackmd連結(有目錄)](https://hackmd.io/SwgrGQ5dRF-Ar-v8cQ52-A?both)

## cheat sheet
{%pdf https://datacamp-community-prod.s3.amazonaws.com/f9f06e72-519a-4722-9912-b5de742dbac4 %}

## python繪圖套件總
![](https://i.imgur.com/7M0OYuX.png)

## histogram
```python
import pandas as pd
df=pd.read_csv('wines.csv')
df['alcohol'].plot.his
```

```python
import seaborn as sns

sns.distplot(df['alcohol'])
```
![](https://i.imgur.com/nzRSxFk.png)
- custimized(rug軸鬚圖)
```python
sns.distplot(df['alcohol'],kde=False,bins=10)
sns.distplot(df['alcohol'],hist=False,rug=True)
sns.distplot(df['alcohol'],hist=False,rug=True,kde_kws={'shade':True})

plt.show()
```
![](https://i.imgur.com/bluY7Ex.png)


## regression plot
```python
sns.regplot(x='alcohol',y='pH',data=df)
```
![](https://i.imgur.com/KKo7L2U.png)

### regplot vs lmplot
```python
```
![](https://i.imgur.com/lg4oc7l.png)

### lmplot facet
```python
sns.lmplot(x='quality',
           y='alcohol',
           data=df,
           hue='type',
           col='type',
           row='type')
```
![](https://i.imgur.com/knnwIAB.png)
## style
```python
for style in ['white','dark','whitegrid','darkgrid','ticks']
sns.set_style(style)
sns.distplot(df['Tuition'])
plt.show()
```
![](https://i.imgur.com/Fz6P6UY.png)
### removing axes with despine()
```python
sns.set_style('white')
sns.distplot(df['Tuition'])
sns.despine(top=True,right=True)
```
### color
```python
sns.set(color_code=True)
sns.distplot(df['Tuition'],color='g')
```
- palettes
    - sns.palplot():display a palette
    - sns.color_palette():return the current palette
```python
for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.distplot(df['Tuition'])
```
![](https://i.imgur.com/li3NoG6.png)

```python
for p in sns.palettes.SEABORN_PALETTES:
    sns.set_palette(p)
    sns.palplot(sns.color_palette())
    plt.show()
```

```python
sns.palplot(sns.color_palette('Paired',12))
sns.palplot(sns.color_palette('blue',12))
```
![](https://i.imgur.com/xZTzW9O.png)
## custimized with matplot
### matplotlib axes
```python
fig, ax = plt.subplots()
sns.distplot(df['Tuition'], ax=ax)
ax.set(xlabel='Tuition 2013-4',
       ylabel='Distribution',
       xlim=(0,50000),
       title='2013-4 Tuistion')
```
### combining plot
It is possible to combine and configure multiple plots
- sharey:share the y axis label
- pandas query用法
    - df[df['total_bill'] > 20]
    - df.query('total_bill > 20')
- axvline : denote the maximum amount we can budget for tuition
```python
fig, (ax0, ax1) = plt.subplots(
nrows=1,ncols=2, sharey=True, figsize=(7,4))

sns.distplot(df['Tuition'], ax=ax0)
sns.distplot(df.query(
'State == "MN"')['Tuition'], ax=ax1)

ax1.set(xlabel="Tuition (MN)", xlim=(0, 70000))
ax1.axvline(x=20000, label='My Budget', linestyle='--')
ax1.legend()

```
![](https://i.imgur.com/IHMKV3s.png)


```python
# Create a figure and axes. Then plot the data
fig, ax = plt.subplots()
sns.distplot(df['fmr_1'], ax=ax)

# Customize the labels and limits
ax.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500), title="US Rent")

# Add vertical lines for the median and mean
ax.axvline(x=median, color='m', label='Median', linestyle='--', linewidth=2)
ax.axvline(x=mean, color='b', label='Mean', linestyle='-', linewidth=2)

# Show the legend and plot the data
ax.legend()
plt.show()
```
![](https://i.imgur.com/RXlwJyg.png)

```python
# Create a plot with 1 row and 2 columns that share the y axis label
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharey=True)

# Plot the distribution of 1 bedroom apartments on ax0
sns.distplot(df['fmr_1'], ax=ax0)
ax0.set(xlabel="1 Bedroom Fair Market Rent", xlim=(100,1500))

# Plot the distribution of 2 bedroom apartments on ax1
sns.distplot(df['fmr_2'], ax=ax1)
ax1.set(xlabel="2 Bedroom Fair Market Rent", xlim=(100,1500))

# Display the plot
plt.show()
```
![](https://i.imgur.com/t1rV469.png)
## categorical plot types
![](https://i.imgur.com/JBdZZB5.png)
![](https://i.imgur.com/VZa09ML.png)
![](https://i.imgur.com/fKHNMhN.png)

```python
sns.stripplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges',
              jitter=True)
```
![](https://i.imgur.com/tUHEbpC.png)

```python
sns.swarmplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges')
```
![](https://i.imgur.com/ML8dm0H.png)

```python
sns.boxplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges')
```
![](https://i.imgur.com/RiE0JXV.png)

```python
sns.violinplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges')
```
![](https://i.imgur.com/5HWeLZd.png)
```python
sns.lvplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges')
```
![](https://i.imgur.com/UPJT26u.png)

```python
sns.barplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges',
              hue='Region')
```
![](https://i.imgur.com/VTnOe5u.png)

```python
sns.pointplot(data=df, 
              y='DRG definition',
              x='Average Covered Charges',
              hue='Region')
```
![](https://i.imgur.com/ZoNyVxE.png)

```python
sns.countplot(data=df, 
              y='DRG definition',
              hue='Region')
```
![](https://i.imgur.com/1i6H7qG.png)

## regression plot
### regplot
```python
sns.regplot(data=df, x='temp',
            y='total_rentals',
            marker='+')
```
![](https://i.imgur.com/0CH0Jug.png)
### residplot
- residplot is a useful plot for understanding the appropriateness of a regression model
- order: support polynomial regression using the order parameter
- x_estimtor:useful for hightlighting trend
```python
sns.residplot(data=df, x='temp',
            y='total_rentals')
```
![](https://i.imgur.com/8n8pZ5g.png)

```python
sns.residplot(data=df, x='temp',
            y='total_rentals',order=2)
```
![](https://i.imgur.com/iuXfdJr.png)

### regplot
```python
sns.regplot(data=df, x='mnth',
            y='total_rentals',
            x_jitter=.1,
            order=2)
```
![](https://i.imgur.com/LJrIKVy.png)

- x_estimtor:useful for hightlighting trend
```python
sns.regplot(data=df, x='mnth',
            y='total_rentals',
            x_estimator=np.mean,
            order=2)
```
![](https://i.imgur.com/RagmNUh.png)
```python
sns.regplot(data=df, x='mnth',
            y='total_rentals',
            x_bins=4)
```
![](https://i.imgur.com/w5ARJLL.png)

## Matrix plot
### heatmap
- seaborn's heatmap requires data to be in a gird format
- pandas crosstab() is frequently used to manipulate the data
    - pd crosstab用法
![](https://i.imgur.com/A0pNPpv.png)

```python
pd.crosstab(df['mnth'],df['weekday'],values=df['total_rentals',aggfunc='mean'].round(0)
```
![](https://i.imgur.com/SFzicyc.png)

```python
sns.heatmap(pd.crosstab(df['mnth'],df['weekday'],values=df['total_rentals',aggfunc='mean'])
```
![](https://i.imgur.com/2kQMvaF.png)

- annot :要不要標原本的數字
- fmt :ensure the result are displayed as integers
- cmap : color yellow green blue
- cbar : the color barr
- linewidths : spacing between cells
```python
sns.heatmap(df_crosstab, annot=True,
            fmt='d',cmap='YlGnBu',
            cbar=False,linewidths=.5)
```
![](https://i.imgur.com/JTdXI3s.png)
### centering a heatmap
- sept and saturday of 9 and 6 using the numeric indices for these values
```python
sns.heatmap(df_crosstab, annot=True,
            fmt='d',cmap='YlGnBu',
            cbar=True,
            center=df_crosstab.loc[9,6])
```
![](https://i.imgur.com/7I0om9A.png)

### correlation heatmap
```python
sns.heatmap(df.corr())
```
![](https://i.imgur.com/SUaQ0yv.png)

## facetgrid
![](https://i.imgur.com/XJiNsBh.png)
![](https://i.imgur.com/Hq3DZun.png)

```python
g = sns.FacetGrid(df, col='HIGHDEG')
g.map(sns.boxplot, 'Tuition',
      order=['1','2','3','4'])
```
![](https://i.imgur.com/NU2ybAd.png)

### factorplot
- simpler way to use a FacetGrid for categorical data
```python
sns.factorplot(x='Tuition',data=df,
               col="HIGHDEG",
               kind='box')
```
![](https://i.imgur.com/NU2ybAd.png)

![](https://i.imgur.com/Ifr4pqu.png)

### lmplot
```python
sns.lmplot(data=df, x='Tuition',
           y='SAT_AVG_ALL',
           col='HIGHDEG',
           fit_reg=False)
```
![](https://i.imgur.com/aXICCh7.png)
## Jointgrid
![](https://i.imgur.com/odnddPH.png)

```python
```

```python
```

```python
```

## 實作問題
### facetgrid legend顏色不見處理
```python
#加顯著
# annotate * for significant data
def sig(x,y, **kwargs):
    axss = sns.pointplot(x,y,**kwargs,dodge=True)
    for c in axss.collections:
        for of in c.get_offsets():
            axss.annotate(data[data['Difference']==of[1]]['sig'].values[0], of)
#加legend顏色
#handle the problem of facetgrid legend color missing
hue_order=data['group'].unique()
labels=hue_order
colors=sns.color_palette("deep").as_hex()[:len(labels)]
handles=[patches.Patch(color=col,label=lab) for col, lab in zip(colors, labels)]
#畫圖+存圖
def plotfig(x):
    g = sns.FacetGrid(data[data['cutoff']==x],
                      col='BPV types',row='BP types',
                      hue='group',size=4,
                      sharex=False,sharey=False,
                      palette='deep')
    g = g.map(sig,'month','Difference')
    plt.legend(handles=handles,title='group',loc='center left',bbox_to_anchor=(1,0.75))
    g.savefig('result'+str(x)+'.png') 

```
