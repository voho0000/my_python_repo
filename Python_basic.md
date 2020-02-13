# Python語法全
###### tags: `python`
[TOC]

[有目錄版HackMD筆記](https://hackmd.io/d6IQGH4HT2uhiWN6G9tToA?both)

**Resource :**
[Intro to Python for Data Science](https://www.datacamp.com/courses/intro-to-python-for-data-science)
[Intermediate Python for Data Science](https://www.datacamp.com/courses/intermediate-python-for-data-science)
[Python Data Science Toolbox (Part 1)](https://www.datacamp.com/courses/python-data-science-toolbox-part-1)

# 擴充
## jupyter
### [Jupyter Notebook Extensions](https://towardsdatascience.com/jupyter-notebook-extensions-517fa69d2231?fbclid=IwAR1aD64oYwk2xGm6byXzCm9qux_c98nJpfgzRVAqxJiVhXrwZRFDY2Vd-0Y)
```
pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
```
## anaconda/miniconda
### 環境設置
[環境設置(Linux)](https://hackmd.io/L6K0gaxxTZa2AQW7CMi66g)此篇最前面的環境設置篇

# 工具
## 路徑相關
### os.path
```python
#返回文件名
os.path.basename(path)

#返回文件路徑
os.path.dirname(path)

#把路徑分割成dirname和basename
os.path.split(path)

#取得所有檔案與子目錄名稱
os.listdir(path)

#取得絕對路徑
os.path.abspath('.') 
```
### glob
選取特定檔名+路徑
```python
import glob
files=glob.glob('./../raw_data/nil/gold standard/20161223 GK Contour/' + '*+.bmp')
#bwt如果關鍵字在中間'*+.bmp*'，後面多加一個*
for i in files:
    print(i.split('+')[0]+i.split('+')[1])
```
### shutil 文件移動
```python
#複製文件：
shutil.copyfile("oldfile_path","newfile_path") #oldfile和newfile都只能是文件
shutil.copy("oldfile_path","newfile_path") #oldfile只能是文件夾，newfile可以是文件，也可以是目標目錄

#複製文件夾：
shutil.copytree("olddir_path","newdir_path") #olddir和newdir都只能是目錄，且newdir必須不存在

#重命名文件（目錄）
os.rename("oldname","newname") #文件或目錄都是使用這條命令

#移動文件（目錄）
shutil.move("oldpos","newpos") 
shutil.move("D:/知乎日報/latest/abc.pdf", "D:/知乎日報/past/")
```

## 測速
```python
t0=time.time()
listx=list(x-structurerange[0][0] for x in (list(zip(*conv2))[0]))
listy=list(y-structurerange[0][1] for y in (list(zip(*conv2))[1]))
t1=time.time()
t1-t0
```
## 資料一覽
### 算分類
```python
# Import pandas
import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)
```
#### dictionary版本
##### 有error版
```python
# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Add try block
    try:
        # Extract column from DataFrame: col
        col = df[col_name]
        
        # Iterate over the column in dataframe
        for entry in col:
    
            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1
    
        # Return the cols_count dictionary
        return cols_count

    # Add except block
    except:
        print('The DataFrame does not have a ' + col_name + ' column.')

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)
```
```python
 # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' + col_name + ' column.') 
```
##### 有function+chunk版
```python
# Define count_entries()
def  count_entries(csv_file,c_size,colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file,chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries( 'tweets.csv',10,'lang')

# Print result_counts
print(result_counts)

```
## print
### .format
```python
mydict = {1: 'a', 2: 'b'}
for i, (k, v) in enumerate(mydict.items()):
    print("index: {}, key: {}, value: {}".format(i, k, v))

# which will print:
# -----------------
# index: 0, key: 1, value: a
# index: 1, key: 2, value: b
```
### print dictionary (from net)
```python
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
```
## if 
### exist
```python
if 'myVar' in locals():
  # myVar exists.
To check the existence of a global variable:

if 'myVar' in globals():
  # myVar exists.
To check if an object has an attribute:

if hasattr(obj, 'attr_name'):
  # obj.attr_name exists.
```
# regular expression
[Regular Expressions in Python](/wsQCQpiPS9e4kc1e_S0-Gw)


# Introduction to python
## list
#### create a list
```python
a = "is"
b = "nice"
my_list = ["my", "list", a, b]
print(my_list)
```
#### List of lists
```python
# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# house information as list of lists
house = [["hallway", hall],["kitchen", kit],["living room", liv],["bedroom",bed],["bathroom",bath]]   
```
#### subset the list
```python
x = ["a", "b", "c", "d"]
x[1]
x[-3] # same result!
````

``` python
my_list[start:end]
x = ["a", "b", "c", "d"]
x[1:3]
```

#### subset the list of list
```python
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]
x[2][0]
x[2][:2]
```

#### replace
```python
x = ["a", "b", "c", "d"]
x[1] = "r"
x[2:] = ["s", "t"]
```

#### extend a list
```python
x = ["a", "b", "c", "d"]
y = x + ["e", "f"]
```
補充
```python
【append範例】
Ya =['a', 'b', 'c']    #相當於綠色抽屜櫃
Yaa = [1, 2, 3]     #相當於塑膠袋
Ya.append(Yaa)  #把一整包塑膠袋放入綠色抽屜櫃的最後一格櫃子裡  
print Ya                #看結果用
---
['a', 'b', 'c', [1, 2, 3]]   #整包塑膠袋被放進了最後一格櫃子

【extend範例】

Ya =['a', 'b', 'c']    #相當於綠色抽屜櫃
Yaa = [1, 2, 3]     #相當於塑膠袋
Ya.extend(Yaa)  #整包塑膠袋打開後，將裡面的剪刀、眼鏡、牛奶，依序放入綠色抽屜櫃，總需三格櫃子  
print Ya                #看結果用
---
['a', 'b', 'c', 1, 2, 3]   #塑膠袋裡的東西都被取出，剪刀、眼鏡、牛奶各別依序被放入最後三格櫃子
```

#### delite list elements
```python
x = ["a", "b", "c", "d"]
del(x[1])
```

## Fuction
### Basics
##### Sorted
```python
# Sort full in descending order: full_sorted
full_sorted=sorted(full,reverse=True)
```
### method
#### string
```python
# string to experiment with: room
room = "poolhouse"
# Use upper() on room: room_up
room_up=room.upper()
# Print out the number of o's in room
print(room.count("o"))
```
#### list method:index, count, append, reverse, remove
```python
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 14.5 appears in areas
print(areas.count(14.5))

# Use append twice to add poolhouse and garage size
areas.append(24.5)

# Reverse the orders of the elements in areas
areas.reverse()

# Remove the element
areas.remove(24.5)
# not sure
areas[2].remove()

```

## package
### import
```python
r=0.43
# Import the math package
import math
# Calculate C
C = 2*math.pi*r

```
```python
# Import radians function of math package
from math import radians

# Travel distance of Moon over 12 degrees. Store in dist.
dist=r*radians(12)
```

## Numpy
### Feature of Numpy
**type coercion**
numpy arrays cannot contain elements with different types. If you try to build such a list, some of the elements' types are changed to end up with a homogeneous list. This is known as type coercion.
### create
#### list to np.array
##### 1D
```python
# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Import the numpy package as np
import numpy as np

# Create a numpy array from baseball: np_baseball
np_baseball=np.array(baseball)
# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * 0.0254
```
##### 2D
```python
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
            
# Create a 2D numpy array from baseball: np_baseball
np_baseball=np.array(baseball)
````

### select
####
```python
# Create the light array
light=bmi<21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])
```
##### 
```python
# Convert positions and heights to numpy arrays: np_positions, np_heights
np_heights=np.array(heights)
np_positions=np.array(positions)

# Heights of the goalkeepers: gk_heights
gk_heights=np_heights[np_positions == "GK"]

# Heights of the other players: other_heights
other_heights=np_heights[np_positions != "GK"]
```

#### subset (1D same as list)
##### 1D
```python
x = ["a", "b", "c"]
x[1]

np_x = np.array(x)
np_x[1]
```
##### 2D
```python
# regular list of lists
x = [["a", "b"], ["c", "d"]]
[x[0][0], x[1][0]]

# numpy
import numpy as np
np_x = np.array(x)
np_x[:,0]
```

#### Calculate
##### Average versus median
``` python
# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0],np_baseball[:,1])
print("Correlation: " + str(corr))
```
##### 2D Arithmetic
```python
# Create np_baseball (3 cols)
np_baseball = np.array(baseball)
# Print out addition of np_baseball and updated
print(np_baseball+updated)
# Create numpy array: conversion
conversion=np.array([0.0254, 0.453592, 1])
# Print out product of np_baseball and conversion
print(np_baseball * conversion)
```

# Intermediate python
## Matplotlib
### line plot
```python
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year,pop)

# Display the plot with plt.show()
plt.show()
```
### Scatter Plot
```python
# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)
# Put the x-axis on a logarithmic scale
plt.xscale('log')
# Show plot
plt.show()
```
#### size+color+transparency
```python
# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s = np_pop,c = col,alpha=0.8)

```

###  histogram
``` python
# Create histogram of life_exp data
plt.hist(life_exp) 

# Display histogram
plt.show()
```

plt.show() displays a plot; 
plt.clf() cleans it up again so you can start afresh
```python
# Build histogram with 20 bins
plt.hist(life_exp,bins=20)

# Show and clean up again
plt.show()
plt.clf()
```
### Custimazation
#### axix and title
```python
# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()
```
#### Ticks
```python
# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val,tick_lab)

# After customizing, display the plot
plt.show()
```

#### text+grid
```python
# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()
```
## Dictionary
### Create
```python
my_dict = {
   "key1":"value1",
   "key2":"value2",
}
```
#### from list to dict
```python
# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)
```
#### from  list to dict to df
```python
# Import the pandas package
import pandas as pd

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# Print the head of the DataFrame
print(df.head())
```
### Manipulation
#### Access
```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])
```
#### Add
```python
# Add italy to europe
europe["italy"]="rome"

# Print out italy in europe
print("italy" in europe) #Result:True
```
```python
default_data['item3'] = 3

default_data.update({'item4': 4, 'item5': 5})
```

#### remove
```python
# Remove australia
del(europe['australia'])    
```

#### Dictionariception
```python
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe["france"]["capital"])

# Create sub-dictionary data
data={"capital":'rome','population':59.83}

# Add data to europe under key 'italy'
europe["italy"]=data
```

## Pandas
### Pandas in Dictionary
#### Dictionary to pandas
```python
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict={
    "country":names,
    "drives_right":dr,
    "cars_per_cap":cpc
}

# Build a DataFrame cars from my_dict: cars
cars=pd.DataFrame(my_dict)
```
#### rowname
```python
import pandas as pd

# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index=row_labels

# Print cars again
print(cars)
```

#### CSV to dataframe
```python
# Import pandas as pd
import pandas as pd

# Import the cars.csv data: cars
cars=pd.DataFrame(pd.read_csv("cars.csv"))

# Print out cars
print(cars)
```
##### row label
Specify the index_col argument inside pd.read_csv(): set it to 0, so that the first column is used as row labels.
```python
# Import pandas as pd
import pandas as pd

# Fix import by including index_col
cars = pd.read_csv('cars.csv',index_col=0)

# Print out cars
print(cars)
```

### select
#### Square Brackets 
The single bracket version gives a Pandas Series, the double bracket version gives a Pandas DataFrame.
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars["country"])

# Print out country column as Pandas DataFrame
print(cars[["country"]])

# Print out DataFrame with country and drives_right columns
print(cars[["country","drives_right"]])
```
```python
cars[0:5]
```

#### loc and iloc
loc and iloc also allow you to select both rows and columns from a DataFrame.
loc is label-based, which means that you have to specify rows and columns based on their row and column labels. iloc is integer index based.
```python
cars.loc['RU']
cars.iloc[4]

cars.loc[['RU']]
cars.iloc[[4]]

cars.loc[['RU', 'AUS']]
cars.iloc[[4, 1]]
```
```python
cars.loc['IN', 'cars_per_cap']
cars.iloc[3, 0]

cars.loc[['IN', 'RU'], 'cars_per_cap']
cars.iloc[[3, 4], 0]

cars.loc[['IN', 'RU'], ['cars_per_cap', 'country']]
cars.iloc[[3, 4], [0, 1]]
```
```python
cars.loc[:, 'country']
cars.iloc[:, 1]

cars.loc[:, ['country','drives_right']]
cars.iloc[:, [1, 2]]
```
#### select with boolean
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Import numpy, you'll need this
import numpy as np

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]

# Print medium
print(medium)
```

## Logic
### Equality
```python
# Comparison of booleans
True==False

# Comparison of integers
-5 * 15!=75

# Comparison of strings
"pyscript"=="PyScript"

# Compare a boolean with an integer
True==1
```
### Boolean
and
or
not
#### numpy array
logical_and()
logical_or()
logical_not()

## if elif else
```python
area = 10.0
if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")
```
符合條件不做任何事的話要加pass，不能空白
```python
if a=0:
    #do nothing
    pass
else:
    b=100
```
### if isinstance
```python
a = 2
isinstance (a,int)
# output:True
```
```python
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
```

## Loop
### Loop over dictionary
```python
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for x,y in europe.items():
    print("the capital of "+str(x)+" is "+str(y))
```
#### iterate index
```python
mydict = {1: 'a', 2: 'b'}
for i, (k, v) in enumerate(mydict.items()):
    print("index: {}, key: {}, value: {}".format(i, k, v))

# which will print:
# -----------------
# index: 0, key: 1, value: a
# index: 1, key: 2, value: b
```

#### print dictionary (from net)
```python
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
```

### Loop over Numpy array
#### 1D
```python
for x in np_height:
    print(str(x)+" inches")  
```
#### 2D
```python
for val in np.nditer(np_baseball):
    print(val)  
```

### Loop over DataFrame
```python
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)
```
```python
# Adapt for loop
for lab, row in cars.iterrows() :
     print(lab+": "+str(row["cars_per_cap"]))
```
#### add column
```python
# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows() :
    cars.loc[lab, "COUNTRY"]=row["country"].upper()
#same result
cars["COUNTRY"] =  cars["country"].apply(str.upper)
```
# Python Data Science Toolbox
## Write Function
### Basic arcchitecture
![](https://i.imgur.com/3GvuimU.png)

### Create
simple
```python
def square():
    new_value = 4 ** 2
    return new_value
```
Single parameter
```python
# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'
    # Replace print with return
    return shout_word
# Pass 'congratulations' to shout: yell
yell = shout('congratulations')
# Print yell
print(yell)
```
multiparameter
```python
# Define shout with parameters word1 and word2
def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'   
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'   
    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1 + shout2
    # Return new_shout
    return new_shout
# Pass 'congratulations' and 'you' to shout: yell
yell = shout('congratulations', 'you')
# Print yell
print(yell)
```
### Tuple
類似list的存在，但建立後就無法修改
而且list是用[]，但tuple是用()
以tuple作為返回多value的儲存地，可以被unpack
```python
# Unpack nums into num1, num2, and num3
num1, num2, num3=nums_tuple

# Construct even_nums
even_nums=(2, num2, num3)
```
```python
# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):
    """Return a tuple of strings"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)
```

### global/nonlocal
#### global
```pyhon
# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():
    """Change the value of the global variable team."""

    # Use team in global scope
    global team

    # Change the value of team in global: team
    team='justice league'
# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)
```
#### nonlocal
```python
# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word=word*2
    
    # Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        # Use echo_word in nonlocal scope
        nonlocal echo_word
        
        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word+'!!!'
    
    # Call function shout()
    shout()
    
    # Print echo_word
    print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout('hello')
```
### Nested function (double argument)

```python
# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))
```

```python
# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo

# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice=echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))
```

### flexible argument (*arg)/(**kwargs)
#### flexible argument (*arg)
Flexible arguments enable you to pass a variable number of arguments to a function. 
```python
# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge=""

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)

```
#### flexible argument (**kwargs)
What makes **kwargs different is that it allows you to pass a variable number of keyword arguments to functions
```python
# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")

# First call to report_status()
report_status(name="luke", affiliation="jedi", status="missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")

```

### Lambda+map/filter
```python
# Define echo_word as a lambda function: echo_word
echo_word = lambda word1, echo:word1 * echo


def echo_word(word1, echo):
    """Concatenate echo copies of word1."""
    words = word1 * echo
    return words

```
#### map
terate所有的sequence的元素並將傳入的function作用於元素，最後以List作為回傳值。
```python
# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda a:a+'!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list=list(shout_spells)

# Convert shout_spells into a list and print it
print(shout_spells_list)
```
#### filter
以傳入的boolean function作為條件函式，iterate所有的sequence的元素並收集 function(元素) 為True的元素到一個List
```python
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda a:len(a)>6, fellowship)

# Convert result to a list: result_list
result_list=list(result)

# Convert result into a list and print it
print(result_list)
```
#### reduce
The reduce() function is useful for performing some computation on a list and, unlike map() and filter(), returns a single value as a result. To use reduce(), you must import it from the functools module.
必須傳入一個binary function(具有兩個參數的函式)，最後僅會回傳單一值。

reduce會依序先取出兩個元素，套入function作用後的回傳值再與List中的下一個元素一同作為參數，以此類推，直到List所有元素都被取完。
```python
# Import reduce from functools
from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1,item2:item1+item2, stark)

# Print the result
print(result)
```

### Error handling (try-except+raising an error)
    value error(要>0,卻是-1), type error (要int卻是str)
![](https://i.imgur.com/wBV0tO1.png)

#### error and exceptions
```python
def sqrt(x):
    try:
        return x**0.5
    except:
        print('x must be an int or float')
```

#### raising an error (if)
```python
def sqrt(x):
    if x<0:
        raise ValueError('x must be non-negative')
    try:
        return x**0.5
    except TypeError:
        print('x must be an int or float')
```

### example (filter words)

#### select word ('RT')
```python
# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: x[0:2] == 'RT', tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)
```

##  iterators
- iterables: (可被iter的)
    - list, string, dictionary, file connectionapply 
    - iter() in an iterable to create iterators
- iterators: (被iter後的)
    - apply next() to get next iterators
```python
word='Data'
it=iter(word)
next(it)
#output: D 
next(it)
#output: a 
```
*會unpack 所有iterators，只能進行一次
```python
word='Data'
it=iter(word)
print(*it)
#output: D a t a
```
也可用在file

```python
file=open('file.txt')
it=iter(file)
print(next(it))
```

```python
# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)
```

### enumerate
class:enumerate
直接產生的是tuple
```python
for index, value in enumerate(avengers):
    print(index,value)
    
for index, value in enumerate(avengers,start=10):
    print(index,value)
```
```python
# Create a list of strings: mutants
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)
#output是[(0,'charles xavier'),(1,'bobby drake')...]

# Unpack and print the tuple pairs
for index1,value1 in enumerate(mutants):
    print(index1, value1)
```

### zip
class:zip
被包起來的是tuple
*in a call to zip() to unpack the tuples produced by zip()
*完，如果還要zip，要重zip一遍
```python
avengers=['hawkeye','iron man','thor','quicksilver']
names=['barton','stark','odinson','maximoff']
z=zip(avengers,names)
# 1
z_list=list(z)
print(z_list)

# 2
print(*z)

#3
for z1, z2 in zip(avengers,names)
    print(z1,z2)
```
```python
# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants,aliases,powers))
# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants,aliases,powers)

# Print the zip object
print(mutant_zip)
#output:<zip object at 0x7f6e0193c1c8>

# Unpack the zip object and print the tuple values
for value1, value2, value3 in zip(mutants,aliases,powers):
    print(value1, value2, value3)

```
#### zip and unzip (變回去)
```python
# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 =  zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)

```

### chunk
chunk 可以想像成是小dataframe
```python
# Initialize an empty dictionary: counts_dict
counts_dict={}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('tweets.csv',chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)
```

## List comprehension (one line of code)
**[[output expression] for iterator variable in iterable]**

![](https://i.imgur.com/9mL25um.png)
```python
nums=[12,8,21,3,16]
new_nums=[]
for num in nums:
    new_nums.append(num+1)

#List comprehension
new_nums=[num+1 for num in nums]
```

### nested 
```python
pair_1=[]
for num1 in range(0,1):
    for num2 in range(6,8):
        pair_1.append(num1,num2)

#List comprehension
pair_2=[(num1,num2)for num1 in range(0,2) for num2 in range(6,8)]
```
**[[output expression] for iterator variable in iterable]**
```python
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)

```

### conditionals
[ output expression for iterator variable in iterable if predicate expression ]
[ output expression+condition on output for iterator variable in iterable+condition on iterable ]
```python
[num**2 for num in range(10) if num%2==0]
#output:[0,4,16,36,64]
```

```python
[num**2 if num%2==0 else 0 for num in range(10)]
#output:[0,0,4,0,16,0,36,0,64,0]
```

### dictionary comprehention
`pos_neg={num:-num for num in range(9)}`

### generators
list comprehension returns a list 
generators return a generator object
both can be iterated over
原因是list被創造時會吃記憶體(如`[num for num in range(10**100000)]`會爆掉)
generator 不會


```python
result=(num for num in range(6))
print(list(result))
```

#### yield 
生成器函数（generator function）和生成器（generator）
如果一个函数包含 yield 表达式，那么它是一个生成器函数；调用它会返回一个特殊的迭代器，称为生成器。
```python
def func():
    return 1

def gen():
    yield 1

print(type(func))   # <class 'function'>
print(type(gen))    # <class 'function'>

print(type(func())) # <class 'int'>
print(type(gen()))  # <class 'generator'>
```
