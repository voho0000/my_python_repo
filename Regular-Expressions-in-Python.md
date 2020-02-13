# Regular Expressions in Python
##### `re` `python` `datacamp`
[TOC]

目錄圖片

![](https://i.imgur.com/Vqw0KLw.png)



[此筆記有目錄版連結](https://hackmd.io/wsQCQpiPS9e4kc1e_S0-Gw?both)

Resource:[Regular Expressions in Python](https://www.datacamp.com/courses/regular-expressions-in-python)
## string manipulation
![](https://i.imgur.com/meII5Ox.png)

```python
my_string.lower()
my_string.upper()
my_string.capitalize()
```


```python
my_string.split(sep=" ",maxsplit=2)
my_string.rsplit(sep=" ",maxsplit=2)
```
![](https://i.imgur.com/40SHoBU.png)


```python

```
![](https://i.imgur.com/MiVUwCk.png)
![](https://i.imgur.com/McWGnrd.png)


```python
my_list=['my','name','is','xxx']
print(''.join(my_list))
```

### strip
```python
#leading space and trailing escape sequence are removed
my_string.strip() 
# remove characters from right end
my_string.rstrip()
# remove characters from left end
my_string.lstrip()
#remove $ word
my_string.strip('$') 
```
![](https://i.imgur.com/mSoDWpm.png)
![](https://i.imgur.com/faolh6a.png)

### find
```python
my_string.find()
```
![](https://i.imgur.com/LMOMiEb.png)

### index
```python
my_string.index()
```
![](https://i.imgur.com/QQutAq7.png)

### counting
```python
my_string.count('fruit')
```
![](https://i.imgur.com/Foo0Pfq.png)

### replace
```python
my_string.replace('old','new',count)
```
![](https://i.imgur.com/BDrJamy.png)

### 比較
- find如果找不到會給-1
```python
for movie in movies:
  	# Find if actor occurrs between 37 and 41 inclusive
    if movie.find("actor", 37, 42) ==-1:
        print("Word not found")
    # Count occurrences and replace two by one
    elif movie.count("actor") == 2:  
        print(movie.replace("actor actor", "actor"))
    else:
        # Replace three occurrences by one
        print(movie.replace("actor actor actor", "actor"))
```
- find index差異，都是回報所在的index，但找不到時find給-1，index會出現error

```python
# find寫法
for movie in movies:
  # Find the first occurrence of word
  print(movie.find("money",12, 51))

#index寫法
for movie in movies:
  try:
    # Find the first occurrence of word
  	print(movie.index("money",12, 51))
  except ValueError:
    print("substring not found")
```

## string formatting
- method for formatting
    - positional formatting
    - formatted string literals
    - template method
- format specifiers
    - e 10^3
    - d digits e.g. 4
    - f float e.g. 4.5353
### positional formatting
- placeholder replace by value:'text{}'.format(value)
    - str.format()
```python
# basic
print('ML provides {} the ability to learn {}'.format("system","automatically"))

# reordering
print("{2} has a friend called {0} and a sister called {1}".format("Betty","Linda","Daisy"))

# named placeholders
tool='us algoriums'
goal='patterns'
print("{title} try to find {aim} in the dataset".format(title=tool,aim=goal))
# named placeholders:dictionary
my_method={tool:'us algoriums',goal:'patterns'}
print("{data[tool]} try to find {data[goal]} in the dataset".format(data=my_method))

# format specifier {index:specifier}
print("Only {0:f}% of the {1} profuced worldwide is {2}!".format(0.51556,"data","analyzed"))
# 0.52%
print("Only {0:.2f}% of the {1} profuced worldwide is {2}!".format(0.51556,"data","analyzed"))

# formatting datetime
from datetime import datetime
print(datetime.now())
#output: datatime.datetime(2019,4,11,20,19,22,58582)
print("Today's date is {:%Y-%m-%d %H:%M}".format(datetime.now()))
```
![](https://i.imgur.com/IIiTCF1.png)

### Formatted string literal
- f-strings:f"literal string{expression}""
- type conversions:
    - !s(string version)
    - !r(string containing a printable representation, i.e. with quotes)
    - !a(some that !r but escape the non-ASCII characters)
- advantage
    - 可直接做計算 inline operation
    - 可直接叫function
```python
way='code'
method='learning python faster'
print(f"practice how to {way} is the best method for {method}")

# type conversion
name= "python"
print(f"python is called {name!r} due to a comedy series")
#output:python is called 'python' due to a comedy series

# dictionary用法有不一樣，要加""
familiy={"dad":"John"}
print("Is your dad called {family[dad]?}".format(family=family))
print(f"Is your dad called {family["dad"]?}")

# escape sequences
print ("my dad is called "John"") #output:error
print ("my dad is called \"John\"") #output:my dad is called "John")
#Backslashes are not allowed in f-strings
print(f"Is your dad called {family[\"dad\"]?}")#output:error

#inline operation
my_number=4
my_multiplier=7
print(f'{my_number} multiplied by {my_multiplier} is {my_number*my_multiplier}')

# function
def my_function(a,b):
    return a+b
print(f"sum up 10 and 20, the result is {my_function(10,20)}")
```
### template method
- simpler syntax
- slower than f-strings
- limited: don't allowed format specifier
- good when working with externally formatted strings

```python
from string import Template
#basic
my_string = Template('Data has been called $identifier')
my_string.substitute(identifier="oooo")

#use variable
job="data science"
name="sexist job"
my_string = Template('$title has been called $description')
my_string.substitute(title=job,description=name)
# use ${identifier} when valid characters following
my_string = Template('I find python very ${noun}ing')

# $ :如下圖
```
![](https://i.imgur.com/8NJJQ5i.png)

## regular expression
- string containing a combination of normal characters and special metacharacters that describes patterns to find text or positions within a text
- r at the begining indicateds a raw string
![](https://i.imgur.com/Ca6YcbE.png)
![](https://i.imgur.com/xhAckAv.png)
![](https://i.imgur.com/XZV5F4P.png)

![](https://i.imgur.com/hjoHyTm.png)
![](https://i.imgur.com/TcEHLJd.png)
![](https://i.imgur.com/BpMlOhs.png)
![](https://i.imgur.com/WcmEmUe.png)
![](https://i.imgur.com/W3DJslw.png)
![](https://i.imgur.com/9uJMqkO.png)


### Repetitions

![](https://i.imgur.com/M9qbC0A.png)
![](https://i.imgur.com/H2ZmM0l.png)

![](https://i.imgur.com/ler460B.png)
![](https://i.imgur.com/dh1qpu2.png)
![](https://i.imgur.com/G2jc9Gr.png)
![](https://i.imgur.com/AN5UluU.png)
![](https://i.imgur.com/2NUgE6D.png)


### Regex metacharacters
![](https://i.imgur.com/Q6R7759.png)
![](https://i.imgur.com/Iid5QAu.png)
![](https://i.imgur.com/3EYdVVS.png)
- ^:start
![](https://i.imgur.com/k2b31Oq.png)
- $:end
![](https://i.imgur.com/m2E6dht.png)
- / :escape
![](https://i.imgur.com/SA6ccRY.png)
- | : or
![](https://i.imgur.com/sZjr2GO.png)
![](https://i.imgur.com/hJUuiFq.png)
![](https://i.imgur.com/SDthWPJ.png)
- Or :[^] 否定
![](https://i.imgur.com/14JdM8r.png)

```python

```
### Greedy vs. non-greedy matching
![](https://i.imgur.com/xSwMMph.png)

![](https://i.imgur.com/5HWMcjo.png)

### group
![](https://i.imgur.com/KmJWcsM.png)
但只想抓人名就好，就把需要的資訊()起來
![](https://i.imgur.com/D0i2i9J.png)
![](https://i.imgur.com/SkIaGYo.png)
![](https://i.imgur.com/r14niS2.png)
![](https://i.imgur.com/wKX2hDL.png)
![](https://i.imgur.com/6O5Vz6D.png)
![](https://i.imgur.com/jkjFjPV.png)
![](https://i.imgur.com/6OCWxCE.png)

### pipe 注意事項/alternation
![](https://i.imgur.com/l2owUYs.png)
![](https://i.imgur.com/CTitiDw.png)
![](https://i.imgur.com/SzY8c3r.png)

### non-capture group
![](https://i.imgur.com/A6nbH6u.png)

![](https://i.imgur.com/4XQcGmk.png)
![](https://i.imgur.com/wNfwRdz.png)

### backreference

![](https://i.imgur.com/LCLywMl.png)
![](https://i.imgur.com/UFjAQR3.png)
- named group
![](https://i.imgur.com/edYux4d.png)
![](https://i.imgur.com/uKOmi8g.png)
- backreference
![](https://i.imgur.com/XzqMFTO.png)
![](https://i.imgur.com/LPtaV9M.png)
![](https://i.imgur.com/3FXO7Li.png)
- backreference in group
![](https://i.imgur.com/S84ZCQV.png)
![](https://i.imgur.com/gcPuF2Z.png)

### look around
![](https://i.imgur.com/KRADra8.png)
![](https://i.imgur.com/xm5LNop.png)

![](https://i.imgur.com/hQrRR0o.png)
![](https://i.imgur.com/LeiXA6R.png)
![](https://i.imgur.com/IiqNNK8.png)
![](https://i.imgur.com/fHsjsDX.png)
![](https://i.imgur.com/uwRymmx.png)


```python

```


```python

```


```python

```


```python

```


