# Python Data Science Toolbox

[TOC]
[有目錄版HackMD筆記](https://hackmd.io/bfInqB8mT869ngs3DdpL2g?both)

**Resource :**
[Python Data Science Toolbox(Part 1)](https://www.datacamp.com/courses/python-data-science-toolbox-part-1)
[Python Data Science Toolbox(Part 2)](https://www.datacamp.com/courses/python-data-science-toolbox-part-2)


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
