# Writing Efficient Python Code
[Resource : Writing Efficient Python Code](https://campus.datacamp.com/courses/writing-efficient-python-code)

[TOC]

![](https://i.imgur.com/2RgnVm1.png)
## definition
- Course goals
    - write clean, fast and efficient code
    - how to profile your code for bottlenecks
    - how to eliminate bottlenecks and bad design patterns
- what is efficient code
    - minimal completion time
    - minimal resource consumption(small memory usage)
- write efficient python code
    - focus on readability
    - using python's construct as intended (ie. pythonic )
    ![](https://i.imgur.com/hSePX7H.png)
![](https://i.imgur.com/4De75ZI.png)


## Building with built-ins
![](https://i.imgur.com/a0Eltxf.png)
### range
- 
```pthon
# Use range() to create a list of arrival times (10 through 50 incremented by 10). 
#Create the list arrival_times by unpacking the range object.
#You can unpack the range object by placing a star character (*) in front of it
arrival_times = [*(range(10, 51, 10))]
```

### enumerate
- 可以用start=5，來使index起始值為5
![](https://i.imgur.com/NZ37uWQ.png)

### map/+lambda
![](https://i.imgur.com/RzjwJyG.png)
![](https://i.imgur.com/J3y2LpE.png)
```python
# Map the welcome_guest function to each (guest,time) pair
welcome_map = map(welcome_guest, guest_arrivals)

guest_welcomes = [*welcome_map]
print(*guest_welcomes, sep='\n')
#直接print(welcome_map)會出現<map object at 0x7f305134d358>
```

## The power of NumPy arrays
### 跟list比較
跟list比較，list不能直接做數學計算
- list的方式比較沒效率
![](https://i.imgur.com/cDX59ke.png)
- 2D 這兩種如何做index
![](https://i.imgur.com/ZSklNA1.png)
- numpy array 支持boolean indexing
![](https://i.imgur.com/ul1HylC.png)
- list 不支持boolean indexing
![](https://i.imgur.com/oL0LEUZ.png)

## examine runtime
### %timeit
![](https://i.imgur.com/dyz3sEx.png)
![](https://i.imgur.com/A0DpIuo.png)
![](https://i.imgur.com/Oxcq80E.png)
![](https://i.imgur.com/73mgFOr.png)
![](https://i.imgur.com/Zs6SWXH.png)
#### 比較不同寫法差異
![](https://i.imgur.com/e1C3Pby.png)
![](https://i.imgur.com/in0StXh.png)

## Code profiling for runtime
```python
pip install line_profiler
```
- 要先叫出來，才能開始用lprun
    - -f是指後面那個是function
![](https://i.imgur.com/oQ2oCcc.png)
- hits:那行被跑了幾次
![](https://i.imgur.com/dZTodOF.png)

## Code profiling for memory usage
### sys
- 看占用的空間(bytes)
![](https://i.imgur.com/XcjslCv.png)
### memory_profiler
```python
pip install memory_profiler
```
![](https://i.imgur.com/sIri8t7.png)
![](https://i.imgur.com/uOf3kDx.png)

## Efficiently combining, counting, and iterating

![](https://i.imgur.com/WIyyYR1.png)
### counting
![](https://i.imgur.com/HPo0dNc.png)
#### counter
![](https://i.imgur.com/LCjRdIu.png)
### itertool
![](https://i.imgur.com/XQv4w2a.png)
### combine
![](https://i.imgur.com/wRuVlYO.png)
#### combination
![](https://i.imgur.com/ZAs91Hi.png)

## Set theory
![](https://i.imgur.com/GNSx3gt.png)
### set
- 使用前都要先set()，才能套
![](https://i.imgur.com/oLQy79x.png)
### difference
![](https://i.imgur.com/7jvLm5H.png)
![](https://i.imgur.com/qGKfoVz.png)
![](https://i.imgur.com/k7l64sR.png)
### union
![](https://i.imgur.com/ktfrsJ9.png)
### in 
- 速度比較
- ![](https://i.imgur.com/fzZzoUo.png)
### unique
- for loop版
 ![](https://i.imgur.com/TqB8BID.png)
- set版
![](https://i.imgur.com/pZmaYjF.png)



## Eliminating loops

![](https://i.imgur.com/hiF7OW4.png)
### for loop->list comprehension or map
![](https://i.imgur.com/8oTqXa2.png)
### for loop->numpy array
![](https://i.imgur.com/lly4SJz.png)

## Writing better loops
### holistic conversion
![](https://i.imgur.com/RBpbgqy.png)

## Intro to pandas DataFrame iteration
### iterate 在iloc跟iterrows比較
![](https://i.imgur.com/0pWYNVw.png)
![](https://i.imgur.com/lbsWSEz.png)
### itertuples
![](https://i.imgur.com/c0qE4cg.png)
#### itertuples跟iterrows比較
![](https://i.imgur.com/jHRHDrF.png)
![](https://i.imgur.com/v1Cp9Wr.png)

## pandas alternative to looping:apply
![](https://i.imgur.com/BGwl2JE.png)
![](https://i.imgur.com/OBJgtbY.png)
### 比較速度
![](https://i.imgur.com/NBKptjy.png)
![](https://i.imgur.com/62AhNfs.png)

## Optimal pandas iterating

- .value是return np.array=使用起來速度快
![](https://i.imgur.com/GRrvWBX.png)
### vectorization
![](https://i.imgur.com/q82KyHv.png)
- 跑起來速度是μs!! 其他方法都是ms
![](https://i.imgur.com/22qFs4L.png)

