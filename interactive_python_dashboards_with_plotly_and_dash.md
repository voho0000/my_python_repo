# interactive python dashboards with plotly and dash

[TOC]

還沒補完，但短期內應該不會有互動式視覺化的需求，短期內不會更新

實作改放在自己的jupyter notebook

Resource:
1. [guidebook](https://docs.google.com/document/d/1DjWL2DxLiRaBrlD3ELyQlCBRu7UQuuWfgjv9LncNp_M/edit#heading=h.17dp8vu)
2. [github](https://github.com/Pierian-Data/Plotly-Dashboards-with-Dash)
3. [Dash App Gallery](https://dash-gallery.plotly.host/Portal/)


## 製造數值方式
### random

```python
import numpy as np

np.random.seed(42)
random_x = np.random.randint(1,101,100)
random_y = np.random.randint(1,101,100)
```
### linspace
```python
import numpy as np

np.random.seed(56)
x_values = np.linspace(0, 1, 100) # 100 evenly spaced values
y_values = np.random.randn(100)   # 100 random values
```
### scatter plot
- data是list []
```python
#######
# This plots 100 random data points (set the seed to 42 to
# obtain the same points we do!) between 1 and 100 in both
# vertical and horizontal directions.
######
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np

np.random.seed(42)
random_x = np.random.randint(1,101,100)
random_y = np.random.randint(1,101,100)

data = [go.Scatter(
    x = random_x,
    y = random_y,
    mode = 'markers',
    marker = dict(      # change the marker style
        size = 12,
        color = 'rgb(51,204,153)',
        symbol = 'pentagon',
        line = dict(
            width = 2,
        )
    )
)]
layout = go.Layout(
    title = 'Random Data Scatterplot', # Graph title
    xaxis = dict(title = 'Some random x-values'), # x-axis label
    yaxis = dict(title = 'Some random y-values'), # y-axis label
    hovermode ='closest' # handles multiple points landing on the same vertical
)
fig = go.Figure(data=data, layout=layout)
pyo.plot(fig, filename='scatter3.html')

```
## plot
### line plot
```python
#######
# This line chart displays the same data
# three different ways along the y-axis.
######
import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np

np.random.seed(56)
x_values = np.linspace(0, 1, 100) # 100 evenly spaced values
y_values = np.random.randn(100)   # 100 random values

# create traces
trace0 = go.Scatter(
    x = x_values,
    y = y_values+5,
    mode = 'markers',
    name = 'markers'
)
trace1 = go.Scatter(
    x = x_values,
    y = y_values,
    mode = 'lines+markers',
    name = 'lines+markers'
)
trace2 = go.Scatter(
    x = x_values,
    y = y_values-5,
    mode = 'lines',
    name = 'lines'
)
data = [trace0, trace1, trace2]  # assign traces to data
layout = go.Layout(
    title = 'Line chart showing three different modes'
)
fig = go.Figure(data=data,layout=layout)
pyo.plot(fig, filename='line1.html')
```

```python
```

# DASH
- 簡介Dash
![](https://i.imgur.com/UAmIUHm.png)
![](https://i.imgur.com/lTDEquh.png)
![](https://i.imgur.com/Mxgw2DD.png)
![](https://i.imgur.com/F6ElxaW.png)
![](https://i.imgur.com/UJiiyQv.png)
- 步驟
![](https://i.imgur.com/LiNp1BF.png)

## layout
### code
- Div:
    - A Div component. Div is a wrapper for the div HTML5 element.
    -  [DOC](https://dash.plot.ly/dash-html-components/div)
    -  children (a list of or a singular dash component, string or number; optional): The children of this component

- 最初階案例，未加上plotly圖
```python
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.Div(children='Dash: A web application framework for Python.'),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server()

```
![](https://i.imgur.com/ENyon89.png)
- 改顏色跟排版
```python
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(
        children='Dash: A web application framework for Python.',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                },
                'title': 'Dash Data Visualization'
            }
        }
    )],
    style={'backgroundColor': colors['background']}
)

if __name__ == '__main__':
    app.run_server()

```
![](https://i.imgur.com/5ASvfFf.png)

- 加上plotly圖

```python
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

app = dash.Dash()

# Creating DATA
np.random.seed(42)
random_x = np.random.randint(1,101,100)
random_y = np.random.randint(1,101,100)

app.layout = html.Div([dcc.Graph(id='scatterplot',
                                figure = {'data':[
                                        go.Scatter(
                                        x=random_x,
                                        y=random_y,
                                        mode='markers',
                                        marker = {
                                            'size':12,
                                            'color': 'rgb(51,204,153)',
                                            'symbol':'pentagon',
                                            'line':{'width':2}
                                            }
                                            )],
                                        'layout':go.Layout(title='My Scatterplot',
                                                            xaxis = {'title':'Some X title'})}
                                                           ),
                    dcc.Graph(id='scatterplot2',
                                        figure = {'data':[
                                                go.Scatter(
                                                x=random_x,
                                                y=random_y,
                                                mode='markers',
                                                marker = {
                                                    'size':12,
                                                    'color': 'rgb(200,204,53)',
                                                    'symbol':'pentagon',
                                                    'line':{'width':2}
                                                }
                                                )],
                                        'layout':go.Layout(title='Second Plot',
                                                            xaxis = {'title':'Some X title'})}
                                        )])

if __name__ == '__main__':
    app.run_server()

```

![](https://i.imgur.com/FIr7guk.png)

- Markdown
```python
import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/) specification of Markdown.

Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!

Markdown includes syntax for things like **bold text** and *italics*,
[links](http://commonmark.org/help), inline `code` snippets, lists,
quotes, and more.
'''

app.layout = html.Div([
    dcc.Markdown(children=markdown_text)
])

if __name__ == '__main__':
    app.run_server()
```
![](https://i.imgur.com/22NKsKX.png)

- 使用df

```python
#######
# Objective: build a dashboard that imports OldFaithful.csv
# from the data directory, and displays a scatterplot.
# The field names are:
# 'D' = date of recordings in month (in August),
# 'X' = duration of the current eruption in minutes (to nearest 0.1 minute),
# 'Y' = waiting time until the next eruption in minutes (to nearest minute).
######

# Perform imports here:
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

# Launch the application:
app = dash.Dash()

# Create a DataFrame from the .csv file:
df = pd.read_csv('../data/OldFaithful.csv')

# Create a Dash layout that contains a Graph component:
app.layout = html.Div([
    dcc.Graph(
        id='old_faithful',
        figure={
            'data': [
                go.Scatter(
                    x = df['X'],
                    y = df['Y'],
                    mode = 'markers'
                )
            ],
            'layout': go.Layout(
                title = 'Old Faithful Eruption Intervals v Durations',
                xaxis = {'title': 'Duration of eruption (minutes)'},
                yaxis = {'title': 'Interval to next eruption (minutes)'},
                hovermode='closest'
            )
        }
    )
])

# Add the server clause:
if __name__ == '__main__':
    app.run_server()
```
![](https://i.imgur.com/3zjj0ly.png)

## component
![](https://i.imgur.com/r3i6Y1B.png)
![](https://i.imgur.com/bcseae2.png)
![](https://i.imgur.com/vA2WRgs.png)

### html component
![](https://i.imgur.com/n3oU6va.png)
![](https://i.imgur.com/My8bDCq.png)
![](https://i.imgur.com/WWWDKKD.png)
![](https://i.imgur.com/0QWGa1N.png)

```python
#######
# This provides examples of Dash HTML Components.
# Feel free to add things to it that you find useful.
######
import dash
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div([
    'This is the outermost Div',
    html.Div(
        'This is an inner Div',
        style={'color':'blue', 'border':'2px blue solid', 'borderRadius':5,
        'padding':10, 'width':220}
    ),
    html.Div(
        'This is another inner Div',
        style={'color':'green', 'border':'2px green solid',
        'margin':10, 'width':220}
    ),
],
# this styles the outermost Div:
style={'width':500, 'height':200, 'color':'red', 'border':'2px red dotted'})

if __name__ == '__main__':
    app.run_server()
```
![](https://i.imgur.com/FHu4Ipm.png)


```python
```


```python
```
