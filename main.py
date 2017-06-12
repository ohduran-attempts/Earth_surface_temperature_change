'''
Get the data from:
https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data

GlobalTemperatures get expanded information from 1800 onwards.

'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')

df_all = pd.read_csv('datasets/GlobalTemperatures.csv')

# Just considering the LandAverageTemperatures...

df = df_all.ix[:, :2]

df.describe()

###

# count = 3180
# mean  =   08.37
# std   =   04.38
# min   =-  02.0
#25%    =   04.31
#50%    =   08.61
#75%    =   12.55
# max   =   19.02

###

def plot_it_1():
    plt.figure(figsize= (15,5))
    plt.plot(df['LandAverageTemperature'])
    plt.title('Average Land Temperature 1750-2017')
    plt.xlabel('Year')
    plt.ylabel('Average Land Temperature')
    plt.show()


# See figure_1. The graph is quite noisy, as it goes up and down with the seasons (added issue: it is noiser on early years, probably because measuring wasn't great

def plot_it_2():
    plt.figure(figsize=(15, 5))
    plt.scatter(x = df['LandAverageTemperature'].index, y = df['LandAverageTemperature'])
    plt.title('Average Land Temperature 1750-2017')
    plt.xlabel('Year')
    plt.ylabel('Average Land Temperature')
    plt.show()

# See figure 2. Scattering helps us see that trend is followed on different parts of the year, so we must filter our data.
# In this case, we are going to average the year to have a solid mean to grasp.

#To do so, we must convert dates into datetime for pandas to better digest.
times = pd.DatetimeIndex(df['dt'])
def plot_it_3(bool = True):
    #Group by year
    grouped = df.groupby([times.year]).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(grouped['LandAverageTemperature'])
    plt.title('Yearly Average Land Temperature 1750-2017')
    plt.xlabel('Year')
    plt.ylabel('Yearly Average Land Temperature')
    if bool:
        plt.show()
    return grouped

# See figure 3. Look at 1752, there must be an Ice Age that suddenly vanished the year after... or maybe someone messed up with the data.

#plot_it_3(False).head()

#1750                8.719364
#1751                7.976143
#1752                5.779833
#1753                8.388083
#1754                8.469333

# Looking at 1752:

#print df[times.year == 1752]

#24  1752-01-01                   0.348
#25  1752-02-01                     NaN
#26  1752-03-01                   5.806
#27  1752-04-01                   8.265
#28  1752-05-01                     NaN
#29  1752-06-01                     NaN
#30  1752-07-01                     NaN
#31  1752-08-01                     NaN
#32  1752-09-01                     NaN
#33  1752-10-01                   7.839
#34  1752-11-01                   7.335
#35  1752-12-01                   5.086

# Data is gone! Summer values are missing and that is so messing up with the data. It is ok, not dataset is perfect.
# We can average what is in between to complete that data. That's not perfect, but feel free to suggest something else.

df['LandAverageTemperature'] = df['LandAverageTemperature'].fillna(method='ffill')

# See figure 4. It is now more reasonably (but still a bit counterintuitive).

## Modelling

# Let's use sklearn.linear_model to come up with a Linear Regression to make sense of this data in the future.

from sklearn.linear_model import LinearRegression as LinReg

grouped = plot_it_3(False)

x = grouped.index.values.reshape(-1,1)
y = grouped['LandAverageTemperature'].values

reg = LinReg()
reg.fit(x,y)

y_preds = reg.predict(x)

# print 'Accuracy: ' + str(reg.score(x,y)) --> 37.67 %

def plot_it_4():
    plt.figure(figsize=(15,5))
    plt.title('Linear Regression')
    plt.scatter(x = x, y = y_preds, c = 'g')
    plt.scatter(x = x, y = y, c = 'r')
    plt.show()

# See figure 4. That means that in 2040, the average temperature will be:

# print reg.predict(2040) --> 9.1