import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})
rc('text', usetex=True)
import pandas as pd
import numpy as np


from pandas.plotting import scatter_matrix
from matplotlib.ticker import NullFormatter


data = pd.read_csv("data.csv", header=None)
data2 = pd.concat([data, data[0:232]])
data3 = pd.concat([data2, data2.iloc[:,0:3]],axis=1)
data4 =  pd.read_csv("data2.csv")

names = ['X1a','X1b','X1c','X2','X3','X4','X5','X6','X7','X8','X9','X10']
df = pd.DataFrame(data3.iloc[:,:].values,columns={'X1a','X1b','X1c','X2','X3','X4','X5','X6','X7','X8','X9','X10'})
df = df.reindex(columns=names)






#############
# BOXPLOTS
#############
# Another useful way to review the distribution of each attribute is to use Box and Whisker Plots or for short.
# Boxplots summarize the distribution of each attribute, drawing a line for the median (middle value) and a box around the 25th and 75th percentiles (the middle 50% of the data). The whiskers give an idea of the spread of the data and dots outside of the whiskers show candidate outlier values (values that are 1.5 times greater than the size of spread of the middle 50% of the data).
if 0==1:
    df.plot(kind='box', subplots=True, figsize=(20,4), sharex=False, sharey=False)
    plt.show()
    #plt.savefig('1-boxplots.eps')
    plt.savefig('1-boxplots.pdf')
    plt.close()


#################
# DENSITY PLOTS
#################
# Density plots are another way of getting a quick idea of the distribution of each attribute. The plots look like an abstracted histogram with a smooth curve drawn through the top of each bin, much like your eye tried to do with the histograms.
if 0==1:
    df.plot(kind='density', subplots=True, figsize=(12,8), sharex=False, color='black')
    plt.show()
    #plt.savefig('2-density_plots.eps')
    plt.savefig('2-density_plots.pdf')
    plt.close()



#################
# HISTOGRAMS
#################
#A fast way to get an idea of the distribution of each attribute is to look at histograms.
#Histograms group data into bins and provide you a count of the number of observations in each bin. From the shape of the bins you can quickly get a feeling for whether an attribute is Gaussian’, skewed or even has an exponential distribution. It can also help you see possible outliers.
#We can see that perhaps the attributes age, pedi and test may have an exponential distribution. We can also see that perhaps the mass and pres and plas attributes may have a Gaussian or nearly Gaussian distribution. This is interesting because many machine learning techniques assume a Gaussian univariate distribution on the input variables.
if 0==1:
    df.hist(figsize=(12,8), color='gray')
    plt.show()
    #plt.savefig('histogram_plots.eps')
    plt.savefig('3-histogram_plots.pdf')
    plt.close()






##########################
# CORRELATION MATRIX PLOT
##########################
#Correlation gives an indication of how related the changes are between two variables. If two variables change in the same direction they are positively correlated. If the change in opposite directions together (one goes up, one goes down), then they are negatively correlated.
#You can calculate the correlation between each pair of attributes. This is called a correlation matrix. You can then plot the correlation matrix and get an idea of which variables have a high correlation with each other.
#This is useful to know, because some machine learning algorithms like linear and logistic regression can have poor performance if there are highly correlated input variables in your data
# We can see that the matrix is symmetrical, i.e. the bottom left of the matrix is the same as the top right. This is useful as we can see two different views on the same data in one plot. We can also see that each variable is perfectly positively correlated with each other (as you would expected) in the diagonal line from top left to bottom right
if 0==1:
    correlations = df.corr()
    # plot correlation matrix
    fig2 = plt.figure(num=1, figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
    ax = fig2.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    fig2.colorbar(cax)
    ticks = np.arange(0,12,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.show()
    #plt.savefig('4-scatter_matrix_correlation.eps')
    plt.savefig('4-scatter_matrix_correlation.pdf')
    plt.close()


##########################
# CORRELATION MATRIX PLOT (2)
##########################
if 0==1:

    corr = data4.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(data4.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    #ax.set_xticklabels(data.columns)
    #ax.set_yticklabels(data.columns)
    plt.show()
    plt.savefig('5-scatter_matrix_correlation.pdf')
    #plt.savefig('5-scatter_matrix_correlation2.eps')
    plt.close()


##########################
# SCATTERPLOT MATRIX
##########################
#A scatterplot shows the relationship between two variables as dots in two dimensions, one axis for each attribute. You can create a scatterplot for each pair of attributes in your data. Drawing all these scatterplots together is called a scatterplot matrix.
#Scatter plots are useful for spotting structured relationships between variables, like whether you could summarize the relationship between two variables with a line. Attributes with structured relationships may also be correlated and good candidates for removal from your dataset.
# Like the Correlation Matrix Plot, the scatterplot matrix is symmetrical. This is useful to look at the pair-wise relationships from different perspectives. Because there is little point oi drawing a scatterplot of each variable with itself, the diagonal shows histograms of each attribute
if 0==1:
    fig = plt.figure(num=1, figsize=(8,8), dpi=80, facecolor='w', edgecolor='k')
    scatter_matrix(df)
    plt.savefig('6-scatter_matrix.pdf')
    plt.show()
    plt.close()



##########################
# SCATTERHIST PLOT
##########################
if 0==1:
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # the random data
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    #x = df.loc[:,['X2']].values
    #y = df.loc[:,['X4']].values
    axScatter.scatter(x,y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))
    axScatter.set_xlabel('$X_3$')
    axScatter.set_ylabel('$X_6$')

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.show()
    plt.savefig('7-scatterhist.pdf')
    plt.close()