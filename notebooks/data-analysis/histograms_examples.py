
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path


if 1==1:
    fig = plt.figure()

    # example data
    mu = 100 # mean of distribution
    sigma = 15 # standard deviation of distribution
    x = mu + sigma * np.random.randn(10000)

    num_bins = 50
    # the histogram of the data
    n, bins, patche = plt.hist(x, num_bins, density=True, facecolor='gray', alpha=0.5, histtype ='stepfilled')
    # add a 'best fit' line
    y = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.subplots_adjust(left=0.15)
    plt.show()



if 0==0:
    fig, ax = plt.subplots()
    # Fixing random state for reproducibility
    np.random.seed(19680801)
    # histogram our data with numpy
    data = np.random.randn(1000)
    n, bins = np.histogram(data, 50)
    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.show()


if 1==1:
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(19680801)

    n_bins = 10
    x = np.random.randn(1000, 3)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()

    colors = ['red', 'tan', 'lime']
    ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('bars with legend')

    ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True)
    ax1.set_title('stacked bar')

    ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
    ax2.set_title('stack step (unfilled)')

    # Make a multiple-histogram of data-sets with different length.
    x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
    ax3.hist(x_multi, n_bins, histtype='bar')
    ax3.set_title('different sample sizes')

    fig.tight_layout()
    plt.show()



# Histograms
if 1==1:
    n_bins = 10
    x = np.random.randn(1000, 3)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flatten()

    colors = ['gray', 'slategray', 'darkgray']
    ax0.hist(x, n_bins, density=True, histtype='bar', color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('bars with legend')

    ax1.hist(x, n_bins, density=True, histtype='bar', stacked=True, color=colors)
    ax1.set_title('stacked bar')

    ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False, color=colors)
    ax2.set_title('stack step (unfilled)')

    # Make a multiple-histogram of data-sets with different length.
    x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
    ax3.hist(x_multi, n_bins, histtype='bar', color=colors)
    ax3.set_title('different sample sizes')

    fig.tight_layout()
    plt.show()