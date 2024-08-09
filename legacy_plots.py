

""" ==================== Others ========================
"""
if 0 == 1:

    """
    """
    """ Plot scatterplots with matplotlib (adding ouutliers) NO
    """
    """
    """
    if 0 == 1:

        def plot_scatterplots_plt_4(df2, df3, variable_x, variable_y, name, option_outliers):
            """
            :param df2:
            :param df3:
            :param variable_x:
            :param variable_y:
            :param name:
            :return:

            """
            fontsize_ = 12
            marker_size_ = 10

            fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

            outliers_plot = option_outliers

            ax1 = fig.add_subplot(2, 2, 1)
            ax1.scatter(df2[variable_x[0]],
                        df2[variable_y[0]], marker='o', s=marker_size_, alpha=0.8)
            if outliers_plot:
                ax1.scatter(df3[variable_x[0]],
                            df3[variable_y[0]], marker='o', s=marker_size_, alpha=0.2, color='red',
                            label='w/o outliers')
            ax1.set_xlabel(variable_x[0] + ' ' + units_[variable_x[0]], fontsize=fontsize_)
            ax1.set_ylabel(variable_y[0] + ' ' + units_[variable_y[0]], fontsize=fontsize_)
            # y_min = df2[variable_y[0]].min()
            # y_max = df2[variable_y[0]].max()
            # x_min = df2[variable_x[0]].min()
            # x_max = df2[variable_x[0]].max()
            # y_range = np.arange(y_min, y_max, round(y_max/5))
            # x_range = np.arange(x_min, x_max, round(x_max / 5))
            # ax1.set_yticks(y_range)
            # ax1.set_yticklabels(y_range)
            # ax1.set_xticks(x_range)
            # ax1.set_xticklabels(x_range)
            ax1.tick_params(axis='both', labelsize=fontsize_)
            ax1.legend(fontsize=fontsize_)

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.scatter(df2[variable_x[1]],
                        df2[variable_y[1]], marker='o', s=marker_size_, alpha=0.8)
            if outliers_plot:
                ax2.scatter(df3[variable_x[1]],
                            df3[variable_y[1]], marker='o', s=marker_size_, alpha=0.2, color='red',
                            label='w/o outliers')
            ax2.set_xlabel(variable_x[1] + ' ' + units_[variable_x[1]], fontsize=fontsize_)
            ax2.set_ylabel(variable_y[1] + ' ' + units_[variable_y[1]], fontsize=fontsize_)
            ax2.tick_params(axis='both', labelsize=fontsize_)
            ax2.legend(fontsize=fontsize_)

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.scatter(df2[variable_x[2]],
                        df2[variable_y[2]], marker='o', s=marker_size_, alpha=0.8)
            if outliers_plot:
                ax3.scatter(df3[variable_x[2]],
                            df3[variable_y[2]], marker='o', s=marker_size_, alpha=0.2, color='red',
                            label='w/o outliers')
            ax3.set_xlabel(variable_x[2] + ' ' + units_[variable_x[2]], fontsize=fontsize_)
            ax3.set_ylabel(variable_y[2] + ' ' + units_[variable_y[2]], fontsize=fontsize_)
            ax3.tick_params(axis='both', labelsize=fontsize_)
            ax3.legend(fontsize=fontsize_)

            ax4 = fig.add_subplot(2, 2, 4)
            ax4.scatter(df2[variable_x[3]],
                        df2[variable_y[3]], marker='o', s=marker_size_, alpha=0.8)
            if outliers_plot:
                ax4.scatter(df3[variable_x[3]],
                            df3[variable_y[3]], marker='o', s=marker_size_, alpha=0.2, color='red',
                            label='w/o outliers')
            ax4.set_xlabel(variable_x[3] + ' ' + units_[variable_x[3]], fontsize=fontsize_)
            ax4.set_ylabel(variable_y[3] + ' ' + units_[variable_y[3]], fontsize=fontsize_)
            ax4.tick_params(axis='both', labelsize=fontsize_)
            ax4.legend(fontsize=fontsize_)

            plt.subplots_adjust(wspace=0.2, hspace=0.4)
            plt.show()
            plt.savefig('{}.png'.format(name), bbox_inches='tight', pad_inches=0)
            # plt.savefig('{}.pdf'.format(name), bbox_inches='tight', pad_inches=0)
            plt.close()


        variable_x = ['occupancy', 'flow', 'headway', 'meanlength']
        variable_y = ['flow', 'headway', 'speed', 'flow']
        name = 'scatterplot'
        plot_scatterplots_plt_4(df2, df3, variable_x, variable_y, name, option_outliers=True)

        variable_x = ['occupancy', 'volume', 'headway', 'volume']
        variable_y = ['speed', 'speed', 'meanlength', 'meanlength']
        name = 'scatterplot_2'
        plot_scatterplots_plt_4(df2, df3, variable_x, variable_y, name, option_outliers=True)

    """ Plot correlation matrix (long)
    """
    if 0 == 1:
        R = df.loc[:, columns_].corr()
        r = round(R, 2)
        fig = go.Figure(data=go.Heatmap(
            z=r,
            x=columns_,
            y=columns_,
            hoverongaps=False,
            colorscale='Viridis'))
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Legacy Way - Correlation Matrix - Pearson correlation coefficient')
        fig.update_layout(showlegend=True)
        fig.show()

    if 1 == 0:
        df.loc['2018-10-28':'2018-11-05', columns_].plot(subplots=True, layout=(7, 1))

    """ Plot together
    """
    if 1 == 1:

        fig = plt.figure(figsize=(30, 10), dpi=150, facecolor='w', edgecolor='w')

        index = {'flow': 1,
                 'volume': 2,
                 'speed': 3,
                 'meanlength': 4,
                 'gap': 5,
                 'headway': 6,
                 'occupancy': 7,
                 }

        start_date = '2019-03-15'
        end_date = '2019-03-18'
        fontsize_ = 12
        t = df.loc[start_date:end_date].index
        for variable in columns_:
            ax = fig.add_subplot(3, 3, index[variable])
            y = df.loc[start_date:end_date, [variable]]
            ax.plot(t, y, label=variable + units_[variable])
            ax.set_xlabel('time (15 min)', fontsize=fontsize_)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.tick_params(axis='both', labelsize=fontsize_)
            ax.legend(fontsize=fontsize_)
            ax.set_title('Legacy way - Time Series - {} to {}'.format(start_date, end_date))

        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.show()
        name = 'timeseries_together'
        plt.savefig('{}.png'.format(name), bbox_inches='tight', pad_inches=0)
        plt.close()

    if 0 == 1:
        variable_y = 'speed'
        variable_x = 'occupancy'
        fig = go.Figure()
        trace1 = fig.add_scatter(x=df[variable_x],
                                 y=df[variable_y],
                                 mode='markers',
                                 name='points',
                                 marker=dict(size=10,
                                             opacity=.1,
                                             color='white',
                                             line=dict(width=1, color='#1f77b4')
                                             )
                                 )
        trace2 = fig.add_histogram(x=df[variable_x],
                                   name='x density',
                                   marker=dict(color='#1f77b4', opacity=0.7),
                                   yaxis='y2'
                                   )
        trace3 = fig.add_histogram(y=df[variable_y],
                                   name='y density',
                                   marker=dict(color='#1f77b4', opacity=0.7),
                                   xaxis='x2'
                                   )
        fig.layout = dict(xaxis=dict(domain=[0, 0.85],
                                     showgrid=False,
                                     zeroline=False),
                          yaxis=dict(domain=[0, 0.85],
                                     showgrid=False,
                                     zeroline=False),
                          showlegend=False,
                          margin=dict(t=50),
                          hovermode='closest',
                          bargap=0,
                          xaxis2=dict(domain=[0.85, 1],
                                      showgrid=False,
                                      zeroline=False),
                          yaxis2=dict(domain=[0.85, 1],
                                      showgrid=False,
                                      zeroline=False),
                          height=600,
                          )
        fig.show()

    if 0 == 1:
        variable_y = 'speed'
        variable_x = 'occupancy'
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df[variable_y],
                                 y=df[variable_x],
                                 mode='markers',
                                 name='Occupacy-flow',
                                 marker=dict(color='blue')))

        fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
        fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])
        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Legacy Way')
        fig.update_layout(showlegend=False)
        fig.show()

    """ Plot scatterplots (together)
    """
    if 0 == 0:
        def scatterplot_four_plotly(df, variables_x, variables_y):
            fig = make_subplots(rows=2, cols=2)

            fig.add_trace(go.Scatter(x=df['occupancy'],
                                     y=df['flow'],
                                     mode='markers',
                                     name='Occupacy-flow',
                                     marker=dict(color='blue')), row=1, col=1)

            fig.add_trace(go.Scatter(x=df['speed'],
                                     y=df['flow'],
                                     mode='markers',
                                     name='speed-flow',
                                     marker=dict(color='blue')), row=2, col=1)

            fig.add_trace(go.Scatter(x=df['headway'],
                                     y=df['flow'],
                                     mode='markers',
                                     name='headway-flow',
                                     marker=dict(color='blue')), row=1, col=2)

            fig.add_trace(go.Scatter(x=df['meanlength'],
                                     y=df['flow'],
                                     mode='markers',
                                     name='meanlength-flow',
                                     marker=dict(color='blue')), row=2, col=2)

            fig.update_xaxes(title_text=variables_x[0] + ' ' + units_[variables_x[0]], row=1, col=1)
            fig.update_yaxes(title_text=variables_y[0] + ' ' + units_[variables_y[0]], row=1, col=1)

            fig.update_xaxes(title_text=variables_x[1] + ' ' + units_[variables_x[1]], row=2, col=1)
            fig.update_yaxes(title_text=variables_y[1] + ' ' + units_[variables_y[1]], row=2, col=1)

            fig.update_xaxes(title_text=variables_x[2] + ' ' + units_[variables_x[2]], row=1, col=2)
            fig.update_yaxes(title_text=variables_y[2] + ' ' + units_[variables_y[2]], row=1, col=2)

            fig.update_xaxes(title_text=variables_x[3] + ' ' + units_[variables_x[3]], row=2, col=2)
            fig.update_yaxes(title_text=variables_y[3] + ' ' + units_[variables_y[3]], row=2, col=2)

            fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
            fig.update_layout(title='Legacy Way')
            fig.update_layout(showlegend=False)
            fig.show()

    if 1 == 1:
        variable_y = 'flow'
        variable_x = 'speed'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[variable_x],
                                 y=sd[variable_y],
                                 mode='markers',
                                 name='Occupacy-flow',
                                 marker=dict(color='blue')))
        fig.update_xaxes(title_text=variable_x + ' ' + units_[variable_x])
        fig.update_yaxes(title_text=variable_y + ' ' + units_[variable_y])

        fig.update_layout(font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"))
        fig.update_layout(title='Legacy Way')
        fig.update_layout(showlegend=False)
        fig.show()





 def plot_scatterplots_with_histogram(df, variable_x, variable_y, group_var, road):
        """
        Scatterplot matplotlib (4 plots)

        :param df:
        :param variable_x:
        :param variable_y:
        :param group_var:
        :param title:
        :param road:
        :return:
        """
        fontsize_ = 12
        marker_size_ = 10

        fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='w')

        y_min = df[variable_y].min()
        y_max = df[variable_y].max()
        x_min = df[variable_x].min()
        x_max = df[variable_x].max()
        y_range = list(np.arange(y_min, y_max, round(y_max / 5)))
        x_range = list(np.arange(x_min, x_max, round(x_max / 5)))

        binwidth = 0.8
        x = df[variable_x]
        y = df[variable_y]
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(0, lim + binwidth, binwidth)

        # scatterplot
        ax1 = fig.add_subplot(2, 2, 3)
        ax1.scatter(x, y, marker='o', s=marker_size_, alpha=0.8, c=df[group_var])
        # cbar = plt.colorbar(sc, orientation="horizontal", pad=0.2)
        # cbar.set_label(group_var, rotation=270, fontsize=fontsize_)
        ax1.set_xlabel(variable_x + ' ' + units_[variable_x], fontsize=fontsize_)
        ax1.set_ylabel(variable_y + ' ' + units_[variable_y], fontsize=fontsize_)

        ax1.set_yticks(y_range)
        ax1.set_yticklabels(y_range)
        ax1.set_xticks(x_range)
        ax1.set_xticklabels(x_range)
        ax1.tick_params(axis='both', labelsize=fontsize_)
        ax1.set_xlabel(variable_x + '' + units_[variable_x], fontsize=fontsize_)
        ax1.set_ylabel(variable_y + '' + units_[variable_y], fontsize=fontsize_)

        # histogram variable x
        ax3 = fig.add_subplot(2, 2, 1)
        ax3.hist(x, bins=bins, orientation='vertical', range=(x_min, x_max))
        ax3.set_yticks(x_range)
        ax3.set_yticklabels(x_range)
        ax3.tick_params(axis='both', labelsize=fontsize_)
        # ax3.set_ylabel(variable_x + '' + units_[variable_x], fontsize=fontsize_)
        ax3.axis('off')

        # histogram variable y
        ax2 = fig.add_subplot(2, 2, 4)
        ax2.hist(y, bins=bins, orientation='horizontal', range=(y_min, y_max))
        ax2.set_yticks(x_range)
        ax2.set_yticklabels(x_range)
        ax2.tick_params(axis='both', labelsize=fontsize_)
        # ax2.set_ylabel(variable_y + '' + units_[variable_y], fontsize=fontsize_)
        ax2.axis('off')

        now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")

        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        plt.show()
        plt.savefig(
            'figures\\{}_scatterplot_with_histogram_{}_{}_{}.png'.format(road, variable_x, variable_y, group_var),
            bbox_inches='tight', pad_inches=0)
        plt.close()


    if 1 == 0:
        df_aux = filter_by_std(df, 'speed', 2)
        plot_scatterplots_with_histogram(df_aux, 'speed', 'volume', 'hour', road)