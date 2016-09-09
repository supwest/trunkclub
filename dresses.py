import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
sed -i 's/%//g' filename.csv
sed -i 's/work/Work/g' filename.csv
'''



if __name__ == '__main__':
    dresses = pd.read_csv("./takehome/code/data/product_test.csv")
    #dresses = pd.read_csv('./takehome/data/test_prod.csv')
    cols = ["_".join(x.lower().split()) for x in dresses.columns]
    dresses.columns = cols
    #dresses['percent_dollars_kept'] = dresses.eval('total_dollars_kept/total_dollars_shipped')
    #dresses['percent_items_kept'] = dresses.eval('total_items_kept/total_items_shipped')
    
    
    #percent_cols = [x for x in dresses.columns if 'percent' in x]
    #dresses[percent_cols] = dresses[percent_cols].applymap(lambda x: int(x.replace("%", "")/float(100.)))
    #dresses['percent_dollars_kept'] = dresses.eval('total_dollars_kept/total_dollars_shipped')
    #dresses['percent_items_kept'] = dresses.eval('total_items_kept/total_items_shipped')
    #dresses['precent_great_style'] = dresses.eval('total_great_style/total


    cats = dresses['style'].unique()
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2,2,1,xlabel="Size")
    x_var = 'style'
    x_title = "Style"
    ax1 = sns.boxplot(data=dresses, x=x_var, y='percent_dollars_kept')
    ax1.set_ylabel("Percent Dollars Kept")
    ax1.set_xlabel(x_title)
    ax2 = fig.add_subplot(2,2,3)
    ax2 = sns.boxplot(data=dresses, x=x_var, y='percent_great_style')
    ax2.set_ylabel("Percent Great Style")
    ax3 = fig.add_subplot(2,2,2)
    ax3 = sns.boxplot(data=dresses, x=x_var, y='percent_items_kept')
    ax3.set_ylabel("Percent Items Kept")
    ax4 = fig.add_subplot(2,2,4)
    ax4 = sns.boxplot(data=dresses, x=x_var, y='percent_poor_style')
    ax4.set_ylabel("Percent Poor Style")
    ax2.set_xlabel(x_title)
    ax3.set_xlabel(x_title)
    ax4.set_xlabel(x_title)
    fig.tight_layout()
    fig.savefig('boxplots.png')
    fig.show()
    

    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    #fig2 = plt.hexbin(x=dresses['percent_dollars_kept'], y=dresses['percent_great_style'], cmap=cmap)
    #plt.show()

    cmap='plasma'
    style_groups = dresses.groupby('style').mean()
    brand_groups = dresses.groupby('brand').mean()
    tick_labels = ['Pct. Dollars Kept', 'Pct. Items Kept', 'Pct. Great Style', 'Pct. Poor Style', 'Pct. Poor Fit']
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(1,2,2)
    ax1 = sns.heatmap(style_groups.loc[:, 'percent_dollars_kept':],cmap=cmap,robust=True)
    ax1.set_xlabel('Performance')
    ax1.set_xticklabels(tick_labels, rotation=30, ha='right')
    ax1.set_ylabel('Style')
    ax1.set_yticklabels(style_groups.index.values, rotation='horizontal')
    
    ax2 = fig.add_subplot(1,2,1)
    ax2 = sns.heatmap(brand_groups.loc[:, 'percent_dollars_kept':], cmap=cmap)
    ax2.set_xlabel('Performance')
    ax2.set_xticklabels(tick_labels, rotation=30,ha='right')
    y_tick_labels = brand_groups.index.values
    ax2.set_ylabel('Brand')
    ax2.set_yticklabels(brand_groups.index.values, rotation='horizontal')
    fig.tight_layout()
    fig.savefig('heatmap1.png')
    
    plt.show()
    
    grouped_means = dresses.groupby(['style', 'color']).mean()
    
    grouped_means = grouped_means.loc[:, 'percent_dollars_kept':]
    
    fig2 = plt.figure(figsize=(12,10))
    ax = fig2.add_subplot(1,1,1)
    #ax = sns.clustermap(grouped_means, row_cluster=True, col_cluster=True)
    ax = sns.heatmap(grouped_means,cmap=cmap)
    ax.set(xlabel='Performance')
    ax.set_xticklabels(tick_labels, ha='center')
    y_tick_labels = [' '.join([x[0], x[1]]) for x in grouped_means.index]
    ax.set_ylabel('Style Color Combos')
    ax.set_yticklabels(y_tick_labels, rotation='horizontal')
    fig2.tight_layout()
    fig2.savefig('heatmap2.png')
    plt.show()
    #merged_data = pd.DataFrame(columns=[seaborn_map.data2d.columns])
    #print "cols: {}".format(seaborn_map.data2d.columns)
    #print "index: {}".format(seaborn_map.data2d.index)
    #for major_val in seaborn_map.data2d.index:
    #    print('major_val is {}'.format(major_val))
    #    minor_rows = grouped_means[grouped_means.index==major_val][seaborn_map.data2d.columns]
    #    major_row = grouped_means.loc[major_val,][seaborn_map.data2d.columns]
    #    merged_data.append(major_row).append(minor_rows)
    #merged_map = sns.clustermap(merged_data, row_cluster=False, col_cluster=False)
    #seaborn_map.dendrogram_row.plot(merged_map.ax_row_dendrogram)

    #plt.show()
    grouped_means = dresses.groupby(['brand', 'style']).mean()
    
    grouped_means = grouped_means.loc[:, 'percent_dollars_kept':]
    
    fig3 = plt.figure(figsize=(14,14))
    ax = fig3.add_subplot(1,1,1)
    #ax = sns.clustermap(grouped_means, row_cluster=True, col_cluster=True)
    ax = sns.heatmap(grouped_means,cmap=cmap)
    ax.set(xlabel='Performance')
    ax.set_xticklabels(tick_labels, ha='center')
    y_tick_labels = [' '.join([x[0], x[1]]) for x in grouped_means.index]
    ax.set_ylabel('Style Color Combos')
    ax.set_yticklabels(y_tick_labels, fontsize='x-small',rotation='horizontal')
    fig3.tight_layout()
    fig3.savefig('heatmap3.png')
    plt.show()
