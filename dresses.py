import pandas as pd





if __name__ == '__main__':
    # dresses = pd.read_csv("./trunkclub_data_science_takehome\ 3/data/product_data.csv")
    dresses = pd.read_csv('./takehome/data/test_prod.csv')
    cols = ["_".join(x.lower().split()) for x in dresses.columns]
    dresses.columns = cols
    #dresses['percent_dollars_kept'] = dresses.eval('total_dollars_kept/total_dollars_shipped')
    #dresses['percent_items_kept'] = dresses.eval('total_items_kept/total_items_shipped')
    
    
    #percent_cols = [x for x in dresses.columns if 'percent' in x]
    #dresses[percent_cols] = dresses[percent_cols].applymap(lambda x: int(x.replace("%", "")/float(100.)))
    #dresses['percent_dollars_kept'] = dresses.eval('total_dollars_kept/total_dollars_shipped')
    #dresses['percent_items_kept'] = dresses.eval('total_items_kept/total_items_shipped')
    #dresses['precent_great_style'] = dresses.eval('total_great_style/total


    cats = dresses['size'].unique()
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)

    ax1 = sns.boxplot(data=dresses, x='size', y='percent_dollars_kept')
    ax1.set_ylabel("Percent Dollars Kept")
    ax2 = fig.add_subplot(2,2,2)
    ax2 = sns.boxplot(data=dresses, x='size', y='percent_great_style')
    ax2.set_ylabel("Percent Great Style")
    fig.show()
    

    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    fig2 = plt.hexbin(x=dressed['percent_dollars_kept'], y=dresses['percent_great_style'], cmap=cmap)
    plt.show()

