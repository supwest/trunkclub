import pandas as pd





if __name__ == '__main__':
    dresses = pd.read_csv("./trunkclub_data_science_takehome\ 3/data/product_data.csv")
    cols = ["_".join(x.lower().split()) for x in dresses.columns]
    dresses.columns = cols
    dresses['percent_dollars_kept'] = dresses.eval('total_dollars_kept/total_dollars_shipped')
    percent_cols = [x for x in dresses.columns if 'percent' in x]
    dresses[percent_cols] = dresses[percent_cols].applymap(lambda x: int(x.replace("%", "")/float(100.)))
    #dresses['percent_dollars_kept'] = dresses.eval('total_dollars_kept/total_dollars_shipped')
    #dresses['percent_items_kept'] = dresses.eval('total_items_kept/total_items_shipped')
    #dresses['precent_great_style'] = dresses.eval('total_great_style/total

