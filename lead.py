import pandas as pd
import numpy as np
import re, ast
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, roc_auc_score, recall_score

class BN(object):

    def __init__(self, converts, non_converts):
        self.converts = converts
        self.non_converts = non_converts
        self.p_c = np.true_divide(converts.shape[0], non_converts.shape[0])
        self.referral_cols = referral_cols
        self.evening_cols = evening_cols
        self.work_cols = work_cols


def bayes_predict(converts, non_converts, obs):
    n_converts = converts.shape[0]
    n_non_converts = non_converts.shape[0]

    prob_c = n_converts/(n_converts + n_non_converts)
    prob_nc = 1 - prob_c

    #
    #print n_converts, n_non_converts
    obs_referral = obs[referral_cols]
    obs_evening = obs[evening_cols]
    obs_work = obs[work_cols]

    #print obs_referral, obs_work, obs_evening

    prob_referral_given_c = converts[referral_cols].sum(axis=0)/n_converts
    prob_evening_given_c = converts[evening_cols].sum(axis=0)/n_converts
    prob_work_given_c = converts[work_cols].sum(axis=0)/n_converts
    
    #print prob_referral_given_c.sum()

    prob_referral_given_nc = non_converts[referral_cols].sum(axis=0)/n_non_converts
    prob_evening_given_nc = non_converts[evening_cols].sum(axis=0)/n_non_converts
    prob_work_given_nc = non_converts[work_cols].sum(axis=0)/n_non_converts

    #print prob_referral_given_nc

    prob_referral = prob_referral_given_c*prob_c + prob_referral_given_nc*prob_nc
    prob_evening = prob_evening_given_c*prob_c + prob_evening_given_nc*prob_nc
    prob_work = prob_work_given_c*prob_c + prob_work_given_nc*prob_nc

    #print prob_referral.sum()

    #print prob_referral_given_c
    #print obs_referral
    #print "prod", prob_referral_given_c[obs_referral==1].values*prob_evening_given_c[obs_evening==1].values*prob_work_given_c[obs_work==1].values

    post_numerator = prob_referral_given_c[obs_referral==1].values*prob_evening_given_c[obs_evening==1].values*prob_work_given_c[obs_work==1].values*np.true_divide(n_converts, (n_converts+n_non_converts))

    #print prob_referral[obs_referral==1]
    #print prob_evening[obs_evening==1]
    #print prob_work[obs_work==1]

    #print "probwork", prob_work
    post_denominator = (prob_referral[obs_referral==1].values*prob_evening[obs_evening==1].values*prob_work[obs_work==1].values)

    #print post_denominator

    post = post_numerator/post_denominator
    #print "post", post

    #if post > 0.5:
    #    return 1
    #else:
    #    return 0
    return post[0]

def parse_stores(s):
    ret = []
    for x in s:
        if re.search(r'\bnan', x):
            x = re.sub(r'\bnan', "['']", x)
        ret.append(ast.literal_eval(x))
    return ret


#def fill_ref(df):
#    df['referral'].apply(lambda x: 'Facebook Ad' if x is None)



if __name__ == '__main__':
    df = pd.read_csv('./data/lead_data.csv')
    df = df.drop(['Unnamed: 0', 'income_zipcode'], axis=1)
    stores = df['preferred_stores']
    stores = stores.fillna("['']")

    df = df.interpolate()
    df['referral'] = df['referral'].fillna('Facebook Ad')
    df = df.dropna()
    df = df.drop('preferred_stores', axis=1)
    df = pd.get_dummies(df)

    train, test = train_test_split(df, test_size=.2)

    y_train = train.pop('converted').values
    X_train = train.values

    y_test = test.pop('converted').values
    X_test = test.values

    rf = RandomForestClassifier()
    model1 = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    #a1 = accuracy_score(y_test, y_pred)
    p1 = precision_score(y_test, y_pred)    #0.76
    r1 = recall_score(y_test, y_pred)       #0.315
    auc1 = roc_auc_score(y_test, y_pred)    #0.655

    gb = GradientBoostingClassifier()
    model2 = gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    p1 = precision_score(y_test, y_pred)    #0.77
    r1 = recall_score(y_test, y_pred)       #0.375
    auc1 = roc_auc_score(y_test, y_pred)    #0.68
    

    converts = df[df['converted']==1]
    non_converts = df[df['converted']==0]

    non_cat_cols = ['lead_id', 'height', 'weight', 'waist', 'shoe_size', 'age']
    converts = converts.drop(non_cat_cols, axis=1)
    non_converts = non_converts.drop(non_cat_cols, axis=1)

    n_converts = converts.shape[0]
    n_non_converts = non_converts.shape[0]

    referral_cols = [x for x in non_converts.columns if 'referral' in x]
    evening_cols = [x for x in non_converts.columns if 'evening' in x]
    work_cols = [x for x in converts.columns if 'work' in x]

    convert_referral_counts = converts[referral_cols].sum(axis=0)
    convert_work_counts = converts[work_cols].sum(axis=0)
    contert_evening_counts = converts[work_cols].sum(axis=0)

    non_convert_referral_counts = non_converts[referral_cols].sum(axis=0)
    non_convert_work_counts = non_converts[work_cols].sum(axis=0)
    non_convert_evening_counts = non_converts[evening_cols].sum(axis=0)


    obs = converts.iloc[1]

    y_pred=[]
    for i in xrange(converts.shape[0]):
        obs = converts.iloc[i]
        y_pred.append(bayes_predict(converts, non_converts, obs))

    #print y_pred
