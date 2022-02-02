import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, plot_tree
from sklearn import tree
from sklearn import metrics
from matplotlib import pyplot as plt
import plotly.express as px

def combo_df():
    combo = pd.read_csv('combo.csv')

    # Clean Variables to reduce categories    
    combo['if_hired'] = np.where(combo['hired'] == 'True', 'True', 'False')
    combo['job'] = np.where(combo['category'] == 'House Cleaning', 'House Cleaning', 'Local Moving (under 50 miles)')
    combo['pro_page_viewed'] = np.where(combo['service_page_viewed'] == 'True', 'True', 'False')
    combo['av_rating'] = np.where(combo['avg_rating']< 4.6, 'Low rating', 'High Rating')
    combo['cost'] = np.where(combo['cost_estimate'] < 93.0, 'Low cost', 'High cost')
    combo['table_position'] = np.where(combo['result_position'] > 12, 'Lower on search table', 'Higher on search table')
    combo.reset_index(drop=True, inplace = True)

    # Create Dummy Variables
    combo = pd.concat([combo, pd.get_dummies(combo['job'], prefix='Job')], axis=1)
    combo = pd.concat([combo, pd.get_dummies(combo['pro_page_viewed'], prefix='Pro_page_viewed')], axis=1)
    combo = pd.concat([combo, pd.get_dummies(combo['av_rating'], prefix='Avg_Rating')], axis=1)
    combo = pd.concat([combo, pd.get_dummies(combo['cost'], prefix='Cost')], axis=1)
    combo = pd.concat([combo, pd.get_dummies(combo['table_position'], prefix='Table_position')], axis=1)
    return combo

def create_model(x_vals, depth=2):
    combo = combo_df()
    x = combo[x_vals]
    y = combo['if_hired']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.30)

    dtree =  DecisionTreeClassifier(max_depth=depth)
    dtree.fit(x_train,y_train)

    y_pred = dtree.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return (dtree, x.columns, accuracy)

def visualize_tree(model, cols):
    if model.max_depth <= 2:
        fig = plt.figure(figsize=(3,1))
    else:
        fig = plt.figure(figsize =(4,2))
    tree.plot_tree(model, feature_names = cols, class_names = ['Hired', 'False'], filled = True, proportion = True, rounded = True)

def find_importance(model, col_names):
    importance = pd.Series(model.feature_importances_, index = col_names).sort_values(ascending=False).to_frame()
    importance.reset_index(inplace=True)
    importance.columns = ['Factor', 'Importance Level']
    return importance

    
def get_col_names(substr):
    cols = list(combo_df().columns)
    cols_subset = [col for col in cols if substr in col]
    return cols_subset
