# CS 315 - Data Mining
# Final Project
# Author: Amethyst Skye
# Language: Python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import mean_absolute_percentage_error


# Reads data file and prepocesses data for training
def read_and_clean(file_name, price_lower_thresh, price_upper_thresh):
    # Open file
    df = pd.read_csv(file_name)
    
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Clean original sale amount
    df['ORIGINAL SALE AMOUNT'] = df['ORIGINAL SALE AMOUNT'].values.astype(str)
    df['ORIGINAL SALE AMOUNT'] = df['ORIGINAL SALE AMOUNT'].map(lambda x: str(x.lstrip('$'))).str.replace(',', '')
    
    # Drop rows with empty values for specified columns
    df = df.dropna(axis = 0, subset = [' YEAR BUILT', 
                                       'STYLE',
                                       'QUALITY digital',
                                       'PARCEL SIZE ACRES',
                                       'MAIN AND UPPER LIVING AREA',])
    
    # One Hot Encoding for categorical data
    # Categorical data -> numerical data
    
    # Home style column needs to be encoded                       
    style_one_hot_encoder = OneHotEncoder()
    style_results = style_one_hot_encoder.fit_transform(df[['STYLE']])
    df = df.join(pd.DataFrame(style_results.toarray(), columns = style_one_hot_encoder.categories_))
    
    # List of areas we are including in our dataset
    area_list = ['YACOLT', 
                'AMBOY', 
                'LA CENTER', 
                'WOODLAND', 
                'RIDGEFIELD',
                'BATTLE GROUND',
                'BRUSH PRAIRIE',
                'VANCOUVER',
                'CAMAS',
                'WASHOUGAL']

    # Only include addresses with these city names
    for item in area_list:
        df.loc[df['Parcel Address'].str.contains(item, na = False), 'Parcel Address'] = item    
    
    # Home area column needs to be encoded
    area_one_hot_encoder = OneHotEncoder()
    area_results = area_one_hot_encoder.fit_transform(df[['Parcel Address']])
    df = df.join(pd.DataFrame(area_results.toarray(), columns = area_one_hot_encoder.categories_))
    
    # Ensuring we have the right data type for training/testing
    df['ORIGINAL SALE AMOUNT'] = df['ORIGINAL SALE AMOUNT'].values.astype(int)
    
    # Drop unwanted columns
    df = df.drop(columns = ['BUILDING TYPE',
                            'Parcel Address', 
                            'STYLE',
                            'QUALITY',
                            'BASEMENT AREA', 
                            'ADJUSTED SALE AMOUNT',
                            'PARCEL SIZE SQ FT',
                            'VIEW',
                            'WATERFRONT', 
                            'SALE DATE', 
                            'ASSESSOR NH (REFERENCE NO)'])
    
    # Original sale amount must be last column
    df.insert(len(df.columns) - 1, 'ORIGINAL SALE AMOUNT', df.pop('ORIGINAL SALE AMOUNT'))
    
    # Do not include houses that are priced above the defined threshold
    df = df[df['ORIGINAL SALE AMOUNT'] < price_upper_thresh]
    df = df[df['ORIGINAL SALE AMOUNT'] > price_lower_thresh]
    
    # Drop any rows with NaN
    df = df.dropna()
    
    return(df)

# Returns unique values in dataframe column
def distinct_styles(df, column_name):
    return(df[column_name].unique())

# Splits training and test set
def train_test_splitting(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Splitting dataset into training/testing
    print("Processing Data...")
    # Training data = 80%
    # Testing data = 20%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    
    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return(x_train, x_test, y_train, y_test)

# Support Vector Machine Implementation
# SVM Learning Algorithm
def svm_learner(x_train, x_test, y_train, y_test):
    # Training SVM model on training set
    print("Training...")
    model_SVR = svm.SVR()
    model_SVR.fit(x_train,y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = model_SVR.predict(x_test)
    
    return(y_pred, y_test)

# Linear Regression Implementation
def linear_reg_learner(x_train, x_test, y_train, y_test):
    # Training Linear Regression model on training set
    print("Training...")
    model_LR = LinearRegression()
    model_LR.fit(x_train, y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = model_LR.predict(x_test)
    
    return (y_pred, y_test)

# Decision Tree Implementation
def decision_tree_learner(x_train, x_test, y_train, y_test):    
    # Training Decision Tree Regression model on training set
    print("Training...")
    classifier = DecisionTreeRegressor(random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = classifier.predict(x_test)
    
    return(y_pred, y_test)

# Random Forest Implementation
def random_forest_learner(x_train, x_test, y_train, y_test):
    # Training Random Forest model on training set
    print("Training...")
    classifier = RandomForestRegressor(random_state = 0)
    classifier.fit(x_train, y_train)
    
    # Predicting test set results
    print("Testing...")
    y_pred = classifier.predict(x_test)
    
    return (y_pred, y_test)

def learning_analysis(y_pred, y_test):
    y_pred = y_pred.round()
    diff_vec = []
    diff_over = []
    diff_under = []
    
    for i in range(0,len(y_pred)):
        ratio_diff = y_pred[i] / y_test[i]
        diff_vec.append(ratio_diff)
        if (ratio_diff < 1):
            diff_under.append(y_pred[i] - y_test[i])
        else:
            diff_over.append(y_pred[i] - y_test[i])
            
    # Pred Matrix:
    # [[Predicted Actual]]
    print("\nPrediction results: ")
    print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1), "\n")
    print("Mean Absolute Error: ")
    print(mean_absolute_percentage_error(y_test, y_pred), "\n")
    print("Absolute Average Difference: ")
    print(statistics.mean(abs(y_pred - y_test)), "\n")
    print("Over-predictions: ")
    print("Average =", statistics.mean(diff_over), ", Count:", len(diff_over), "\n")
    print("Under-predictions: ")
    print("Average =", statistics.mean(diff_under), ", Count:", len(diff_under))
    x_axis = []
    
    for i in range(1, len(y_pred) + 1):
        x_axis.append(i)
        
    # Data Visualization    
    plt.figure(figsize=(10,10))
    plt.scatter(y_test, y_pred, c = 'crimson')
    plt.yscale('linear')
    plt.xscale('linear')
    p1 = max(max(y_pred), max(y_test))
    p2 = min(min(y_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.title('Predictions vs Actual (in Million $)')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()
    
    return

def main():
    # Import and clean data
    house_df = read_and_clean('Data/housing_data.csv', 0, 500000)
    
    # Copy of clean dataset
    house_df.to_csv('Data/clean_house.csv', sep=',')
    
    # Create training and testing sets
    x_train, x_test, y_train, y_test = train_test_splitting(house_df)
    
    # --- ML Model implementation Options --- #
    
    # 1. Linear Regression
    #y_pred, y_test = linear_reg_learner(x_train, x_test, y_train, y_test)
    
    # 2. Decision Tree
    #y_pred, y_test = decision_tree_learner(x_train, x_test, y_train, y_test)
    
    # 3. Random Forest (this model yielded the best prediction results)
    y_pred, y_test = random_forest_learner(x_train, x_test, y_train, y_test)
    
    # Analyze results of model usage
    learning_analysis(y_pred, y_test)
    
if __name__ == "__main__":
    main()