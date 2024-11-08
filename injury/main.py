#pip install pandas
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

import pickle
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np


#function to test model

def create_model(data):

    # Setting x and y values
    # X = data.drop(['diagnosis'], axis=1)
    # y = data['diagnosis']
    y = data.loc[:,'injury'].values
    y = y.reshape(-1,1)
    X = data.iloc[:, 1:5 ].values
    
    # oversampling
    oversample = SMOTE()
    X_new, y_new = oversample.fit_resample(X, y)

    # # Scale the data 
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Splitting the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=0)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification Report: ", classification_report(y_test, y_pred))



    return model




def get_clean_data():
    # reading files
    metrics = pd.read_csv("data/metrics.csv")
    workload = pd.read_csv("data/game_workload.csv")
    injuries = pd.read_csv("data/injuries.csv")
    
    # changing date datatype
    metrics['date'] = metrics['date'].astype('datetime64[ns]')
    workload['date'] = workload['date'].astype('datetime64[ns]')
    injuries['date'] = injuries['date'].astype('datetime64[ns]')
    injuries["injury"] = "Yes"
    games_data = pd.merge(workload,injuries,  how='left', left_on=['athlete_id','date'], right_on = ['athlete_id','date'])
    games_data["injury"].fillna("No", inplace = True)
    new_metrics_df = metrics.pivot_table('value', ['athlete_id', 'date'], 'metric').reset_index()
    final_data = pd.merge(games_data,new_metrics_df,  how='left', left_on=['athlete_id','date'], right_on = ['athlete_id','date'])
    final_data['rest_period'] = final_data.groupby('athlete_id')['date'].diff()
    first_day = '2016-05-01'
    date_object = pd.to_datetime(first_day)
    final_data["rest_period"].fillna(final_data['date'] - date_object, inplace = True)
    final_data['rest_period'] = final_data['rest_period']/np.timedelta64(1,'D')
    final_data.injury.replace(to_replace=['No', 'Yes'], value=[0, 1], inplace = True)
    final_data = final_data[['injury','athlete_id','date','game_workload','groin_squeeze','hip_mobility','rest_period']]
    ready_data=final_data
    ready_data.drop('athlete_id', axis = 1, inplace= True)
    ready_data.drop('date', axis = 1, inplace = True)
    # y = ready_data.loc[:,'injury'].values
    # y = y.reshape(-1,1)
    # X = ready_data.iloc[:, 1:5 ].values
   
    # data = pd.read_csv("data/data.csv")

    # data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    
    return ready_data

def main():
   
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model,f) 

    with open('model/ scaler.pkl','wb' ) as f:
        pickle.dump(scaler, f) 
      

   

#data.head() prints the first 5 rows and columns
#data.info prints all the columns name of the dataset, used to check clean data
    print(data.head())
    







#is a common Python idiom used to ensure that certain code is only executed when the script is run directly, 
# rather than when it is imported as a module in another script.
if __name__ == "__main__":
    main()








