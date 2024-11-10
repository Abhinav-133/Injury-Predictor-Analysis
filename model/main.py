#pip install pandas
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier


#function to test model

def create_model(data):

    # Setting x and y values
    y = data.loc[:,'injury'].values
    y = y.reshape(-1,1)
    X = data.iloc[:, 1:5 ].values
    # X = data.drop(['diagnosis'], axis=1)
    # y = data['diagnosis']

    #oversampeling
    oversample = SMOTE()
    X_new, y_new = oversample.fit_resample(X, y)

    # # Scale the data 
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Splitting the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.3, random_state=0)


    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification Report: ", classification_report(y_test, y_pred))



    return model




def get_clean_data():
   
    data = pd.read_csv("E:/ml-p/Sports-Injury-Analysis/final_data.csv")

    return data

def main():
   
    data = get_clean_data()

    model = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model,f) 

    # with open('model/ scaler.pkl','wb' ) as f:
    #     pickle.dump(scaler, f) 
      

   

#data.head() prints the first 5 rows and columns
#data.info prints all the columns name of the dataset, used to check clean data
    print(data.head())
    







#is a common Python idiom used to ensure that certain code is only executed when the script is run directly, 
# rather than when it is imported as a module in another script.
if __name__ == "__main__":
    main()








