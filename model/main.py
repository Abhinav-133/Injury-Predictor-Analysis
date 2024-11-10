import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

DATA_FILE_PATH = "/Users/abhinavmittal/Desktop/Injury-Predictor-Analysis/final_data.csv"
MODEL_FILE_PATH = "/Users/abhinavmittal/Desktop/Injury-Predictor-Analysis/model/model.pkl"
SCALER_FILE_PATH = "/Users/abhinavmittal/Desktop/Injury-Predictor-Analysis/model/scaler.pkl"

def get_clean_data():
    try:
        data = pd.read_csv(DATA_FILE_PATH)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {DATA_FILE_PATH} was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The CSV file is corrupted or incorrectly formatted.")
        return None

def create_model(data):
    X = data.iloc[:, 1:5].values  
    y = data['injury'].values
    
    oversample = SMOTE()
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    scaler = StandardScaler()
    X_resampled = scaler.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=0)

    model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=0)  # Tuning parameters
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, scaler

def main():
    data = get_clean_data()
    if data is None:
        return  

    model, scaler = create_model(data)

    try:
        with open(MODEL_FILE_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)
        print(f"Model saved to {MODEL_FILE_PATH}")
        
        with open(SCALER_FILE_PATH, 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        print(f"Scaler saved to {SCALER_FILE_PATH}")
    except Exception as e:
        print(f"Error saving model or scaler: {e}")

    print("\nSample Data Overview:")
    print(data.head())

if __name__ == "__main__":
    main()
