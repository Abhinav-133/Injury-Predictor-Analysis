import streamlit as st
import csv
import plotly.graph_objects as go
import pickle

# Load and clean data
def get_clean_data():
    data = []
    try:
        with open("/Users/abhinavmittal/Desktop/Injury-Predictor-Analysis/final_data.csv", mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
    except FileNotFoundError:
        st.error("Data file not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
    return data

# Function to find max value for sliders
def find_max_value(data, key):
    values = [float(row[key]) for row in data if key in row]
    return max(values) if values else 1

# Function to calculate mean value for sliders
def find_mean_value(data, key):
    values = [float(row[key]) for row in data if key in row]
    return sum(values) / len(values) if values else 0

def to_numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
# Sidebar for user input
def add_sidebar():
    st.sidebar.header("Injury Data")

    data = get_clean_data()
    slider_labels = [
        ("Game workload (mean)", "game_workload"),
        ("Groin squeeze (mean)", "groin_squeeze"),
        ("Hip mobility (mean)", "hip_mobility"),
        ("Rest period (mean)", "rest_period")
    ]
    
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value=find_max_value(data, key),
            value=find_mean_value(data, key)
        )

    return input_dict

# Scaling function for user inputs
def get_scaled_values(input_dict):
    data = get_clean_data()
    feature_columns = [key for key in data[0].keys() if key != 'diagnosis']
    
    min_values = {key: float('inf') for key in feature_columns}
    max_values = {key: float('-inf') for key in feature_columns}

    for row in data:
        for key in feature_columns:
            value = to_numeric(row.get(key))
            if value is not None:
                min_values[key] = min(min_values[key], value)
                max_values[key] = max(max_values[key], value)

    scaled_dict = {}
    for key, value in input_dict.items():
        value = to_numeric(value)
        min_val = min_values[key]
        max_val = max_values[key]
        scaled_dict[key] = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
    
    return scaled_dict

# Radar chart visualization
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Game Workload', 'Groin Squeeze', 'Hip Mobility', 'Rest Period']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['game_workload'], input_data['groin_squeeze'],
            input_data['hip_mobility'], input_data['rest_period']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True
    )
    
    return fig

# Prediction based on user input
def add_predictions(input_data):
    try:
        model = pickle.load(open("/Users/abhinavmittal/Desktop/Injury-Predictor-Analysis/model/model.pkl", "rb"))
        # scaler = pickle.load(open("E:/ml-p/Sports-Injury-Analysis/Injury-Predictor-Analysis/model/scaler.pkl", "rb"))

        input_list = [float(value) for value in input_data.values()]
        input_array = [input_list]

        # input_array_scaled = scaler.transform(input_array)  # Uncomment if scaler is used

        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)

        st.subheader("Injury Prediction")
        if prediction[0] == 0:
            st.write("**Injury will not occur**")
        else:
            st.write("**Injury will occur**")

        st.write("Probability of no injury: ", prediction_proba[0][0])
        st.write("Probability of injury: ", prediction_proba[0][1])

    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model file is in the correct path.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Main function to run the app
def main():
    st.set_page_config(
        page_title="Injury Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    input_data = add_sidebar()

    st.title("Injury Predictor")
    st.write("This app predicts the likelihood of injury occurrence based on training and physical metrics. Adjust the measurements using the sliders in the sidebar.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()
