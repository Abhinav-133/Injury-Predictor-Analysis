import streamlit as st
import csv
import plotly.graph_objects as go
import pickle
import pandas as pd

# Load and clean data
def get_clean_data():
    data = []
    try:
        with open("./data/final_data.csv", mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data.append(row)
    except FileNotFoundError:
        st.error("Data file not found.")
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
    return data

def find_max_value(data, key):
    values = [float(row[key]) for row in data if key in row]
    return float(max(values)) if values else 1.0

def find_mean_value(data, key):
    values = [float(row[key]) for row in data if key in row]
    return float(sum(values) / len(values)) if values else 0.0


# Function to transform value to numeric
def to_numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

# Input parameters with number fields and improved layout
def add_input_fields():
    st.subheader("Input Parameters")
    data = get_clean_data()
    if not data:
        st.error("No data available to generate input fields. Check your data file.")
        st.stop()

    st.write("Adjust the parameters below to predict the likelihood of injury:")

    col1, col2 = st.columns(2)
    with col1:
        game_workload = st.number_input(
            "Game Workload (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "game_workload"),
            value=find_mean_value(data, "game_workload"),
        )
        groin_squeeze = st.number_input(
            "Groin Squeeze (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "groin_squeeze"),
            value=find_mean_value(data, "groin_squeeze"),
        )
    with col2:
        hip_mobility = st.number_input(
            "Hip Mobility (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "hip_mobility"),
            value=find_mean_value(data, "hip_mobility"),
        )
        rest_period = st.number_input(
            "Rest Period (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "rest_period"),
            value=find_mean_value(data, "rest_period"),
        )

    input_dict = {
        "game_workload": game_workload,
        "groin_squeeze": groin_squeeze,
        "hip_mobility": hip_mobility,
        "rest_period": rest_period,
    }
    return input_dict

    st.subheader("Input Parameters")
    data = get_clean_data()
    st.write("Adjust the parameters below to predict the likelihood of injury:")

    col1, col2 = st.columns(2)
    with col1:
        game_workload = st.number_input(
            "Game Workload (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "game_workload"),
            value=find_mean_value(data, "game_workload"),
        )
        groin_squeeze = st.number_input(
            "Groin Squeeze (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "groin_squeeze"),
            value=find_mean_value(data, "groin_squeeze"),
        )
    with col2:
        hip_mobility = st.number_input(
            "Hip Mobility (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "hip_mobility"),
            value=find_mean_value(data, "hip_mobility"),
        )
        rest_period = st.number_input(
            "Rest Period (mean)",
            min_value=0.0,
            max_value=find_max_value(data, "rest_period"),
            value=find_mean_value(data, "rest_period"),
        )

    input_dict = {
        "game_workload": game_workload,
        "groin_squeeze": groin_squeeze,
        "hip_mobility": hip_mobility,
        "rest_period": rest_period,
    }
    return input_dict

# Radar chart visualization
def get_radar_chart(input_data):
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

# Prediction display based on user input
def add_predictions(input_data):
    try:
        model = pickle.load(open("./model.pkl", "rb"))
        
        input_list = [float(value) for value in input_data.values()]
        input_array = [input_list]

        prediction = model.predict(input_array)
        prediction_proba = model.predict_proba(input_array)

        st.subheader("Injury Prediction Results")
        st.write("Based on your input parameters, here are the results:")

        if prediction_proba[0][0] < 0.5:
            st.success("**Low risk: Injury is unlikely**")
        else:
            st.error("**High risk: Injury is likely**")
            

        st.write("### Probability Scores")
        st.write(f"- Probability of injury: **{prediction_proba[0][0]:.2f}**")
        st.write(f"- Probability of no injury: **{prediction_proba[0][1]:.2f}**")

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

    st.title("Athlete Injury Predictor")
    st.write("This application predicts the likelihood of an athlete getting injured based on training and physical metrics. Use the sliders below to adjust the parameters.")

    input_data = add_input_fields()

    st.markdown("---")

    st.subheader("Visualize Input Data")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.plotly_chart(get_radar_chart(input_data))

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()
