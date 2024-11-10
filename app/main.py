import streamlit as st
import csv
import plotly.graph_objects as go
import pickle

def get_clean_data():
  
    data = []
    # Open and read the CSV file
    with open("E:/ml-p/Sports-Injury-Analysis/final_data.csv", mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # Drop unwanted columns and map diagnosis
            # if 'Unnamed: 32' in row:
            #     del row['Unnamed: 32']
            # if 'id' in row:
            #     del row['id']
            # row['diagnosis'] = 1 if row['diagnosis'] == 'M' else 0
            data.append(row)

    return data

def find_max_value(data, key):
    # Find the maximum value for the given key across all rows
    max_value = max(float(row[key]) for row in data if key in row)
    return max_value

def find_mean_value(data, key):
    # Calculate the mean value for the given key across all rows
    values = [float(row[key]) for row in data if key in row]
    mean_value = sum(values) / len(values) if values else 0
    return mean_value


def add_sidebar():
    st.sidebar.header("injury data ")

    data = get_clean_data()

  
    slider_labels = [
           ("Game workload (mean)","game_workload"),
           ("Groin squeeze (mean)","groin_squeeze"),
           ("Hip mobility (mean)","hip_mobility"),
           ("Rest period (mean)","rest_period")
        # ("Radius (mean)", "radius_mean"),
        # ("Texture (mean)", "texture_mean"),
        # ("Perimeter (mean)", "perimeter_mean"),
        # ("Area (mean)", "area_mean"),
        # ("Smoothness (mean)", "smoothness_mean"),
        # ("Compactness (mean)", "compactness_mean"),
        # ("Concavity (mean)", "concavity_mean"),
        # ("Concave points (mean)", "concave points_mean"),
        # ("Symmetry (mean)", "symmetry_mean"),
        # ("Fractal dimension (mean)", "fractal_dimension_mean"),
        # ("Radius (se)", "radius_se"),
        # ("Texture (se)", "texture_se"),
        # ("Perimeter (se)", "perimeter_se"),
        # ("Area (se)", "area_se"),
        # ("Smoothness (se)", "smoothness_se"),
        # ("Compactness (se)", "compactness_se"),
        # ("Concavity (se)", "concavity_se"),
        # ("Concave points (se)", "concave points_se"),
        # ("Symmetry (se)", "symmetry_se"),
        # ("Fractal dimension (se)", "fractal_dimension_se"),
        # ("Radius (worst)", "radius_worst"),
        # ("Texture (worst)", "texture_worst"),
        # ("Perimeter (worst)", "perimeter_worst"),
        # ("Area (worst)", "area_worst"),
        # ("Smoothness (worst)", "smoothness_worst"),
        # ("Compactness (worst)", "compactness_worst"),
        # ("Concavity (worst)", "concavity_worst"),
        # ("Concave points (worst)", "concave points_worst"),
        # ("Symmetry (worst)", "symmetry_worst"),
        # ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
         input_dict[key] = st.sidebar.slider(
             label,
             min_value = 0.0,
             max_value= find_max_value(data, key),
             value = find_mean_value(data, key)
         )

    return input_dict


def to_numeric(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
    

def get_scaled_values(input_dict):
    data = get_clean_data()
    
    # Extract column names except 'diagnosis'
    feature_columns = [key for key in data[0].keys() if key != 'diagnosis']
    
    # Initialize dictionaries to hold min and max values for each feature
    min_values = {key: float('inf') for key in feature_columns}
    max_values = {key: float('-inf') for key in feature_columns}
    
    # Calculate min and max values for each feature
    for row in data:
        for key in feature_columns:
            value = to_numeric(row.get(key))
            if value is not None:
                min_values[key] = min(min_values[key], value)
                max_values[key] = max(max_values[key], value)
    
    # Scale the input dictionary based on min and max values
    scaled_dict = {}
    for key, value in input_dict.items():
        value = to_numeric(value)
        if value is not None and key in min_values and key in max_values:
            min_val = min_values[key]
            max_val = max_values[key]
            scaled_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
            scaled_dict[key] = scaled_value
    
    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    
    categories = ['Game Workload', 'Groin Squeeze', 'Hip Mobility', 'Rest Period']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
            r=[
            input_data['game_workload'], input_data['groin_squeeze'], input_data['hip_mobility'],
            input_data['rest_period']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
             r=[
            input_data['game_workload'], input_data['groin_squeeze'], input_data['hip_mobility'],
            input_data['rest_period']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
             r=[
            input_data['game_workload'], input_data['groin_squeeze'], input_data['hip_mobility'],
            input_data['rest_period']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
    )
    
    return fig


def add_predictions(input_data):
    model = pickle.load(open("E:/ml-p/Sports-Injury-Analysis/Injury-Predictor-Analysis/model/model.pkl", "rb"))
    #scaler = pickle.load(open("E:/ml-p/Sports-Injury-Analysis/Injury-Predictor-Analysis/model/scaler.pkl", "rb"))
    
    # Convert input data to a list of floats
    input_list = [float(value) for value in input_data.values()]

    # Reshape the input list to be 2D (single sample with multiple features)
    input_array = [input_list]

    # Scale the input array
    #input_array_scaled = scaler.transform(input_array)

    # Predict using the model
    prediction = model.predict(input_array)
    prediction_proba = model.predict_proba(input_array)

    # Display the results
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>injury will  not occur</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>injury will occur</span>", unsafe_allow_html=True)
    
    st.write("Probability of being benign: ", prediction_proba[0][0])
    st.write("Probability of being malicious: ", prediction_proba[0][1])
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")






def main():
    st.set_page_config(
        page_title="injury  Predictor",
        page_icon = "female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"

     )
    
    input_data = add_sidebar()


    with st.container():
        st.title("injury Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == "__main__":
    main()