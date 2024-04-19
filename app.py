from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

#load the serialized model
model_path = os.path.join('DATA', 'rf_best_model.joblib')
loaded_model = load(model_path)



def preprocess_data(new_data):
    """
    Preprocess new_data by applying saved LabelEncoders to 
    the categorical columns to ensure the data matches the training format.
    """

    for col in new_data.columns:
        if new_data[col].dtype == object: #to check if the column contains categorical data
            #sanitize the column names to match how the encoder files were saved
            safe_col_name = col.replace(' ','_').replace('/','_').replace('\\','_').replace('-','_')
           
            #construct the path to the encoder file
            encoder_path = os.path.join('DATA', f'{safe_col_name}_encoder.joblib')
            
            #Load the corresponding encoder
            encoder = load(encoder_path)
            # Apply the encoder to transform the data
            new_data[col] = encoder.transform(new_data[col])
    return new_data

@app.route('/predict', methods=['POST'])
def predict():
    # extract data from the POST request
    data = request.get_json(force=True)
    new_data = pd.DataFrame([data])  # this is to convert data into dataframe

    # Preprocess the new data to fit the model's requirements using saved encoders
    preprocessed_data = preprocess_data(new_data)

    # predict using the loaded data
    predictions = loaded_model.predict(preprocessed_data)
    
    # Additional information about the predicted track
    track_info = {
        "predicted_track": predictions[0],
        "details": {
            "learning_path": "Web Development",
            "recommended_resources": ["Online courses", "Books", "Tutorials"],
            "next_steps": "Start with HTML and CSS"
        }
    }

    # Return the track information in JSON format
    return jsonify(track_info)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port)
