# AI-model
# Psychometric Test API for Tech Enthusiasts

Welcome to the Psychometric Test API designed to assist tech enthusiasts in choosing the best learning track tailored to their interests and strengths. This API utilizes machine learning to predict the most suitable learning track based on the provided data.

## Models Used

We experimented with three machine learning models to predict the best learning track:

- **Linear Regression**
- **Decision Tree**
- **Random Forest**

After thorough testing and evaluation, we decided to go with **Random Forest** as it demonstrated the highest accuracy among the three models.

## Getting Started

### Prerequisites

Make sure you have Python installed on your system. You can download it from [here](https://www.python.org/downloads/).

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/psychometric-test-api.git
    ```

2. Navigate to the project directory:

    ```bash
    cd psychometric-test-api
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the API

Run the following command to start the Flask server:

```bash
python app.py

The API will start running on http://localhost:5000/.

## API Endpoints
/predict (POST)
This endpoint accepts POST requests with JSON data containing the user's information. The API then preprocesses the data and predicts the best learning track using the Random Forest model.

### Example Request:


{
  "age": 25,
  "gender": "Male",
  "programming_experience": "Intermediate",
  "preferred_language": "Python",
  "learning_style": "Visual"
}
### Example Response:


["Web Development"]
## Data Preprocessing
Before making predictions, the API preprocesses the data to match the format used during model training. This includes applying saved LabelEncoders to categorical columns. The preprocessing function ensures that the input data is transformed correctly before prediction.

## File Structure
app.py: Main Flask application file containing the API endpoints.
requirements.txt: List of Python packages required for the project.
DATA/: Directory containing serialized models and encoders.
rf_best_model.joblib: Serialized Random Forest model.
*_encoder.joblib: Serialized LabelEncoders for categorical columns.
## Deployment
The API is configured to run on any cloud platform or local server. You can set the PORT environment variable to specify the port number.


export PORT=5000

## Conclusion
We believe that this API will help tech enthusiasts make informed decisions about their learning paths, leading to more effective and enjoyable learning experiences. If you have any questions or suggestions, please feel free to reach out.
