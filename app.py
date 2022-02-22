import numpy as np
from flask import Flask, request, jsonify, render_template
import dill as pickle
import pandas as pd

app = Flask(__name__)
# Load notebook output and assign values
summary = pickle.load(open('summary.pickle', 'rb'))
model = summary['model']
scaler = summary['scaler']
convert_dict = summary['convert_dict']
features = summary['features']


@app.route('/')
def home():
    """
    This function populates the dropdown menus in the landing page 
    """
    # Populate index.html with dropdown selections
    # Destination 
    countrylist = sorted(list(set(convert_dict.keys())))

    # Agency: find features with 'agency_' and remove 'agency_type_'
    agency = sorted([s.split('agency_')[-1] for s in features if 'agency_' in s])
    agency = [s for s in agency if not ('type_' in s)]

    # Select agency type: hardcoded 
    agency_type = ['Airlines', 'Travel Agency']

    # Select distribution channel: hardcoded
    distribution_channel = ['Offline', 'Online']

    # Select product name: find features with 'product_name_' and replace '_' with space
    product_name = sorted([s.split('product_name_')[-1] for s in features if 'product_name_' in s])
    product_name = [' '.join(s.split('_')) for s in product_name]

    # Return variables to the HTML
    return render_template("index.html",**locals())
    
@app.route('/predict',methods=['POST'])
def predict():
    """
    Post prediction from model
    """
    # Create a dataframe to store input values
    x_test = pd.DataFrame(np.zeros((1,len(features))),columns=features)

    # Convert form input to dictionary. Refer to 'parse_input' helper function
    parsed_data = parse_input(request.form.to_dict())

    # Populate dataframe with results from dictionary 
    for key in parsed_data.keys():
        x_test.loc[0,key] = parsed_data[key]

    # Use 'convert_dict' to convert country to continent 
    continent = convert_dict[request.form.get('country')].lower()

    # Replace space with '_'
    continent = '_'.join(continent.split())

    # Add continent if found in feature 
    if 'continent_'+continent in features:
        x_test.loc[0,'continent_'+continent] = 1

    # Scale test data 
    x_test = scaler.transform(x_test)

    # Convert prediction to integer
    pred = int(model.predict(x_test))

    # Choises for user prompt 
    choices = ['denied!', 'approved!']

    # Outgoing message 
    prediction_text = "Your claim has been "+choices[pred]

    # Return prediction to the webpage 
    return render_template('index.html', prediction_text=prediction_text)

def parse_input(in_dict):
    """
    This function customizes input from form to match features
    """
    # Output dictionary 
    output = {}

    # Iterate through each key
    for key in in_dict.keys():
        # Check if entry is a number 
        try:            
            output[key] = float(in_dict[key])
        except:
            # If not, make key by joining key and value with '_' 
            tmp = key+'_'+'_'.join(in_dict[key].split())
            # Add keys that only exist in features
            if tmp in features:
                # Add one-hot-encoded value 
                output[tmp] = 1

    return output

if __name__ == "__main__":
    app.run(debug=True)