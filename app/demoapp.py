from flask import Flask, render_template, request
import json
import requests
from InputForm import Inputform


data_folder = './data'
scoring_uri = 'http://5298bb33-0d7b-4376-ab1d-c04e79f718df.westus.azurecontainer.io/score'
key = 'O13psE1xgbdmsZpVjyzE8BZs69dh6uBw'


app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'



def get_data(form):
    return {
        'User_ID': form.user_id.data,
        'Product_ID': form.product_id.data,
        'Gender': form.gender.data,
        'Age': form.age.data,
        'Occupation': form.occupation.data,
        'City_Category': form.city_category.data,
        'Stay_In_Current_City_Years': form.years_in_city.data,
        'Marital_Status': form.marital_status.data,
        'Product_Category_1': form.pc1.data,
        'Product_Category_2': form.pc2.data,
        'Product_Category_3': form.pc3.data
    }


@app.route('/', methods=['GET', 'POST'])
def display():
    form = Inputform(request.form)
    prediction = None
    test_data=None
    if form.validate_on_submit():
        test_data = get_data(form)
        test_data_json = bytes(json.dumps(test_data), encoding='utf8')
        headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}
        resp = requests.post(scoring_uri, test_data_json, headers=headers)
        prediction = resp.text
    return render_template('demoapp.html',
                           form=form,
                           debug=test_data,
                           prediction=f"${prediction}" if prediction is not None else "")


if __name__ == "__main__":
    app.run()
