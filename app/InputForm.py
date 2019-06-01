from flask_wtf import FlaskForm
from wtforms import SubmitField, SelectField
import pickle

data_folder = './data'


def load_fields():
    with open(data_folder + '/test_val.pkl', 'rb') as f:
        test_val = pickle.load(f)
    with open(data_folder + '/outcome_var.pkl', 'rb') as f:
        target_col = pickle.load(f)
    with open(data_folder + '/index_map.pkl', 'rb') as f:
        index_map = pickle.load(f)

    actual = test_val[target_col]
    test_data = {k: v for k, v in test_val.items() if not k == target_col}

    return index_map, test_data, actual


class Inputform(FlaskForm):
    index_map, test_data, actual = load_fields()

    user_id_options = [(k, k) for k, v in index_map['User_ID'].items()]
    user_id_default = test_data['User_ID']
    user_id = SelectField(u'UserId', choices=user_id_options, default=user_id_default)

    product_id_options = [(k, k) for k, v in index_map['Product_ID'].items()]
    product_id_default = test_data['Product_ID']
    product_id = SelectField(u'ProductId', choices=product_id_options, default=product_id_default)

    gender_options = [(k, k) for k, v in index_map['Gender'].items()]
    gender_default = test_data['Gender']
    gender = SelectField(u'Gender', choices=gender_options, default=gender_default)

    age_options = [(k, k) for k, v in index_map['Age'].items()]
    age_default = test_data['Age']
    age = SelectField(u'Age', choices=age_options, default=age_default)

    occupation_options = [(k, k) for k, v in index_map['Occupation'].items()]
    occupation_default = test_data['Occupation']
    occupation = SelectField(u'Occupation', choices=occupation_options, default=occupation_default)

    city_category_options = [(k, k) for k, v in index_map['City_Category'].items()]
    city_category_default = test_data['City_Category']
    city_category = SelectField(u'City Category', choices=city_category_options, default=city_category_default)

    years_in_city_options = [(k, k) for k, v in index_map['Stay_In_Current_City_Years'].items()]
    years_in_city_default = test_data['Stay_In_Current_City_Years']
    years_in_city = SelectField(u'Years In Current_City', choices=years_in_city_options, default=years_in_city_default)

    marital_status_options = [(k, k) for k, v in index_map['Marital_Status'].items()]
    marital_status_default = test_data['Marital_Status']
    marital_status = SelectField(u'Marital Status', choices=marital_status_options, default=marital_status_default)

    pc1_options = [(k, k) for k, v in index_map['Product_Category_1'].items()]
    pc1_default = test_data['Product_Category_1']
    pc1 = SelectField(u'Product Category 1', choices=pc1_options, default=pc1_default)

    pc2_options = [(k, k) for k, v in index_map['Product_Category_2'].items()]
    pc2_default = test_data['Product_Category_2']
    pc2 = SelectField(u'Product Category 2', choices=pc2_options, default=pc2_default)

    pc3_options = [(k, k) for k, v in index_map['Product_Category_3'].items()]
    pc3_default = test_data['Product_Category_3']
    pc3 = SelectField(u'Product Category 3', choices=pc3_options, default=pc3_default)

    submit = SubmitField('Submit')
