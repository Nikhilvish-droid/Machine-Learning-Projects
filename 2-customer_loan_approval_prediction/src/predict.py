import joblib
from pathlib import Path 
import pandas as pd

model_dir = Path(__file__).resolve().parent.parent/'models'

def load_model():
    model_pipeline = joblib.load(model_dir/'loan_status_model.pkl')
    return model_pipeline


def user():

    name = input("enter your name: ")
    education = int(input("Graduate? (1 = Yes, 0 = No): "))
    no_of_dependents = int(input("Number of person depends on you in your house: "))
    self_employed = int(input("Self employed? (1 = Yes, 0 = No): "))

    income_annum = float(input("Annual income: "))
    loan_amount = float(input("Loan amount: "))
    loan_term = int(input("Loan term (in years): "))
    cibil_score = int(input("CIBIL score: "))

    residential_assets_value = float(input("Residential assets value: "))
    commercial_assets_value = float(input("Commercial assets value: "))
    luxury_assets_value = float(input("Luxury assets value: "))
    bank_asset_value = float(input("Bank asset value: "))

    user_data = pd.DataFrame([{
        'no_of_dependents': no_of_dependents,
        'education': education,
        'self_employed': self_employed,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value
    }])

    model = load_model()

    loan_status = model.predict(user_data)[0]

    if loan_status == 1:
        print(f" hey {name}! your loan will be approved")
    else:
        print(f" hey {name}! unfortunately your loan will be Rejected")


if __name__ == '__main__':
    user()