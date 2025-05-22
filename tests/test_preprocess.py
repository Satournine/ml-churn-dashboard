import pandas as pd
from app.preprocess import preprocess_input


def test_preprocess_output_shape():
    input_df = pd.DataFrame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 50.0,
                "TotalCharges": 600.0,
            }
        ]
    )

    processed = preprocess_input(input_df)
    assert isinstance(processed, pd.DataFrame)
    assert processed.shape[0] == 1
    assert processed.shape[1] > 0
