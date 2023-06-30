from pickle import load
import pandas as pd

def preprocess_data(df: pd.DataFrame):
    df_csv = pd.read_csv("data/credit_scoring.csv")

    df = pd.concat([df_csv, df], axis=0)

    one_hot_encoded = pd.get_dummies(df['RealEstateLoansOrLines'])
    df = pd.concat([df, one_hot_encoded], axis=1)
    df.drop('RealEstateLoansOrLines', axis=1, inplace=True)

    one_hot_encoded = pd.get_dummies(df['GroupAge'])
    df = pd.concat([df, one_hot_encoded], axis=1)
    df.drop('GroupAge', axis=1, inplace=True)
    df = df[
        ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', 'C',
         'b', 'c']]
    return df.tail(1)

def load_model_and_predict(df, path="data/model.pickle"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    prediction_df = pd.DataFrame(prediction_proba, index=[0])
    prediction_df = prediction_df.rename(columns={1: 'Не вернет кредит',
                                                  0: 'Вернет кредит'})

    if prediction == 1:
        text = "Клиент не вернет кредит"
    else:
        text = "Клиент вернет кредит"

    return text, prediction_df