import pandas as pd
import streamlit as st
from PIL import Image
from model_use import preprocess_data, load_model_and_predict


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/scoring.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Credit Scoring",
        page_icon=image,

    )

    st.write(
        """
        # Прогноз возврата клиентом кредита
        """
    )

    st.image(image)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Введите параметры')
    user_input_df = sidebar_input_features()

    preprocess_df = preprocess_data(user_input_df)

    st.write("## Ваши данные")
    st.write(user_input_df)
    st.write(
        """
        - `TotalBalance`: общий баланс средств
- `age`: возраст заемщика
- `DaysPastDueNotWorse_1`: сколько раз за последние 2 года наблюдалась просрочка 30-59 дней
- `DebtRatio`: ежемесячные расходы деленные на месячный доход
- `MonthlyIncome`: ежемесячный доход
- `NumberOfOpenCreditLinesAndLoans`: количество открытых кредитов и кредитных карт
- `NumberOfTimes90DaysLate`: сколько раз наблюдалась просрочка (90 и более дней)
- `NumberRealEstateLoansOrLines`: количество кредиов (в том числе под залог жилья)
- `RealEstateLoansOrLines`: закодированное количество кредиов - чем больше код буквы, тем больше кредитов
- `DaysPastDueNotWorse_2`: сколько раз за последние 2 года заемщик задержал платеж на 60-89 дней
- `NumberOfDependents`: количество иждивенцев на попечении (супруги, дети и др)
- `GroupAge`: закодированная возрастная группа - чем больше код, тем больше возраст
        """
    )

    prediction, prediction_probas = load_model_and_predict(preprocess_df)
    write_prediction(prediction, prediction_probas)

def sidebar_input_features():
    RevolvingUtilizationOfUnsecuredLines = st.sidebar.text_input("Общий баланс средств:", value="100000")
    DaysPastDueNotWorse_1 = st.sidebar.text_input(
        "Сколько раз за последние 2 года наблюдалась просрочка 30-59 дней?", value = "0")
    DebtRatio = st.sidebar.text_input("Ежемесячные расходы:", value = "10000")
    MonthlyIncome = st.sidebar.text_input("Ежемесячный доход:", value = "30000")
    NumberOfOpenCreditLinesAndLoans = st.sidebar.slider("Количество открытых кредитов и кредитных карт:", min_value=1,
                                                    max_value=80, value=13,
                                                    step=1)
    NumberOfTimes90DaysLate = st.sidebar.text_input("Сколько раз наблюдалась просрочка 90 и более дней?", value="0")
    NumberRealEstateLoansOrLine = st.sidebar.text_input("Количество кредитов:", value = "0")
    RealEstateLoansOrLines = st.sidebar.selectbox("Закодированное количество кредиов", ("A", "B", "C", "D", "E"))
    DaysPastDueNotWorse_2 = st.sidebar.text_input("Сколько раз за последние 2 года был задержан платеж на 60-89 дней?", value="0")
    NumberOfDependents = st.sidebar.slider("Количество иждивенцев на попечении (супруги, дети и др):", min_value=1,
                                       max_value=30, value=10,
                                       step=1)
    GroupAge = st.sidebar.selectbox("Закодированная возрастная группа", ("a", "b", "c", "d", "e"))
    age = st.sidebar.slider("Возраст", min_value=1, max_value=110, value=22,
                            step=1)

    data = {
        "TotalBalance": int(RevolvingUtilizationOfUnsecuredLines),
        "Age": age,
        "DaysPastDueNotWorse_1": int(DaysPastDueNotWorse_1),
        "DebtRatio": int(DebtRatio),
        "MonthlyIncome": int(MonthlyIncome),
        "NumberOfOpenCreditLinesAndLoans": NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate": int(NumberOfTimes90DaysLate),
        "NumberRealEstateLoansOrLine": int(NumberRealEstateLoansOrLine),
        "RealEstateLoansOrLines": RealEstateLoansOrLines,
        "DaysPastDueNotWorse_2": int(DaysPastDueNotWorse_2),
        "NumberOfDependents": NumberOfDependents,
        "GroupAge": GroupAge
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()