
import streamlit as st
from ui.sidebar_menu import display_menu
from utils.model_loader import load_models
from views.upload_predict import render_upload_predict
from views.eda import render_eda
from views.shap_explain import render_shap
from utils.load_data import load_Data
from utils.print_helper import print_data_distribution, prin_df_head,print_dataset_info,print_classification,print_confusion_matrix
from utils.preprocessing import prepare_pivoted_data, label_faults
from utils.model_builder import split_data, build_rf_model, predict_model, get_scaler
from config import features

#from utils.save_util import save_pivot_df

def main():
    menu = display_menu()
    # Load the Data and pre process
    df=load_Data()
    print_dataset_info(df)

    pivot_df=prepare_pivoted_data(df)
    label_faults(pivot_df)
    print_data_distribution(pivot_df)
    print_dataset_info(pivot_df)
    #save_pivot_df(pivot_df)

    # Build the model and train
    X = pivot_df[features]
    y = pivot_df['Fault']
    rf_scaler, X_scaled=get_scaler(X)

    X_train, X_test, y_train, y_test=split_data(X_scaled, y)
    rf_model=build_rf_model(X_train, y_train)

    y_pred= predict_model(rf_model,X_test)

    # Print the confusion matrix and classification report   
    print_classification(y_test, y_pred)

    fig = print_confusion_matrix(y_test, y_pred)
    st.pyplot(fig)  # For Streamlit

    # save the Model 

    # Load the Model
    #rf_model, rf_scaler, lstm_model, lstm_scaler = load_models()




    if menu == "Upload & Predict":
        render_upload_predict(rf_model, rf_scaler)
    elif menu == "EDA":
        render_eda(pivot_df)

    # if menu == "Upload & Predict":
    #     render_upload_predict(rf_model, rf_scaler, lstm_model, lstm_scaler)
    # elif menu == "EDA":
    #     render_eda()
    # elif menu == "SHAP Explainability":
    #     render_shap(rf_model, rf_scaler)

if __name__ == "__main__":
    main()
