import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
le = preprocessing.LabelEncoder()


def page_config():
    st.set_page_config(layout= "wide")
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://media.istockphoto.com/id/1406528811/photo/dark-brown-rough-texture-toned-concrete-wall-surface-close-up-brown-background-with-space-for.jpg?s=612x612&w=0&k=20&c=KeT1jdiXSsrJjqk5-wlW_8DB-8nqWe4rU9JKZbmyF-4=");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
# background-image: url("https://news.mit.edu/sites/default/files/styles/news_article__image_gallery/public/images/202402/MIT-Diagnostic-Accuracy-01-press_0.jpg?itok=dDIIq7Hb");
    st.write("""
    <div style='text-align:center'>
        <h1 style='color:#20EEF7;'>Industrial Copper Modeling Prediction</h1>
    </div>
    """,unsafe_allow_html=True)
    
page_config()

## Below code will create the header for sidebar as well the options table 

selected = option_menu(None, ["Home","Predict Selling Price","Predict Status"], 
                icons=["house","flag-fill","bar-chart-line"],
                menu_icon= "menu-button-wide",
                orientation= "horizontal",
                default_index=0,
                styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin": "-1px", "--hover-color": "#A7A405"},
                        "nav-link-selected": {"background-color": "#A7A405"}})

if selected == "Home":
    st.write(" ")
    st.markdown("### <span style='color:#20EEF7;'>Overview :</span>",
             unsafe_allow_html=True)
    st.markdown("#### <span style='color:#000500;'>This streamlit app aims to give users a friendly environment which can be used to predict Selling price and Status of Copper.</span>",
             unsafe_allow_html=True)
    st.write(" ")
    st.markdown("### <span style='color:#20EEF7;'>Objective :</span>",
            unsafe_allow_html=True)
    st.markdown("### <span style='color:#000500;'>This project focuses on building Machine Learning algorithms to predict industrial copper data for 'Selling Price' and 'Status' using various libraries such as pandas, numpy, scikit-learn. The objective of the project is to preprocess the data, handle missing values, detect outliers, and handle skewness.</span>",
             unsafe_allow_html=True)
    
    col1 ,col2=st.columns([2,2])
    with col1:
            st.write("#### <span style='color:#20EEF7;'>Technologies used</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#000500;'>- PYTHON   (PANDAS, NUMPY)</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#000500;'>- SCIKIT-LEARN</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#000500;'>- DATA PREPROCESSING</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#000500;'>- EXPLORATORY DATA ANALYSIS</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#000500;'>- STREAMLIT</span>",unsafe_allow_html=True)

    with col2:
            st.write("#### <span style='color:#20EEF7;'>MACHINE LEARNING MODEL</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#000500;'>REGRESSION - ***:red[RandomForestRegressor]***</span>",unsafe_allow_html=True)
            st.write("- The RandomForestClassifier is an ensemble learning method that combines multiple decision trees and consider the highest frequency out of all to create a robust and accurate Regression model.")
            st.write("#### <span style='color:#000500;'>CLASSIFICATION - ***:green[KnnClassifier]***</span>",unsafe_allow_html=True)
            st.write("- The KnnClassifier is an algorithm that is used to group the nearby data based on grouping methods and gets the highest frequency data out of those nearby for better accuracy of the model.")


elif selected == "Predict Selling Price":
    st.write("### :green[Copper Selling Price prediction ]")
    col1,col2= st.columns([1,1],gap="large")
    
    with col1:
        country = st.text_input("Country code(eg: 28,33 etc..)")
        status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                        'Wonderful', 'Revised', 'Offered', 'Offerable']
        status = st.selectbox(label='Status', options=status_values)
        status_dict = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}
        item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
        item_type = st.selectbox(label='Item Type', options=item_type_values)
        item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}
    with col2:
        width = st.number_input(label="Enter the Width of the Copper (Min: 0.1 & Max: infinity)",min_value=0.01)
        quantity_tons = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: infinity)')
        thickness = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)
        application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                        27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                        59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
        application = st.selectbox(label='Application value', options=application_values)
    

    if ((country == "") | (quantity_tons == "")):
        st.warning("Please fill all the details")
    else:
        if st.button("Predict Copper Selling Price"):
            x = [status_dict[status],np.log(float(quantity_tons)),np.log(float(thickness)),width,country,item_type_dict[item_type],application]
            with open('regression_modelfn.pkl', 'rb') as f:
                regg = pickle.load(f)

            pred = regg.predict([x])
            st.markdown(f"### :orange[Predicted Copper Selling Price is] :green[{round(np.exp(pred)[0],2)}]")
            st.info("Note: Price is predicted based on past data. In real time it can defer based on circumstances")

else:
    st.write("### :green[Copper winning Status prediction ]")
    col1,col2= st.columns([1,1],gap="large")
    
    with col1:
        customer = st.text_input(label='Customer ID (eg: 30161088, 30201846 etc..)')
        country = st.text_input("Country code(eg: 28,33 etc..)")
        item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
        item_type = st.selectbox(label='Item Type', options=item_type_values)
        item_type_dict = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}
        application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                        27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                        59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
        application = st.selectbox(label='Application value', options=application_values)
        material_ref_values = set(["DX51D+Z","DC01",'G9010','G9006','G9002','DEQ1 S460MC','S0380700','S0380700','DX51D+ZM310MAO 2.3X1317','2_S275JR+AR-CL1','NBW_L+_A_1125_0.4',
                               'NBW_L+_A_1125_0.4','NBW_L+_1125_0.4','NBW_L+_1125_0.4','NBW_L+_1125_0.4','NBW_L+_1125_0.4','DC04EK','DC04EK',
                               'DC04EK','PEA1265X595SP','PEA1265X595SP','PEA1265X595SP','684Z WHITE ETEX B7032','GRE1265X595SP','GRE1265X595SP'])
        material_ref = st.selectbox(label='Select any one of the material reference code', options=material_ref_values)
    with col2:
        width = st.number_input(label="Enter the Width of the Copper (Min: 0.01 & Max: infinity)",min_value=0.01)
        product_ref_values = [611993,164141591,640665,1670798778,628377,1668701718,
                              640405,1671863738,1332077137,1693867550,1668701376,1671876026,
                              628117,164337175,1668701698,1693867563,1282007633,1721130331,1665572374,628112]
        product_ref = st.selectbox(label='Select any one from the product reference code', options=product_ref_values)
        quantity_tons = st.text_input(label='Quantity Tons (Min: 0.00001 & Max: infinity)')
        selling_price = st.text_input(label='selling price (Min: 0.1 & Max: infinity)')
        thickness = st.number_input(label='Thickness', min_value=0.1, max_value=2500000.0, value=1.0)
    

    if ((country == "") | (quantity_tons == "") | (customer == "") | (selling_price == "")):
        st.warning("Please fill all the details")
    else:
        if st.button("Predict Copper Status"):
            mat = le.fit_transform([material_ref])
            y = [int(customer),int(country),item_type_dict[item_type],application,width,mat[0],product_ref,
                 np.log(float(quantity_tons)),np.log(float(selling_price)),np.log(float(thickness))]
            
            with open('classification_modelfn.pkl', 'rb') as f:
                class1 = pickle.load(f)
            class_pred = class1.predict([y])
            if class_pred[0] == 1:
                b = "Success"
                st.markdown(f"### :orange[Predicted Status of Copper winning is] :green[{b}]")
            else:
                b = "Failure"
                st.markdown(f"### :orange[Predicted Status of Copper winning is] :red[{b}]")

            st.info("Note: Status predicted in here is based on previous data Success/Failure ratio. It may vary based on real world factors")











