import numpy as np
import pandas as pd
import streamlit as st

# Sayfa Ayarları
st.set_page_config(
    page_title="Churn Analysis",
    page_icon="/Users/Lenovo/Desktop/ISTDSA/DSAG22/proje3/streamlit/CS_logo.png",
    menu_items={
        "Get help": "mailto:juneight79@gmail.com",
        
    }
)

# Başlık Ekleme
st.title("Churn Analysis Project")

# Markdown Oluşturma
st.markdown("A Machine Learning Model That Can Predict Customers Who Will Leave The Company.")
st.markdown("The aim is to predict whether a bank's customers leave the bank or not. If the Client has closed his/her bank account, he/she has left.")

# Resim Ekleme
st.image("/Users/Lenovo/Desktop/ISTDSA/DSAG22/proje3/streamlit/churn_image.jpg")

st.markdown("Finding a new customer is much more difficult and costly than retaining an existing customer.")
st.markdown("For this reason, it is very important to anticipate customers that we are likely to lose.")
st.markdown("This point shows the necessity of Churn Analysis expected from us.")


# Header Ekleme
st.header("Data Dictionary")

st.markdown("- **Exited**: Whether or not the customer left the bank? (0 = Not_Churn, 1 = Churn)")
st.markdown("- **Creditscore**: Score showing the regularity of loan payments.")
st.markdown("- **Age**: The age of customer.")
st.markdown("- **Tenure**: Refers to the number of years that the customer has been a client of the bank.")
st.markdown("- **Balance**: Account balance of customer.")
st.markdown("- **Num Of Products**: Refers to the number of products that a customer has purchased through the bank.")
st.markdown("- **Estimated Salary**: Estimated salary off customer.")
st.markdown("- **Balance/Income Ratio**: (Balance/Estimated Salary)- It shows how much of the income is a bank account balance. ")
st.markdown("- **Card Product Ratio**: (hascrcard/NumOfProduct)-The ratio of credit card ownership to the number of products used. If the rate is 100, there is no product usage other than the card.")
st.markdown("- **Geografy**: Customer Location (There are 3 locations in the data set, namely France, Spain and Germany, and after the one hot encoding process, an analysis was made over Germany and Spain.)")
st.markdown("- **Gender**: The Gencer of customer. 0= Female ,  1=Male")
st.markdown("- **Is Active Member**: Refers to Whether or not the aktif customer of the bank. 1=Actice Customer")


# Pandasla veri setini okuyalım
df = pd.read_csv("/Users/Lenovo/desktop/ISTDSA/DSAG22/proje3/streamlit/ChurnData.csv")


# Tablo Ekleme
st.table(df.sample(10, random_state=42))



#---------------------------------------------------------------------------------------------------------------------

# Sidebarda Markdown Oluşturma
st.sidebar.markdown("**Choose** the features below to see the result!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.sidebar.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.sidebar.text_input("Surname", help="Please capitalize the first letter of your surname!")
creditscore = st.sidebar.number_input("Credit Score", min_value=1, format="%d")
age = st.sidebar.number_input("Age", min_value=18 , format="%d")
tenure = st.sidebar.number_input("Tenure", min_value=0 , format="%d")
balance = st.sidebar.number_input("Balance", min_value=0 , format="%d")
numofproducts=st.sidebar.number_input("Number Of Products", min_value=1 , format="%d")
estimatedsalary=st.sidebar.number_input("Estimated Salary", min_value=1 , format="%d")
geography_Germany = st.sidebar.number_input("Is Geografy Germany?", min_value=0 , max_value=1, format="%d", help= "If Geografy is Germany ,please select 1, not 0!")
geography_Spain = st.sidebar.number_input("Is Geografy Spain?", min_value=0 , max_value=1, format="%d", help= "If Geografy is Spain ,please select 1, not 0!")
gender = st.sidebar.number_input("Is Male?", min_value=0 , max_value=1, format="%d", help= "If Gender is Male ,please select 1, not 0!")

hascrcard_1=st.sidebar.number_input("Has Credit Card? ", min_value=0 , max_value=1, format="%d", help= "If has, please select 1, not 0!") 

ısactivemember = st.sidebar.number_input("Is Active? ", min_value=0 , max_value=1, format="%d", help= "If is actice please select 1, not 0!") 
IncomeBalanceRatio = balance/estimatedsalary
CardProductRatio = hascrcard_1/numofproducts


#---------------------------------------------------------------------------------------------------------------------

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
from joblib import load

logreg_model = load('/Users/Lenovo/Desktop/ISTDSA/DSAG22/proje3/streamlit/logreg1_model.pkl')

input_df = pd.DataFrame({

     
    'Credit Score': [creditscore],
    'Age' :[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'Number Of Products':[numofproducts],
    'Estimated Salary':[estimatedsalary],
    'Geografy is Germany?':[geography_Germany],
    'Geografy is Spain?':[geography_Spain],
    'Gender is Mail ?':[gender],
    'Has Card ?':[hascrcard_1],
    'Is Actice?':[ısactivemember],
    'IncomeBalanceRatio':[IncomeBalanceRatio],
    'CardProductRatio':[CardProductRatio]
})

pred = logreg_model.predict(input_df.values)
pred_probability = np.round(logreg_model.predict_proba(input_df.values), 2)

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    # Sorgulama zamanına ilişkin bilgileri elde etme
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
    'Name': [name],
    'Surname': [surname],
    'Date': [today],
    'Time': [time],
    'Age' :[age],
    'Balance':[balance],    
    'Credit Score': [creditscore],
    'Tenure':[tenure],
    'Number Of Products':[numofproducts],
    'Estimated Salary':[estimatedsalary],
    'Geografy is Germany?':[geography_Germany],
    'Geografy is Spain?':[geography_Spain],
    'Gender is Mail ?':[gender],
    'Prediction': [pred],
    'Not_Churn Probability': [pred_probability[:,:1]],
    'Churn Probability': [pred_probability[:,1:]]
    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Not_Churn"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Churn"))

    st.table(results_df)

    if pred == 0:
        st.image("/Users/Lenovo/Desktop/ISTDSA/DSAG22/proje3/streamlit/mutlu_emoji.gif")
    else:
        st.image("/Users/Lenovo/Desktop/ISTDSA/DSAG22/proje3/streamlit/üzgün.gif")
else:
    st.markdown("Please click the *Submit Button*!")