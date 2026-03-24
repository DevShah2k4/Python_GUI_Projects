import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit_option_menu as som
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
#READING THE CSV FILE
customer = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#EDA
print(customer.isna().sum())
print("Summary of Data\n",customer.describe())
print("Information of Data\n",customer.info())
print("Random Data Sample\n",customer.sample())
print("Checking null Values\n",customer.isnull())


customer.replace(' ', np.nan, inplace=True)
customer["gender"] = customer["gender"].map({"Male":0,"Female":1})
columns_name = ["Partner","Dependents","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling"]
customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
print(customer["TotalCharges"].dtypes)
for col in columns_name:
     customer[col] = customer[col].map({"Yes":1,"No":0,2:"No Phone service"})

customer["InternetService"] = customer["InternetService"].map({"DSL":2,"Fiber optic":1,"No":0})
customer["Contract"] = customer["Contract"].map({"no internet service":0,"One year":1,"Two year":2,"Month-to-month":3})

#Converting to Yes and no into 1 and 0 respectively
customer["Churn"] = customer["Churn"].map({"No":0,"Yes":1})

#Visulization
sns.countplot(x = customer["gender"])
plt.title("Count of Male and Female")
plt.show()

sns.countplot(x = customer["Dependents"])
plt.title("Count of Dependents")
plt.show()

plt.hist(customer["MonthlyCharges"],bins=20)
plt.title("Histogram of Monthly Charges")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.show()

#plt.bar(customer["PaymentMethod"])


#Here x is an input features
x = customer[["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","MonthlyCharges","TotalCharges"]]
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(x)
print(customer.dtypes)
#Here y is an Output feature(Target Feature)
y = customer["Churn"]

#hre X is the imputed data all mean is done in null values 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#LogisticRegression
loreg = LogisticRegression(max_iter=100,random_state=42)
loreg.fit(x_train,y_train)
y_pred = loreg.predict(x_test)

print("Classification Report of Logistic Regression:-\n",classification_report(y_pred,y_test))
sns.heatmap(confusion_matrix(y_pred,y_test),cmap="coolwarm",annot=True)
plt.title("Confusion Matrix of Logistic Regression")
plt.show()

#DecisionTree Classfier
dtc = DecisionTreeClassifier(max_depth=4,min_samples_leaf=4,min_samples_split=2,random_state=42)
dtc.fit(x_train,y_train)
y_pred_dtc = dtc.predict(x_test)
print("Claasification Report of Decision Tree Classifier:\n",classification_report(y_pred_dtc,y_test))

sns.heatmap(confusion_matrix(y_pred_dtc,y_test),cmap="plasma",annot=True)
plt.title("Confusion Matrix of Decision Tree Classifier")
plt.show()

#Showing fetaure importance of Decisoon tree classifier 
plt.figure(figsize=(16,12))
features = dtc.feature_importances_
plt.bar(x.columns,features)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Random Forest Classfier
rfc = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42)
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
print("Classification Report of Random Forest Classifier:\n",classification_report(y_pred_rfc,y_test))

#Feature Importances
plt.figure(figsize=(18,12))
features_rfc = rfc.feature_importances_
plt.xticks(rotation=45)
plt.tight_layout()
plt.bar(x.columns,features_rfc)
plt.show()


sns.heatmap(confusion_matrix(y_pred_rfc,y_test),cmap="inferno",annot=True)
plt.title("Confusion Matrix of Random Forest Classifier")
plt.show()

#XGBoost
xgboost = XGBClassifier(n_estimators = 100,max_depth=4,learning_rate=0.3,randon_state=42)
xgboost.fit(x_train,y_train)
y_pred_xgboost = xgboost.predict(x_test)
print("Classfication Report of XGBoost \n",classification_report(y_pred_xgboost,y_test))
sns.heatmap(confusion_matrix(y_pred_xgboost,y_test),cmap="Blues",annot=True)
plt.title("Confusion Matrix of XGBoost")
plt.show()

model = pd.DataFrame({"Model Name":["Logisitic Regression","Decision Tree Classifier","Random Forest Classifier","XGB Classifier"],"Accuracy(%)":[round(accuracy_score(y_pred,y_test)*100,2),round(accuracy_score(y_pred_dtc,y_test)*100,2),round(accuracy_score(y_pred_rfc,y_test)*100,2),round(accuracy_score(y_pred_xgboost,y_test)*100,2)]})

with st.sidebar:
    selected_option = som.option_menu(menu_title="Menu",options=["Project Title","Dataset","Data Analysis or EDA","Graphs","Prediction","Model Accuracy","About Project"])
if selected_option=="Project Title":
    st.header("Customer Churn Prediction Using Streamlit")
if selected_option=="Dataset":
    st.title("Dataset of Customer Churn Prediction from Kaggle")
    st.dataframe(customer)
if selected_option=="Data Analysis or EDA":
    st.title("DataSet Summary")
    st.header("First 5 rows of Dataset")
    st.dataframe(customer.head(5))
    st.header("Last 5 rows of Dataset")
    st.dataframe(customer.tail(5))
    c_mean = customer["MonthlyCharges"].mean()
    st.header("Mean Monthly Charges")
    st.write(c_mean)
    st.header("Most Frequent Payment Method ")
    m_feq = customer["PaymentMethod"].mode()
    st.write(m_feq)
    st.header("Summary of Dataset")
    st.write(customer.describe())
if selected_option=="Graphs":
        st.title("Graphs of Customer Churn Prediction")
        
        st.subheader("Gender Distribution")
        st.bar_chart(customer["Churn"].value_counts())
        #st.image("Count of Male and Female.png")
        st.subheader("Dependents Distribution")
        st.bar_chart(customer["Dependents"].value_counts())
        
        st.subheader("Churn Distribution")
        st.bar_chart(customer["Churn"].value_counts())

        st.subheader("Payment Method Distribution")
        st.bar_chart(customer["PaymentMethod"].value_counts())

        st.subheader("Contracts Distribution")
        st.bar_chart(customer["Contract"].value_counts())

        st.subheader("Monthly Charges Distribution")
        st.image("Histogram of Monthly Charges.png")
if selected_option=="Prediction":
    st.title("Taking Input Values from User")
    gender = st.number_input("Enter 0(Male) or 1(Female):-",min_value=0,max_value=1)
    senior_citizen = st.number_input("Enter 0(NoSeniorCitizen) or 1(SeniorCitizen):-",min_value=0,max_value=1)
    partner = st.number_input("Enter 0(No Partner) or 1(Yes Partner):-",min_value=0,max_value=1)
    dependent = st.number_input("Enter 0(No Dependent) or 1(Yes Dependent):-",min_value=0,max_value=1)
    tenure = st.number_input("Enter Your Tenure(in Years):-",min_value=1,max_value=80)
    phone_service = st.number_input("Enter 0(No Service) or 1(Yes Service) or 2(No Phone Service):-",min_value=0,max_value=1)
    multiple_service = st.number_input("Enter 0(No Multiple Service) or 1(Yes Multiple Service):-",min_value=0,max_value=1)
    internet_service = st.number_input("Enter 0(No Internet Service) or 1(Fibre Optic) or 2(DSL):-",min_value=0,max_value=2)
    online_security = st.number_input("Enter 0(No Security Service) or 1(Yes Security Service):-",min_value=0,max_value=1)
    online_backup = st.number_input("Enter 0(No Online Backup) or 1(Yes Online Backup):-",min_value=0,max_value=1)
    device_protection = st.number_input("Enter 0(No Device Protection) or 1(yes Device Protection):-",min_value=0,max_value=1)
    tech_support = st.number_input("Enter 0(No Tech Support) or 1(Yes Tech Support):-",min_value=0,max_value=1)
    streaming_tv = st.number_input("Enter 0(No Streaming TV) or 1(Yes Streaming TV):-",min_value=0,max_value=1)
    streaming_movies = st.number_input("Enter 0(No Streaming Movies) or 1(Yes Streaming Movies):-",min_value=0,max_value=1)
    contract = st.number_input("Enter 0(No Internet Service) or 1(One Year) or 2(Two Year) or 3(Month-to-Month):-",min_value=0,max_value=3)
    paper_less_billing = st.number_input("Enter 0(No Papaer_less_Billing) or 1(Yes Paper_less_Billing):-",min_value=0,max_value=1)
    monthly_charges = st.number_input("Enter Monthly Charges:-",min_value=1.00)
    total_charges = st.number_input("Enter Total Charges:-",min_value=1) 
    if st.button("Predict"):
            predicted_output = loreg.predict([[gender,senior_citizen,partner,dependent,tenure,phone_service,multiple_service,internet_service,online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies,contract,paper_less_billing,monthly_charges,total_charges]])
         
            if predicted_output==1:
                st.write("Customer will leave the Company")
            else:
                st.write("Customer will not leave the Company")
if selected_option=="Model Accuracy":

     st.header("Model Accuracy Comparison")     
     st.bar_chart(data=model,x="Model Name",y="Accuracy(%)")

     st.header("Logistic Regression")
     st.subheader("Classification Report of Logistic Regression")
     report = classification_report(y_pred,y_test,output_dict=True)
     final_report = pd.DataFrame(report).transpose()
     st.dataframe(final_report)
     st.subheader("Accuracy Score of Logisitic Regression")
     st.write(round(accuracy_score(y_pred,y_test)*100,2),"%")
     st.subheader("Confusion Matrix of Logistic Regression")
     st.image("Confusion matrix of Logistic Regression.png")
    
     st.header("Decision Tree Classifier")
     st.subheader("Classification Report of Decision Tree Classifier")
     report_tree = classification_report(y_pred_dtc,y_test,output_dict=True)
     final_report_tree = pd.DataFrame(report_tree).transpose()
     st.dataframe(final_report_tree)
     st.subheader("Accuracy Score of Decision Tree Classifier")
     st.write(round(accuracy_score(y_pred_dtc,y_test)*100,2),"%")
     st.subheader("Confusion Matrix of Decision Tree Classifier")
     st.image("Confusion Matrix of Decision Tree Classifier.png")

     st.header("Random Forest Classifier")
     st.subheader("Classification Report of Random Forest Classifier")
     report_forest = classification_report(y_pred_rfc,y_test,output_dict=True)
     final_report_forest = pd.DataFrame(report_forest).transpose()
     st.dataframe(final_report_forest)
     st.subheader("Accuracy Score of Decision Tree Classifier")
     st.write(round(accuracy_score(y_pred_rfc,y_test)*100,2),"%")
     st.subheader("Confusion Matrix of Random Forest Classifier")
     st.image("Confusion Matrix of Random Forest Classifier.png")

     st.header("XGB Classifier")
     st.subheader("Classification Report of XGB Classifier")
     report_boost = classification_report(y_pred_xgboost,y_test,output_dict=True)
     final_report_boost = pd.DataFrame(report_boost).transpose()
     st.dataframe(final_report_boost)
     st.subheader("Accuracy Score of XGB Classifier")
     st.write(round(accuracy_score(y_pred_xgboost,y_test)*100,2),"%")
     st.subheader("Confusion Matrix of Random Forest Classifier")
     st.image("Confusion Matrix of XGB Classifier.png")

     st.header("Feature Importance of Decision Tree Classifier")
     st.image("Feature Importance of Decision Tree Classfier.png")

     
     st.header("Feature Importance of Random Forest Classifier")
     st.image("Feature Importance of Random Forest Classifier.png")
if selected_option=="About Project":
     st.title("About Customer Churn Prediction")
     st.write("Customer churn prediction means finding out which customers may stop using a company’s service in the future. When a customer leaves a service, it is called churn. For example, if a person stops using a mobile network, closes a bank account, or cancels an app subscription, that customer has churned. Customer churn prediction looks at customer behavior such as how often they use the service, how much they pay, or whether they have problems with the service. By understanding these patterns, companies can guess who might leave and try to keep those customers by improving service, giving offers, or providing better support.")
     st.title("About Project of Customer Churn Prediction")
     st.write("In this project, customer data such as tenure, contract type, services used, and billing details are analyzed to understand customer behavior. The dataset includes a target column called Churn, where 0 represents customers who stay and 1 represents customers who leave. Four machine learning models were implemented and evaluated: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and XGBoost Classifier. After comparing their performance, Logistic Regression achieved the highest accuracy of 81%, making it the most suitable model for this project. XGBoost achieved an accuracy of 80%, followed by Random Forest with 79% and Decision Tree with 79%.")
     st.write("Logistic Regression was selected because it provides better accuracy, faster execution, and clear interpretability for binary classification problems like churn prediction. The final model predicts whether a customer will churn or not based on input features, allowing businesses to identify high-risk customers and improve customer retention strategies.")