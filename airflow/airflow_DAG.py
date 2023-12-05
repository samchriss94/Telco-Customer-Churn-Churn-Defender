import pandas as pd
import psycopg2 as db
import numpy as np

import datetime as dt
from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

def getData():
    conn_string = "dbname='airflow' host='postgres' user='airflow' password='airflow'"
    conn = db.connect(conn_string)

    SQL = '''SELECT * FROM table_finpro_telco'''

    df = pd.read_sql(SQL, conn)

    # Save Raw Data from SQL
    df.to_csv('/opt/airflow/data/telco_data_raw.csv', index=False)

def cleaningData():
    # Load File
    df = pd.read_csv('/opt/airflow/data/telco_data_raw.csv', index_col=False)

    # Remove Duplicate
    if df.duplicated().sum() > 0: df.drop_duplicates(inplace=True)

    # Remove Missing Value
    if df.isnull().sum().sum() > 0: df.dropna(inplace=True)

    # Rename Header
    df.rename(columns={'customerID':'customer_id',  'SeniorCitizen':'senior_citizen',  'PhoneService':'phone_service', 'MultipleLines':'multiple_lines',  
                       'InternetService':'internet_service', 'OnlineSecurity':'online_security',  'OnlineBackup':'online_backup', 'DeviceProtection':'device_protection', 
                       'TechSupport':'tech_support', 'StreamingTV':'streaming_tv',  'StreamingMovies':'streaming_movies', 'PaperlessBilling':'paperless_billing',  
                       'PaymentMethod':'payment_method', 'MonthlyCharges':'monthly_charges',  'TotalCharges':'total_charges'}, inplace=True)
    df.columns = df.columns.str.lower()

    # Remove Column
    df.drop(['customer_id'], axis = 1, inplace=True)
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)

    # Data Type Conversion
    df['total_charges'] = df['total_charges'].astype(float)

    # Convert to Yes No
    df["senior_citizen"]= df["senior_citizen"].map({0: "No", 1: "Yes"})

    #Function to define cholesterol level
    def monthlyChargesCat(row):
        if row['monthly_charges'] <= 30:
            return 'Low Expense'
        elif row['monthly_charges'] <= 60:
            return 'Medium Expense'
        elif row['monthly_charges'] <= 90:
            return 'Medium High Expense'
        else:
            return 'High Expense'
    
    def tenureCat(row):
        if row['tenure'] <= 12:
            return 1
        elif row['tenure'] <= 24:
            return 2
        elif row['tenure'] <= 36:
            return 3
        elif row['tenure'] <= 48:
            return 4
        elif row['tenure'] <= 60:
            return 5
        else:
            return 6
    
    # Column Creation
    df['monthly_charges_cat'] = df.apply(lambda row: monthlyChargesCat(row), axis=1)
    df['tenure_year'] = df.apply(lambda row: tenureCat(row), axis=1)

    # Save Clean Data
    df.to_csv('/opt/airflow/data/telco_data_clean.csv', index=True)

default_args = {
    'owner': 'gilang',
    'start_date': dt.datetime(2023, 11, 20, 13, 30, 0) - dt.timedelta(hours=7),
}

with DAG('churn_defender_v1_0', default_args=default_args, 
            schedule_interval='0 0 1 * *',
        ) as dag:

    # Task 1
    fetchFromPostgreSQL = PythonOperator(task_id='fetchFromPostgreSQL', python_callable=getData)

    # Task 2
    cleaningDataTask = PythonOperator(task_id='cleaningDataTask', python_callable=cleaningData)

fetchFromPostgreSQL >> cleaningDataTask