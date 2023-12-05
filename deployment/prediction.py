import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

def run():
    # Load Model Classification
    with open('adaboost_logreg_best.pkl', 'rb') as file_1:
    # with open('adaboost_logreg_10_features.pkl', 'rb') as file_1:
        classification_model = pickle.load(file_1)
    
    # Load Model Clustering
    with open('kp.pkl','rb') as file_2:
        clustering_model = pickle.load(file_2)

    # Load Clustering Scaler
    with open('scaler.pkl','rb') as file_3:
        scaler = pickle.load(file_3)
    
    # Load Clustering Numerical
    with open('num_col.txt','rb') as file_4:
        num_col = pickle.load(file_4)

    # Load Clustering Categorical
    with open('cat_col.txt','rb') as file_5:
        cat_col = pickle.load(file_5)

    # Choice of input: Upload or Manual Input
    inputType = st.selectbox("How would you like to input data ?", ["Upload Excel or CSV File", "Manual Input"])
    st.markdown('---')

    # Create Function for Prediction
    def predictData(df):
        totalCustomer = len(df)

        if totalCustomer < 1:
            st.write('## There is no Customer on this data, please check again.')
        else:
            # Classification prediction
            y_pred_uploaded = classification_model.predict(df)
            df['churn'] = y_pred_uploaded
            # st.dataframe(df)

            # Filter the DataFrame for Predicted Churn (1) 
            df_churn = df[df['churn'] == 1]
            
            churnCustomer = len(df_churn)

            if churnCustomer == 0:
                st.write('## There is no Customer predicted as Churn from this Data!')
            else:
                # Clustering prediction for Predicted Churn (1)
                ## Split Numerical and Categorical for K-Prototype
                data_cluster_num = df_churn[num_col]
                data_cluster_cat = df_churn[cat_col]

                ## Scale Numerical column
                num_scaled = scaler.transform(data_cluster_num)

                ## Merge Scaled Numerical + Categorical
                data_cluster_final = np.concatenate([num_scaled, data_cluster_cat], axis=1)
                data_cluster_final = pd.DataFrame(data_cluster_final, columns=['tenure', 'monthly_charges'] + cat_col)
                data_cluster_final = data_cluster_final.infer_objects()

                ## Mark Categorical Column
                index_cat_columns = [data_cluster_final.columns.get_loc(col) for col in cat_col] 

                ## Predict Cluster
                y_cluster = clustering_model.predict(data_cluster_final, categorical=index_cat_columns)
                # y_cluster = []
                #for rd in range(0, len(df_churn)): y_cluster.append(random.randint(0, 2)) # Random Generator for testing
                df_churn['cluster'] = y_cluster

                temp_cols = df_churn.columns.tolist()
                new_cols = temp_cols[0:1] + temp_cols[-2:] + temp_cols[1:-2]
                df_churn = df_churn[new_cols]
                df_churn = df_churn.sort_values(by=['cluster'], ascending=True)

                # Saving Result to Excel
                df_churn.to_excel('model_result.xlsx', index=False)
                
                # Split Data into 3 Cluster DataFrames
                df_cluster_0 = df_churn[df_churn['cluster'] == 0]
                df_cluster_1 = df_churn[df_churn['cluster'] == 1]
                df_cluster_2 = df_churn[df_churn['cluster'] == 2]

                st.write(f'## Result : `{churnCustomer} customer` from total {totalCustomer} customer ({int((churnCustomer/totalCustomer)*100)}%) are predicted as churn!')
                st.write('##### Here are some suggestion to minimalize churn potential for each customer depend on their cluster')
                c0, c1, c2 = '', '', ''
                for x in df_cluster_0['name']: c0 += str(x) + ', '
                for y in df_cluster_1['name']: c1 += str(y) + ', '
                for z in df_cluster_2['name']: c2 += str(z) + ', '
                
                cluster_0 = '''
                    - Most of them are senior citizen
                    - Having partner and dependents
                    - High monthly charges
                '''

                # suggestion_0 = '''
                #     - Offers packages with additional speed for 3 months for those who have subscribed for more than 3 years
                #     - Open all TV channels during big holiday events such as Eid, Christmas and others
                #     - Provide special offers to increase internet speed to them
                # '''
                suggestion_0 = '''
                    - Offers long term packages
                    - Give limited time offer
                    - Maintain good communication with this customer
                '''

                cluster_1 = '''
                    - Mix of senior citizan and youngster
                    - Having partner and dependents
                    - Low monthly charges
                '''

                suggestion_1 = '''
                    - Provides offers with many benefits if they subscribe for the long term
                    - Offers annual DSL internet packages at affordable prices
                    - New customer onboarding and orientation
                '''

                cluster_2 = '''
                    - Most of them are young people
                    - Most of them have no partner and dependents
                    - Moderate monthly charges
                '''

                # suggestion_2 = '''
                #     Providing special packages with the following criteria:
                #     - High speed internet but lower bandwidth at a cheaper price than normal packages
                #     - Low speed internet but large bandwidth so the connection is much more stable at a cheaper price than normal packages
                # '''
                suggestion_2 = '''
                    - Make an affordable internet package prices for this cluster
                    - Provides variation in Payment Method
                '''

                if c0 != '':
                    st.write(f'##### Cluster 1 - Elder Group - {len(df_cluster_0)} customer ({((len(df_cluster_0)/churnCustomer)*100):.1f}%)')
                    st.write(cluster_0)
                    st.write('Suggestion for `', c0[0:-2], '` is')
                    st.write(suggestion_0)
                    st.markdown('---')
                
                if c1 != '':
                    st.write(f'##### Cluster 2 - Mixage - {len(df_cluster_1)} customer ({((len(df_cluster_1)/churnCustomer)*100):.1f}%)')
                    st.write(cluster_1)
                    st.write('Suggestion for `', c1[0:-2], '` is')
                    st.write(suggestion_1)
                    st.markdown('---')
                
                if c2 != '':
                    st.write(f'##### Cluster 3 - Young Blood - {len(df_cluster_2)} customer ({((len(df_cluster_2)/churnCustomer)*100):.1f}%)')
                    st.write(cluster_2)
                    st.write('Suggestion for `', c2[0:-2], '` is')
                    st.write(suggestion_2)
                    st.markdown('---')
                
                # Create Bar Plot for Analyze Cluster
                num_agg_df = df_churn.groupby(['cluster']).agg({'tenure': 'mean', 'monthly_charges': 'mean'})
                num_agg_df = np.round(num_agg_df, decimals=2)
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))

                # Loop through each subplot to populate it
                for ax, column in zip(axes, num_agg_df.columns):
                    sns.barplot(ax=ax, data=num_agg_df, x=num_agg_df.index, y=column, orient='v')
                    ax.set_title(f'Average {column} by Cluster')
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel(f'Average {column}')
                    ax.bar_label(ax.containers[0])
                
                # plt.style.use('dark_background')
                plt.tight_layout()
                st.pyplot(fig)
                # plt.style.use('default')

                with open('model_result.xlsx', 'rb') as file:
                    st.download_button(
                        label='💾 Download Prediction Result',
                        data=file,
                        file_name='model_result.xlsx',
                        mime='application/vnd.ms-excel'
                )
    
    def tenureMonthToYear():
        year = st.session_state.tenurem % 12
        if year == 0:
            st.session_state.tenurey = int((st.session_state.tenurem / 12))
        else:
            st.session_state.tenurey = int((st.session_state.tenurem / 12) + 1)
    
    def calculateChargesAndCategory():
        st.session_state.tcharges = int((st.session_state.mcharges * st.session_state.tenurem))
        if st.session_state.mcharges <= 30:
            st.session_state.catcharges = 'Low Expense'
        elif st.session_state.mcharges <= 60:
            st.session_state.catcharges = 'Medium Expense'
        elif st.session_state.mcharges <= 90:
            st.session_state.catcharges = 'Medium High Expense'
        else:
            st.session_state.catcharges = 'High Expense'

    # A. For CSV
    if inputType == "Upload Excel or CSV File":
        dl_1, dl_2, dl_3 = st.columns([3, 3, 3])
        with open('telco_data_test.xlsx', 'rb') as file:
            dl_1.download_button(
                label='💾 Download Data Example',
                data=file,
                file_name='telco_example.xlsx',
                mime='application/vnd.ms-excel'
            )
        
        with open('telco_data_template.xlsx', 'rb') as file:
            dl_2.download_button(
                label='💾 Download Template Excel',
                data=file,
                file_name='telco_template.xlsx',
                mime='application/vnd.ms-excel'
            )

        uploaded_file = st.file_uploader("Choose Excel or CSV file", type=["csv", "xlsx"], accept_multiple_files=False)
        if uploaded_file is not None:
            split_file_name = os.path.splitext(uploaded_file.name)
            # file_name = split_file_name[0]
            file_extension = split_file_name[1]

            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            else:    
                df = pd.read_excel(uploaded_file)
            # st.dataframe(df.head())
            predictData(df)
    # B. For Manual        
    else:
        # Create Form
        # with st.form(key='Form Parameters'):
        name = st.text_input('Name', value='', help='Customer Name')

        col_left, col_mid, col_right = st.columns([3, 2, 2])
        gender =  col_left.selectbox('Gender', ('Male', 'Female'), index=0)
        with col_mid:
            tenure =  st.number_input('Tenure (Month)', min_value=1, max_value=999, step=1, help='Month', key='tenurem', on_change=tenureMonthToYear)
        with col_right:
            tenure_year = st.number_input('Tenure (Year)', min_value=1, max_value=999, step=1, disabled=True, key='tenurey')
        
        col1, col2, col3 = st.columns([1, 1, 1])
        senior_citizen = col1.radio(label='Senior Citizen?', options=['Yes', 'No'], help='Choose \'Yes\' for 61 years old above')
        partner = col2.radio(label='Having a partner?', options=['Yes', 'No'])
        dependents = col3.radio(label='Having a dependents?', options=['Yes', 'No'], help='For example : children')
        
        # col4, col5 = st.columns([1, 1])
        # internet_service =  col4.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'), index=0)

        col4, col5, col6 = st.columns([1, 1, 1])
        internet_service =  col4.radio(label='Subs for Internet service?', options=['DSL', 'Fiber optic', 'No'])
        phone_service = col5.radio(label='Subs for Phone service?', options=['Yes', 'No'])
        multiple_lines = col6.radio(label='Subs for Multiple Lines?', options=['Yes', 'No', 'No Phone Services'])

        col7, col8, col9 = st.columns([1, 1, 1])
        online_security = col7.radio(label='Subs for Online Security?', options=['Yes', 'No', 'No Internet Services'])
        online_backup = col8.radio(label='Subs for Online Backup?', options=['Yes', 'No', 'No Internet Services'])
        device_protection = col9.radio(label='Having Device Protections?', options=['Yes', 'No', 'No Internet Services'])
        tech_support = col7.radio(label='Having Tech Support service?', options=['Yes', 'No', 'No Internet Services'])
        streaming_tv = col8.radio(label='Subs for TV Streaming?', options=['Yes', 'No', 'No Internet Services'])
        streaming_movies = col9.radio(label='Subs for Movie Streaming?', options=['Yes', 'No', 'No Internet Services'])

        col_pm1, col_pm2, col_pm3 = st.columns([3, 3, 2])
        contract = col_pm1.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'), index=0)
        payment_method = col_pm2.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), index=0)
        paperless_billing = col_pm3.selectbox('Paperless billing?', ('Yes', 'No'), index=0)

        col_charges1, col_charges2, col_charges3 = st.columns([1, 1, 2])
        monthly_charges =  col_charges1.number_input('Monthly Charges', min_value=1, max_value=999, step=1, help='Amount to paid per month', key='mcharges', on_change=calculateChargesAndCategory)
        total_charges = col_charges2.number_input('Total Charges', min_value=1, max_value=999999, step=1, disabled=True, key='tcharges')
        charges_cat = col_charges3.text_input('Charges Category', disabled=True, key='catcharges')

        # st.button('Predict', on_click=predict)
        data_inf = {
            'name': name, 
            'gender': gender, 
            'senior_citizen': senior_citizen, 
            'partner': partner, 
            'dependents': dependents, 
            'tenure': tenure, 
            'phone_service': phone_service, 
            'multiple_lines': multiple_lines, 
            'internet_service': internet_service, 
            'online_security': online_security, 
            'online_backup': online_backup, 
            'device_protection': device_protection, 
            'tech_support': tech_support, 
            'streaming_tv': streaming_tv, 
            'streaming_movies': streaming_movies, 
            'contract': contract, 
            'paperless_billing': paperless_billing, 
            'payment_method': payment_method, 
            'monthly_charges': monthly_charges, 
            'total_charges': total_charges, 
            'monthly_charges_cat': charges_cat, 
            'tenure_year': tenure_year
            }

        if st.button('Predict'):
            data_inf = pd.DataFrame([data_inf])
            # st.dataframe(data_inf.head())
            predictData(data_inf)

if __name__ == '__main__':
    run()