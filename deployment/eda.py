import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def run():
    # Show dataframe
    # st.title('Data Overview')
    df = pd.read_csv('telco_data_clean.csv')
    # st.dataframe(df.head())

    st.title('Exploratory Data Analysis')
    plot_selection = st.selectbox(label='Choose', 
                                  options=['Customer Demographic', 
                                           'Churn by Monthly Charge',
                                           'Churn by Tenure', 
                                           'Churn by Internet Service'])
    
    # Plot 1
    def plot_1():
        st.write('#### Pie Chart for Customer Status Distribution')
        target = df.groupby(['churn']).agg(total_churn=('churn', 'count'))
        gender = df.groupby(['gender']).agg(total_gender=('gender', 'count'))
        target['percentage'] = (target['total_churn'] / target['total_churn'].sum())

        fig_1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].pie(target["total_churn"], labels=target["total_churn"].index, autopct='%.0f%%')
        ax[0].set_title("Churn and No-Churn")
        ax[1].pie(gender["total_gender"], labels=gender["total_gender"].index, autopct='%.1f%%')
        ax[1].set_title("Customer Gender")

        st.pyplot(fig_1)
        st.write('''
                From the plot above, it is found that of the total number of customers who churn 
                is 27% (1869 customers) and customers who is not churn / stay is 73% (5163 customers).
                From all the customer 50.5% is male and 49.5% is female.
                ''')
        st.markdown('---')
    
    # Plot 2
    def plot_2():
        df_churn_by_mcharges = df.groupby(['monthly_charges_cat']).agg(total=('monthly_charges_cat', 'count')).sort_values(by=['total'], ascending=True)
        fig_2 = plt.figure(figsize=(7, 5))
        ax = sns.barplot(data=df_churn_by_mcharges, x=df_churn_by_mcharges.index.to_list(), y='total', orient='v')
        ax.bar_label(ax.containers[0])
        ax.set(title='Churn by Monthly Charges')
        st.pyplot(fig_2)
        st.write('''
                 From the bar plot above we can see that Medium-High Expenses and High Expense 
                 have the highest churn rate.
                 ''')
        st.markdown('---')

    # Plot 3
    def plot_3():
        df_churn_by_mcharge = df.groupby(['tenure_year', 'churn']).agg(total=('churn', 'count'))
        fig_3 = plt.figure(figsize=(7, 5))
        ax = sns.lineplot(data=df_churn_by_mcharge, x="tenure_year", y="total", hue="churn")
        # ax.bar_label(ax.containers[0])
        ax.set(title='Churn by Monthly Charges')
        st.pyplot(fig_3)
        st.write('''
                 From the bar plot above we can see that as tenure increases, the churn rate tends 
                 to decrease. Customers with a longer usage period tend to be more loyal.
                 ''')
        st.markdown('---')
    
    # Plot 4
    def plot_4():
        fig_4 = plt.figure(figsize=(7, 5))
        ax = sns.countplot(data=df, x='internet_service', hue='churn')
        ax.bar_label(ax.containers[0])
        ax.bar_label(ax.containers[1])
        ax.set(title='Churn by Internet Service')
        st.pyplot(fig_4)
        st.write('There are more customers with Fiber Optic services who churn (1297) than DSL customers who churn (459). The ratio of churn to total customers appears to be higher for customers with Fiber Optic services (41.9%) compared to DSL customers (19.0%).')
        st.write('The solution that needs to be done is to evaluate and update Fiber Optic services to improve quality and customer satisfaction, such as focusing on improving speed, stability and ease of use. In addition, there needs to be an adjustment to the marketing strategy to emphasize the advantages and benefits of Fiber Optic services that can meet customer needs by identifying market segments that are more likely to be interested in this service.')
        st.markdown('---')


    if plot_selection == "Customer Demographic":
        plot_1()
    elif plot_selection == "Churn by Monthly Charge":
        plot_2()
    elif plot_selection == "Churn by Tenure":
        plot_3()
    elif plot_selection == "Churn by Internet Service":
        plot_4()

if __name__ == '__main__':
    run()