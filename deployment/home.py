import streamlit as st

def run():
    st.write('## Our Team :')
    st.write('##### :adult: [Gilang Wiradhyaksa](https://www.linkedin.com/in/gilangwiradhyaksa/) | [GitHub](https://github.com/gilangwd)')
    st.write('##### :adult: [Stephanus Adinata Susanto](https://www.linkedin.com/in/stephanus-adinata-susanto-1b115b170/) | [GitHub](https://github.com/StephanusAdinata)')
    st.write('##### :adult: [Samuel Christian Soendjojo](https://www.linkedin.com/in/samchriss94/) | [GitHub](https://github.com/samchriss94)')
    st.write('##### :adult: [Joshua Osaze Kurniawan](https://www.linkedin.com/in/joshua-osaze-kurniawan-45560228a/) | [GitHub](https://github.com/JoshuaOsazeKurniawan)')

    st.write('## Background :')
    st.markdown('''
                The telecommunications industry, being highly competitive, faces challenges in retaining customers. 
                Churn or customer attrition, is a critical metric that directly impacts the revenue and growth of a Telco company.
                ''')

    st.write('## Project Objective :')
    st.markdown('''
                Develop a machine learning model that can predicts customer churn in Telco Company. Then segment this churn potential customer into cluster 
                for retention strategies.
                ''')

if __name__ == '__main__':
    run()