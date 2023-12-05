import streamlit as st
from streamlit_option_menu import option_menu 
import home
import eda
import prediction
from PIL import Image

# navigation = st.sidebar.selectbox('Select Page :', ('EDA', 'Predict Credit Card Default'))

# if navigation == 'EDA':
#     eda.runEDA()
# else:
#     prediction.runPredictor()

# Set page title and icon
# st.set_page_config(page_title='Final Project', page_icon='⭐')

# Create sidebar navigation

# st.markdown(
#     f"""
#     <style>
#         [data-testid="stSidebar"] {{
#             background-image: url(https://raw.githubusercontent.com/FTDS-assignment-bay/main/assets/ChurnGuardian-Logo-Transparants.png);
#             background-repeat: no-repeat;
#             padding-top: 20px;
#             background-position: 10px 50px;
#             background-size: 310px 85px;
#         }}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

st.set_page_config(
    page_title='Telco Customer Churn and Segmentation', 
    layout='centered', #wide
    initial_sidebar_state='expanded'
)

# st.title('Telco Customer Churn and Segmentation')
# image = Image.open('images\logo_crop_clean.png')
# image = Image.open('images\logo_grey_clean.png')
col1, col2, col3 = st.columns([10, 1, 5])
image_url = 'https://raw.githubusercontent.com/FTDS-assignment-bay/p2-final-project-ftds-001-sby-group-001/main/images/logo_crop_clean.png'
# qr_url = 'https://raw.githubusercontent.com/FTDS-assignment-bay/p2-final-project-ftds-001-sby-group-001/main/images/qr_link.png'
col1.image(image_url, width=450)
# col2.write('')
# col3.image(qr_url, width=150)
st.write('# Customer Churn and Segmentation')
st.subheader('Predict churn and retain your customer!')
st.markdown('---')

selected = option_menu(None, ["About", "EDA", "Predict"], 
    icons=['house', 'file-earmark-bar-graph', 'search'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={
        "icon": {"color": "orange", "font-size": "15px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"1px", "--hover-color": "#eee"}, 
        "nav-link-selected": {"background-color": "grey"},
    }
)   

if selected == 'About':
    home.run()
elif selected == 'EDA':
    eda.run()
else:
    prediction.run()
#streamlit run app.py