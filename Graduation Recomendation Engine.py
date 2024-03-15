#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from imblearn.over_sampling import SMOTE


#Page configurations
st.set_page_config(layout='wide',page_title='Graduate Data Analysis',page_icon='🎓')

#Head and intro
st.title('Recommendation Engine')
st.divider()
st.write("Our application dives deep into the trends and patterns prevalent among graduates, offering valuable insights into their academic and career journeys. With a user-friendly interface, our app allows you to explore various aspects of graduate data, from academic performance to placement rates. But that's not all – we've integrated a recommendation engine that provides personalized insights based on the details you input. Whether you're a student, educator, or industry professional, our app empowers you to make informed decisions and navigate the dynamic landscape of higher education and employment.")

#reading file
data = pd.read_csv("Data/Dataset_Statathon.csv")
model_data = data.drop('Unnamed: 0',axis=1).copy()
stream_dum = pd.get_dummies(data['Stream'])
rank_dum = pd.get_dummies(data['Rank_Group'])
model_data = pd.concat([model_data,stream_dum],axis=1)
model_data = pd.concat([model_data,rank_dum],axis=1)
model_data.drop(['College_Rank','College_Name','Rank_Group','Stream'],axis=1,inplace=True)

#prepping and training model
features = model_data.drop(['Salary','Placement_Status'],axis=1)
target = model_data['Placement_Status']
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=.40,random_state=40)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_rfc = RandomForestClassifier(n_estimators=1000)
model_rfc.fit(X_resampled,y_resampled)
y_pred_rfc = model_rfc.predict(X_test)

#st.write(classification_report(y_test,y_pred_rfc))


#st.dataframe(model_data)

recom_engine = st.container(border=True)
description_ , recom_engine_filter_ = recom_engine.columns([.8,.2])

recom_engine_filter = recom_engine_filter_.container(border=True)

#Select boxes with the variables for prediction
recom_engine_filter.write('Select Your Features')
gend = recom_engine_filter.selectbox('Select Gender',['Male','Female'])
age = recom_engine_filter.selectbox('Select Age',data['Age'].unique())
college = recom_engine_filter.selectbox('Select College',data['College_Name'].unique())
stream = recom_engine_filter.selectbox('Select Stream',data['Stream'].unique())
gpa = recom_engine_filter.selectbox('Select GPA',data['GPA'].unique())
exp = recom_engine_filter.selectbox('Select Experience (Years)',data['Years_of_Experience'].unique())

#ranking encoding and decoding
def gender_dummier(gend):
    if gend=='Male':
        return 0
    else:
        return 1

gender = gender_dummier(gend)

#creating a new row for predicting probabilty
predi_row = pd.DataFrame()
predi_row['Gender'] = [gender]
predi_row['Age'] = [age]
predi_row['GPA'] = [gpa]
predi_row['Years_of_Experience'] = [exp]
rank_ = data[data['College_Name']==college]['College_Rank'].tolist()[0]
def stream_dummier(stream):
    if stream=='Computer Science':
        predi_row['Computer Science'] = [1]
        predi_row['Electrical Engineering'] = [0]
        predi_row['Electronics and Communication'] = [0]  
        predi_row['Information Technology'] = [0]
        predi_row['Mechanical Engineering'] = [0]
    elif stream=='Electrical Engineering':
        predi_row['Computer Science'] = [0]
        predi_row['Electrical Engineering'] = [1]
        predi_row['Electronics and Communication'] = [0]  
        predi_row['Information Technology'] = [0]
        predi_row['Mechanical Engineering'] = [0]
    elif stream=='Electronics and Communication':
        predi_row['Computer Science'] = [0]
        predi_row['Electrical Engineering'] = [0]
        predi_row['Electronics and Communication'] = [1]  
        predi_row['Information Technology'] = [0]
        predi_row['Mechanical Engineering'] = [0]
    elif stream=='Information Technology':
        predi_row['Computer Science'] = [0]
        predi_row['Electrical Engineering'] = [0]
        predi_row['Electronics and Communication'] = [0]  
        predi_row['Information Technology'] = [1]
        predi_row['Mechanical Engineering'] = [0]
    elif stream=='Mechanical Engineering':
        predi_row['Computer Science'] = [0]
        predi_row['Electrical Engineering'] = [0]
        predi_row['Electronics and Communication'] = [0]  
        predi_row['Information Technology'] = [0]
        predi_row['Mechanical Engineering'] = [1]
stream_dummier(stream)

#ranking encoding and decoding
def rank_dummier(rank_):
    if 1<=rank_<=100:
        predi_row['100 - 200'] = [0]
        predi_row['200 - 300'] = [0]
        predi_row['300 - 400'] = [0]  
        predi_row['400 - 500'] = [0]
        predi_row['Top 100'] = [1]
    elif 100<rank_<=200:
        predi_row['100 - 200'] = [1]
        predi_row['200 - 300'] = [0]
        predi_row['300 - 400'] = [0]  
        predi_row['400 - 500'] = [0]
        predi_row['Top 100'] = [0]
    elif 200<rank_<=300:
        predi_row['100 - 200'] = [0]
        predi_row['200 - 300'] = [1]
        predi_row['300 - 400'] = [0]  
        predi_row['400 - 500'] = [0]
        predi_row['Top 100'] = [0]
    elif 300<rank_<=400:
        predi_row['100 - 200'] = [0]
        predi_row['200 - 300'] = [0]
        predi_row['300 - 400'] = [1]  
        predi_row['400 - 500'] = [0]
        predi_row['Top 100'] = [0]
    elif 400<rank_<=500:
        predi_row['100 - 200'] = [0]
        predi_row['200 - 300'] = [0]
        predi_row['300 - 400'] = [0]  
        predi_row['400 - 500'] = [1]
        predi_row['Top 100'] = [0]
rank_dummier(rank_)




description = description_.container(border=True)

description.write('Know your chances of getting placed!')
description.write('\n')
description.markdown('<div style="text-align: justify; font-size: 14px">Introducing our Cutting-Edge Recommendation Engine for Career Success! In today\'s fast-paced world, securing the ideal placement and shaping a prosperous future requires more than just academic prowess. That\'s where our state-of-the-art recommendation engine comes into play. Tailored specifically for students, our platform harnesses a diverse array of factors to provide personalized guidance and support, ensuring every individual is equipped to excel in their journey towards professional success. Our recommendation engine delves deep into various dimensions crucial for career preparedness, encompassing not only academic achievements but also extracurricular engagements, skill proficiencies, personality traits, and industry insights.</div>',unsafe_allow_html=True)
description.write('\n')
description.markdown('<div style="text-align: justify; font-size: 14px">By analyzing this multifaceted data, our engine gains a comprehensive understanding of each student\'s unique strengths and areas for development. Armed with this rich insight, our platform offers targeted recommendations designed to empower students in their preparations for placements and future endeavors. From suggesting relevant skill enhancement courses to providing tailored interview tips, networking opportunities, and career path guidance, our recommendations are meticulously crafted to address the specific needs and aspirations of each individual. By leveraging the power of data-driven insights and personalized recommendations, our engine empowers students to make informed decisions and take proactive steps towards achieving their career goals. Whether it\'s navigating the complexities of job interviews, exploring new career paths, or honing in-demand skills, our platform serves as a trusted companion, guiding students towards a brighter and more fulfilling future. With our recommendation engine at their disposal, students can embark on their career journey with confidence, knowing they have access to the resources and support needed to unlock their full potential and seize every opportunity that comes their way. Welcome to a world where personalized guidance meets unparalleled career success welcome to our recommendation engine.</div>',unsafe_allow_html=True)
description.write('\n')
description.markdown('<div style="text-align: justify; font-size: 14px">Based on the details your chanches of being placed</div>',unsafe_allow_html=True)
description.write('\n')

class_pred = model_rfc.predict_proba(predi_row)
description.metric(value = f'{(class_pred[0][0]*100).round(2)} %',label='Expected Placement Rate')
