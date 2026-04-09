import streamlit as st
import pickle
import pandas as pd

# Loading all pickle files
l=pickle.load(open('log.pkl','rb'))
d=pickle.load(open('dtree.pkl','rb'))
r=pickle.load(open('rfor.pkl','rb'))
a=pickle.load(open('adab.pkl','rb'))
g=pickle.load(open('grad.pkl','rb'))
x=pickle.load(open('xgb.pkl','rb'))
l_g=pickle.load(open('lgb.pkl','rb'))
b=pickle.load(open('bagg.pkl','rb'))

st.title('Bankruptcy Prevention Project')

# Input Parameters
def input_feature():
  c1,c2,c3=st.columns([1,1,1])
  with c1:
    ind=st.radio('Industrial Risk',('High','Medium','Low'))
    mgmt=st.radio('Managaement Risk',('High','Medium','Low'))
  with c2:
    fin=st.radio('Financial Flexibility',('High','Medium','Low'))
    cred=st.radio('Credibility',('High','Medium','Low'))
  with c3:
    comp=st.radio('Competitiveness',('High','Medium','Low'))
    # opr=st.radio('Operating Risk',('High','Medium','Low'))
  inp={'industrial_risk':ind,'management_risk':mgmt,'financial_flexibility':fin,'credibility':cred,'competitiveness':comp}
  # ,'operating_risk':opr}
  df=pd.DataFrame([inp]) # Converting to a Table
  # Mapping the columns
  df['industrial_risk']=df['industrial_risk'].map({'High':1,'Medium':0.5,'Low':0})
  df['management_risk']=df['management_risk'].map({'High':1,'Medium':0.5,'Low':0})
  df['financial_flexibility']=df['financial_flexibility'].map({'High':1,'Medium':0.5,'Low':0})
  df['credibility']=df['credibility'].map({'High':1,'Medium':0.5,'Low':0})
  df['competitiveness']=df['competitiveness'].map({'High':1,'Medium':0.5,'Low':0})
  # df['operating_risk']=df['operating_risk'].map({'High':1,'Medium':0.5,'Low':0})
  return df
st.divider()
y=input_feature()
c1,c2=st.columns([1,1])
with c1:
  # Selection of Model
  model={'LogisticRegression':l,'Decision Tree':d,'Random Forest':r,'Ada Boost':a,'Gradient Boost':g,'Extreme Gradient Boost':x,'Light Gradient Boost':l_g,'Bagging':b}
  m=st.selectbox('Choose a Model for Prediction',list(model.keys()))
  md=model[m]
  # Prediction
  pred=md.predict(y)
  prob=md.predict_proba(y)
  button=st.button('Prediction')
with c2:
  if button is True:
    st.subheader('Prediction')
    st.write('Bankrupt' if prob[0][1]>0.5 else 'Not Bankrupt') # Output
    # Confidence Level
    if prob[0][0]>prob[0][1]:
      st.write(f'With confidence of {round(prob[0][0],4)*100}%')
    else:
      st.write(f'With confidence of {round(prob[0][1],4)*100}%')
