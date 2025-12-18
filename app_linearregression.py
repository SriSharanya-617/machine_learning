import streamlit as slt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# page configuration #
slt.set_page_config("Linear Regression App",layout="centered")

# install requiremnts : pip install -r requirements.txt #

# load css #
def load_css(file):
    with open(file) as f:
        slt.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
load_css("style.css")

# title #
slt.markdown("""
<div class="card">
        <h1> Linear Regression </h1>
        <p>Predict <b>Tip Amount </b>from <b>Total Bill</b> using Linear Regression...</p>
</div>  
                        
""",unsafe_allow_html=True)

# Load Data #
@slt.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

# Dataset Preview #
slt.markdown('<div class="card">',unsafe_allow_html=True)
slt.subheader("Dataset Preview")
slt.dataframe(df.head())
slt.markdown('</div>',unsafe_allow_html=True)

# Prepare Data #
x,y = df[["total_bill"]],df["tip"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Model #
model=LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

# Metrics #
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mse)
adj_r2= 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# Visualizations #
slt.markdown('<div class="card">',unsafe_allow_html=True)
slt.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df["total_bill"],df["tip"],alpha=0.6)
ax.plot(df["total_bill"],model.predict(scaler.transform(x)),color='red')
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
slt.pyplot(fig)
slt.markdown('</div>',unsafe_allow_html=True)

# performance metrics #
slt.markdown('<div class="card">',unsafe_allow_html=True)
slt.subheader("Model Performance Metrics")
c1,c2=slt.columns(2)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c3,c4=slt.columns(2)
c3.metric("R-squared (R2)",f"{r2:.2f}")
c4.metric("Adjusted R-squared",f"{adj_r2:.2f}")
slt.markdown('</div>',unsafe_allow_html=True)

# m & c #
slt.markdown(f"""
<div class="card">
    <h3> Model Interception </h3>
    <p><b>co-efficiecent: </b>{model.coef_[0]:.2f}</p>
    <p><b>intercept: </b>{model.intercept_:.2f}</p>
</div> 
""",unsafe_allow_html=True)

# predictions
slt.markdown('<div class="card">',unsafe_allow_html=True)
slt.subheader("Predict Tip Amount")
bill_amount = slt.slider("Total Bill ($):",float(df.total_bill.min()),float(df.total_bill.max()),30.0)
tip=model.predict(scaler.transform([[bill_amount]]))[0]
slt.markdown(f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',unsafe_allow_html=True)
slt.markdown('</div>',unsafe_allow_html=True)