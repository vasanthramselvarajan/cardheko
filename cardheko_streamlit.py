import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Set up the page layout
st.set_page_config(page_title="Used Car Price Prediction",page_icon="🚗",layout="wide")

st.title(" Used Car Price :blue[Prediction] 🚙💰")
st.header("Welcome to the :red[Cardheko] 🚘")


# Random Forest model
with open('randomforest_model.pkl', 'rb') as file:
    model = pickle.load(file)

df=pd.read_csv("cardheko_cleaned.csv")

tabs=st.sidebar.selectbox("Tabs",("Project Info","Feature Columns","Data Visualization","Model Performance"))

# Sidebar for future selection and model prediction
if tabs=="Feature Columns":
    st.sidebar.header("Feature Columns")
    transmission =st.sidebar.selectbox("Select Transmission",("Automatic","Manual"))
    ft = st.sidebar.selectbox("Select Fuel Type",('Petrol', 'Diesel', 'Electric', 'Cng', 'Lpg'))
    km= st.sidebar.number_input("Select KM (0 to 50,00,000)", min_value=min(df["km"]),max_value= max(df["km"]),value=10000,step=1000)
    max_power=st.sidebar.slider("Select Power", min_value=min(df["Max Power"]),max_value= max(df["Max Power"]),value=100.0,step=10.0)
    Mileage=st.sidebar.slider("Select Mileage", min_value=min(df["Mileage"]),max_value= max(df["Mileage"]),value=10.0,step=0.5)
    insurance=st.sidebar.selectbox("Select Insurance Type",('Not Available','Third Party', 'Comprehensive', 'Zero Dep'),index=1)
    model_year=st.sidebar.selectbox("Select Model Year",(1985, 1995, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023),index=27)
    owner=st.sidebar.selectbox("Select Owner",('5th Owner','4th Owner','3rd Owner','2nd Owner','1st Owner','0th Owner'),index=4)
    bt=st.sidebar.selectbox("Select Body Type",(df["bt"].unique()))
    oem=st.sidebar.selectbox("Select Car Brand",(df["oem"].unique()),index=0)
    city=st.sidebar.selectbox("Select City",(df["city"].unique()))
    cylinder=st.sidebar.selectbox("Select No. Cylinders",([0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),index=3)
    Seats=st.sidebar.selectbox("Select No of Seats",(df["Seats"].unique()))
    color=st.sidebar.selectbox("Select Color",(df["Color"].unique()))
    wheelbase=st.sidebar.slider("Select Wheel base", min_value=2000,max_value=3200,value=2500,step=100)

    if st.sidebar.button("Predict"):
        st.subheader("These are the :violet[car details]:")
        st.write("Transmission :  ",transmission)
        st.write("Fuel Type :  ",ft)
        st.write("KM :  ",km)
        st.write("Power :  ",max_power)
        st.write("Mileage :  ",Mileage)
        st.write("Insurance Type :  ",insurance)
        st.write("Model Year :  ",model_year)
        st.write("Owner :  ",owner)
        st.write("Body Type :  ",bt)
        st.write("Car Brand :  ",oem)
        st.write("City :  ",city)
        st.write("No. Cylinders :  ",cylinder)
        st.write("No of Seats :  ",Seats)
        st.write("Color :  ",color)
        st.write("Wheel Base :  ",wheelbase)
        st.write("dataframe : ")

        d= {
            'km':km,
            'Max Power':max_power,
            'Mileage':Mileage,
            'Insurance Validity':insurance,
            'modelYear':model_year,
            'owner':owner,
            'No of Cylinder':cylinder,
            'oem':oem,
            'bt':bt,
            'city':city,
            'Color':color,
            'Seats':Seats,
            'ft':ft,
            'transmission':transmission,
            'Wheel Base':wheelbase
        }

        data= pd.DataFrame(d,index=[0])
        st.dataframe(data)

        with open('ordinal_insurance.pkl', 'rb') as oe_file:
            ordinal_encoder = pickle.load(oe_file)
            data["Insurance Validity"]=ordinal_encoder.transform(data[["Insurance Validity"]])

        with open('ordinal_modelyear.pkl', 'rb') as oe_file:
            ordinal_encoder = pickle.load(oe_file)
            data["modelYear"]=ordinal_encoder.transform(data[["modelYear"]])

        with open('ordinal_owner.pkl', 'rb') as oe_file:
            ordinal_encoder = pickle.load(oe_file)
            data["owner"]=ordinal_encoder.transform(data[["owner"]])

        with open('ordinal_cylinder.pkl', 'rb') as oe_file:
            ordinal_encoder = pickle.load(oe_file)
            data["No of Cylinder"]=ordinal_encoder.transform(data[["No of Cylinder"]])

        
        for i,j in zip(['oem','bt','city','Color','Seats','ft','transmission'],["label_oem.pkl","label_bt.pkl","label_city.pkl","label_Color.pkl","label_Seats.pkl","label_ft.pkl","label_transmission.pkl"]):
            with open(j, 'rb') as le_file:
                label_encoder = pickle.load(le_file)
                data[i]=label_encoder.transform(data[[i]])

        st.write("Encoded dataframe : ")

        st.dataframe(data)

        
        predict=model.predict(data)
        
        with st.spinner(text="Predicting the car price..."):
            time.sleep(2)
        
        #st.balloons()
        st.subheader("Pridicted car price is : ")
        #st.header(f"{predict:.2f} Lakhs")
        st.header(f"{predict[0]:.2f} Lakhs")



# Data Visualization
elif tabs == "Data Visualization":
    st.sidebar.header("Data Visualization")
    st.sidebar.markdown("""
    ### Data Visualization
    Visualize and explore your data with various charts:
    - **Bar Chart**: Compare categories and their frequencies.
    - **Box Plot**: Understand the spread and outliers of numeric data.
    - **Scatter Plot**: Analyze relationships between two numeric variables.
    - **Histogram**: Show the distribution of numeric data.

    Interact with the selectbox to customize and gain insights from the data.
    """)
    st.subheader("Use Different Types of Charts for :green[Visualization]")
    chart=st.selectbox("Select Charts",("Bar chart","Histogram","Box plot","Scatter Plot"))

    if chart=="Bar chart":
        x=st.selectbox("Select x (categorical)",(df.columns))
        y=st.selectbox("Select y (numeric or categorical)",(df.columns))
        if df[y].dtype == "object":
            agg_func = st.selectbox("Select aggregation function", ["count"])
            car_counts_by_year = df.groupby(x)[y].count()
        else:
            agg_func = st.selectbox("Select aggregation function", ["sum", "mean", "median"])
            if agg_func == "sum":
                car_counts_by_year = df.groupby(x)[y].sum()
            elif agg_func == "mean":
                car_counts_by_year = df.groupby(x)[y].mean()
            elif agg_func == "median":
                car_counts_by_year = df.groupby(x)[y].median()
        plt.bar(car_counts_by_year.index, car_counts_by_year.values, color='red')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Bar Chart of {x} vs {y}')
        if st.button("Show Plot"):
            st.pyplot(plt)

    elif chart=="Scatter Plot":
        x=st.selectbox("Select x",(df.columns))
        y=st.selectbox("Select y",(df.columns))
        sns.scatterplot(x=x, y=y, data=df)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{y} vs {x}')
        if st.button("Show Plot"):
            st.pyplot(plt)

    elif chart=="Histogram":
        x=st.selectbox("Select column name",(df.columns))
        plt.hist(df[x], bins=25, color='skyblue', edgecolor='black')
        plt.xlabel(x)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {x}')
        if st.button("Show Plot"):
            st.pyplot(plt)

    else:
        x = st.selectbox("Select x (categorical)", df.columns)
        y = st.selectbox("Select y (numeric)", df.columns)
        plt.figure(figsize=(8, 6))
        df.boxplot(column=y, by=x, patch_artist=True, notch=True, vert=False, grid=False, 
                boxprops=dict(facecolor='skyblue', color='black'),
                whiskerprops=dict(color='black'), flierprops=dict(marker='o', color='red', markersize=6))
        plt.xlabel(y)
        plt.ylabel(x)
        plt.title(f'Box Plot of {y} by {x}')
        plt.suptitle('')
        if st.button("Show Plot"):
            st.pyplot(plt)

elif tabs == "Project Info":
    st.image("C:/Users/HP/Downloads/Cardekho_logo.jpg")
    st.markdown('[Cardheko](https://www.cardekho.com/)')
    st.markdown('''This interactive tool predicts the prices of used cars based on key attributes such as make, model, year, fuel type, transmission, mileage, and location.

### How to Use:

1. **Select Car Details**: Select the car's make, model, year, fuel type, transmission, mileage, city ect...
2. **Get Price Estimate**: Submit the details and get an instant price prediction for the used car.
3. **Evaluate Accuracy**: View model performance metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

### Try it now and get an accurate price estimate for your used car!''')
    st.subheader("")
    st.write("To know about packages use for this project ... click below!")
    if st.button("Show Packages"):
        st.subheader("These are the :green[packages] needed:")
        with st.echo():
            import streamlit as st
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split, cross_val_score, KFold
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error,mean_absolute_error
            import pickle

elif tabs == "Model Performance":
    st.sidebar.markdown("""
    ### Regression performance metrics
    - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
  
    - **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values.

    - **Mean Absolute Percentage Error (MAPE)**: Measures the average percentage difference between actual and predicted values.

    - **R-squared (R²)**: Indicates how well the model's predictions match the actual values.
    """)
    st.header("All Model and its Metrics")
    model=st.selectbox("Model",("Linear Regression","Random Forest Regressor","Decision Tree Regressor","Gradient Boosting Regressor"))
    if model=="Linear Regression":
        st.markdown('''
        ### Linear Regression model
        Testing Performance Metrics:
        - Mean Squared Error:  76.34055143042924
        - Squared MSE:  8.737308019660817
        - Mean Absolute Precentage Error:  0.7119235347397102
        - R2 Score: 0.4757486528448698
        
        Training Performance Metrics:
        - Mean Squared Error:  81.36432044317432
        - Squared MSE:  9.020217316848543
        - Mean Absolute Precentage Error:  0.7023246618079874
        - R2 Score: 0.6246971220893498       
        ''')
    elif model=="Random Forest Regressor":
        st.markdown('''
        ### RandomForest Regressor model
        Testing Performance Metrics:
        - Mean Squared Error:  5.369932136274251
        - Mean Absolute Error:  1.0682652439653326
        - Mean Absolute Precentage Error:  0.1466133840679531
        - R2 Score: 0.9348600305226451 
                
        Training Performance Metrics:
        - Mean Squared Error:  0.9683358534190487
        - Squared MSE:  1.0682652439653326
        - Mean Absolute Precentage Error:  0.05555388015058994
        - R2 Score: 0.9901990152206179
        #### This is the model we used for car price prediction
        ''')

    elif model=="Decision Tree Regressor":
        st.markdown('''
        ### Decision Tree Regressor model
        Testing Performance Metrics:
        - Mean Squared Error:  8.897339949666934
        - Mean Absolute Error:  1.4784085582822086
        - Mean Absolute Precentage Error:  0.2008354269089165
        - R2 Score: 0.8920708049854292 
                
        Training Performance Metrics:
        - Mean Squared Error:  0.01602422150636601
        - Mean Absolute Error:  1.4784085582822086
        - Mean Absolute Precentage Error:  0.0007863431774064663
        - R2 Score: 0.9998378112815602
        ''')

    elif model=="Gradient Boosting Regressor":
        st.markdown('''
        ### Gradient Boosting Regressor model
        Testing Performance Metrics:
        - Mean Squared Error:  6.048806772200883
        - Mean Absolute Error:  1.2950243855931727
        - Mean Absolute Precentage Error:  0.18722971089733936
        - R2 Score: 0.9266249407783094 
                
        Training Performance Metrics:
        - Mean Squared Error:  4.888016886959444
        - Mean Absolute Error:  1.2950243855931727
        - Mean Absolute Precentage Error:  0.18037771198517574
        - R2 Score: 0.950526071154653
        ''')



