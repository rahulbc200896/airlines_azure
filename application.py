from src.exception import Airlines_Exeption
from src.logger import logging
import os
import sys
from src.pipeline.predict_pipeline import Custom_data,PredictPipeline
from flask import Flask,render_template,request

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_ticket',methods=['GET','POST'])
def predict_ticket_price():
    try:

        if request.method == 'GET':
            return render_template("home.html")
        else:
            data = Custom_data(
                days_left = request.form.get('days_left'),
                duration = request.form.get('duration'),
                airline = request.form.get('airline'), 
                source_city = request.form.get('source_city'), 
                departure_time = request.form.get('departure_time'), 
                stops = request.form.get('stops'), 
                arrival_time = request.form.get('arrival_time'), 
                destination_city = request.form.get('destination_city'), 
                classs = request.form.get('class')
                )

            data_df = data.get_data_as_dataframe()
            print("Before Prediction")
            print(data_df)
            pipeline = PredictPipeline()
            results = pipeline.predict_data(data_df)

            return render_template("home.html",results=results[0])

    except Exception as e:
        raise Airlines_Exeption(e,sys)

if __name__ =='__main__':
    app.run(debug=True)
