import io
import base64
import matplotlib
# Force matplotlib to use non-interactive backend for web servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from flask import Flask, render_template, request

app = Flask(__name__)

def load_and_prepare_data(file_stream, file_name, date_col, sales_col):
    """Loads and pre-processes data from the Flask file stream."""
    if file_name.lower().endswith('.csv'):
        df = pd.read_csv(file_stream)
    else:
        df = pd.read_excel(file_stream)
        
    df.columns = df.columns.str.strip()
    
    if date_col not in df.columns or sales_col not in df.columns:
        raise KeyError(f"Columns '{date_col}' or '{sales_col}' not found. Available columns: {list(df.columns)}")
        
    df[date_col] = pd.to_datetime(df[date_col])
    prepared_df = df[[date_col, sales_col]].rename(columns={date_col: 'ds', sales_col: 'y'})
    prepared_df = prepared_df.dropna(subset=['ds', 'y'])
    prepared_df = prepared_df.sort_values('ds').reset_index(drop=True)
    return prepared_df

def forecast_sales(df, forecast_periods=30, frequency='D'):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_periods, freq=frequency)
    forecast = model.predict(future)
    return model, forecast

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 string for HTML rendering."""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    base64_str = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig) # Free up memory
    return base64_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file.")
            
        # Get configurations from the web form
        date_col = request.form.get('date_col', 'Date')
        sales_col = request.form.get('sales_col', 'Sales')
        forecast_days = int(request.form.get('forecast_days', 90))
        
        try:
            # Process the file
            df = load_and_prepare_data(file, file.filename, date_col, sales_col)
            model, forecast = forecast_sales(df, forecast_periods=forecast_days)
            
            # Extract the next 5 days summary
            summary_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
            # Format dates nicely for the web table
            summary_df['ds'] = summary_df['ds'].dt.strftime('%Y-%m-%d')
            # Convert pandas dataframe to HTML table
            summary_html = summary_df.to_html(classes="table table-striped", index=False, float_format="%.2f")
            
            # Generate Main Plot
            fig1 = model.plot(forecast)
            plt.title("Sales Forecast Visualization")
            plt.xlabel("Date")
            plt.ylabel("Sales Volume")
            plot1_b64 = fig_to_base64(fig1)
            
            # Generate Components Plot
            fig2 = model.plot_components(forecast)
            plot2_b64 = fig_to_base64(fig2)
            
            return render_template('index.html', 
                                   summary_table=summary_html, 
                                   plot1=plot1_b64, 
                                   plot2=plot2_b64)
                                   
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
