import os
import base64
import io
from flask import Flask, render_template, request
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')  # Prevents GUI compilation errors on cloud deployments
import matplotlib.pyplot as plt

app = Flask(__name__)

def fig_to_base64(fig):
    """Converts a Matplotlib figure into a base64 string to render cleanly in HTML."""
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 1. Grab form metrics
        date_col = request.form.get('date_col')
        sales_col = request.form.get('sales_col')
        forecast_days = int(request.form.get('forecast_days'))
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template('index.html', error="No file selected.")

        try:
            # 2. Parse uploaded file format
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # Validate structural presence of columns
            if date_col not in df.columns or sales_col not in df.columns:
                return render_template('index.html', error=f"Could not find exact columns '{date_col}' or '{sales_col}' in file headers.")

            # 3. Transform data for Prophet engine
            df[date_col] = pd.to_datetime(df[date_col])
            prepared_df = df[[date_col, sales_col]].rename(columns={date_col: 'ds', sales_col: 'y'})
            prepared_df = prepared_df.sort_values('ds').reset_index(drop=True)

            # 4. Initialize and fit modeling sequence
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            model.fit(prepared_df)

            # 5. Build forecast vectors
            future = model.make_future_dataframe(periods=forecast_days, freq='D')
            forecast = model.predict(future)

            # Extract final elements for the table summary
            summary_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
            summary_df.columns = ['Date', 'Predicted Sales', 'Lower Value Bound', 'Upper Value Bound']
            summary_df['Date'] = summary_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Format numbers to look uniform
            for col in ['Predicted Sales', 'Lower Value Bound', 'Upper Value Bound']:
                summary_df[col] = summary_df[col].round(0).astype(int)

            # Generate Bootstrap styled responsive tables from Pandas
            html_table = summary_df.to_html(classes='table table-striped table-hover table-bordered', index=False)

            # 6. Transform visualization charts into static HTML safe streams
            fig1 = model.plot(forecast)
            plot1_base64 = fig_to_base64(fig1)
            plt.close(fig1)

            fig2 = model.plot_components(forecast)
            plot2_base64 = fig_to_base64(fig2)
            plt.close(fig2)

            return render_template(
                'index.html',
                summary_table=html_table,
                plot1=plot1_base64,
                plot2=plot2_base64
            )

        except Exception as e:
            return render_template('index.html', error=f"Processing Error: {str(e)}")

    # Standard initialization on standard GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
