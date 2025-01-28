# Stock Performance Dashboard

This is a small test project designed to learn how to use the OpenAI API. It leverages OpenAI's GPT models and the `yfinance` Python library to fetch news, key indicators, and stock performance using stock tickers. The data is presented in a Dash-based dashboard. Additionally, the key indicators can be analyzed by GPT-4 to provide a buy or sell recommendation.

## Features
- **Stock Performance Analysis**: Visualizes stock prices over time.
- **Key Financial Indicators**: Displays metrics like P/E ratio, EPS, and more.
- **Company News**: Fetches recent news headlines for each stock ticker.
- **GPT-4 Analysis**: Provides investment recommendations based on key metrics and recent news.

## Installation

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**  
   Make sure you have Python installed (version 3.7+). Then, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables**  
   Create a `.env` file in the project directory and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the App**  
   Launch the application by running:
   ```bash
   python app.py
   ```

5. **Access the Dashboard**  
   Open your browser and navigate to `http://127.0.0.1:8050`.

## How It Works
1. Enter stock tickers (e.g., `DELL KMI NEE TSM`) and a time period (e.g., `1 Year`) in the input fields.
2. Fetch stock data and view:
   - **Stock Performance** (visualized charts).
   - **Key Indicators** (financial metrics and ratios).
   - **Company News** (recent articles).
3. Click "Generate GPT Analysis" to get AI-driven insights, including an investment recommendation.

