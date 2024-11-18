import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=dropout_prob)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(out[:, -1, :])
        out = self.dropout(self.relu(out))
        out = self.fc2(out)
        return out

def prepare_stock_data(df, ma_periods=[5, 10, 20, 50]):
    """Prepare stock data with the same preprocessing as training"""
    data = df.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    for period in ma_periods:
        data[f'MA_{period}'] = data['Adj Close'].rolling(window=period).mean()

    data['Price_Change'] = data['Adj Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()

    selected_features = ['Adj Close', 'Volume', 'Price_Change', 'Volume_Change', 'sentiment'] + \
                       [f'MA_{period}' for period in ma_periods]

    processed_data = data[selected_features]
    processed_data.dropna(inplace=True)
    return processed_data

def create_sequences(data, sequence_length):
    """Create sequences from data"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def load_model(stock_symbol):
    """Load the saved model and scaler for a specific stock"""
    try:
        model_state = torch.load(f'stock_model_{stock_symbol.lower()}.pth')

        loaded_model = LSTMModel(input_dim=9, hidden_dim=128, layer_dim=2, output_dim=1)
        loaded_model.load_state_dict(model_state['model_state_dict'])
        loaded_model.eval()

        return loaded_model, model_state['scaler']
    except Exception as e:
        raise Exception(f"Error loading model for {stock_symbol}: {str(e)}")

def get_validation_predictions(model, X_val, y_val, scaler, n_samples):
    """Get predictions for validation set"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i in range(n_samples):
            pred = model(torch.tensor(X_val[i:i+1], dtype=torch.float32))
            predictions.append(pred.item())
            actuals.append(y_val[i])

    # Inverse transform predictions and actuals
    pred_array = np.array(predictions).reshape(-1, 1)
    actual_array = np.array(actuals).reshape(-1, 1)

    original_shape = scaler.scale_.shape[0]
    temp_pred = np.zeros((len(predictions), original_shape))
    temp_actual = np.zeros((len(actuals), original_shape))

    temp_pred[:, 0] = pred_array[:, 0]
    temp_actual[:, 0] = actual_array[:, 0]

    predictions_transformed = scaler.inverse_transform(temp_pred)[:, 0]
    actuals_transformed = scaler.inverse_transform(temp_actual)[:, 0]

    return predictions_transformed, actuals_transformed

def create_prediction_plot(historical_values, predicted_values, actual_values, dates):
    """Create an interactive plot using plotly"""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=dates[:-len(predicted_values)],
        y=historical_values[:-len(predicted_values)],
        name='Historical',
        line=dict(color='blue')
    ))

    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates[-len(actual_values):],
        y=actual_values,
        name='Actual',
        line=dict(color='green')
    ))

    # Add predictions
    fig.add_trace(go.Scatter(
        x=dates[-len(predicted_values):],
        y=predicted_values,
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title='Stock Price Prediction vs Actual Values',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )

    return fig

def predict_stock_price(stock_symbol, n_samples):
    try:
        # Load the stock data
        df = pd.read_csv(f"processed_stock_data_{stock_symbol.lower()}.csv")
        df = prepare_stock_data(df)

        # Load model and scaler
        loaded_model, loaded_scaler = load_model(stock_symbol)

        # Prepare validation data
        features = ['Adj Close', 'Volume', 'Price_Change', 'Volume_Change',
                   'MA_5', 'MA_10', 'MA_20', 'MA_50', 'sentiment']

        scaled_data = loaded_scaler.transform(df[features])
        sequence_length = 24
        X, y = create_sequences(scaled_data, sequence_length)

        # Use last 20% as validation set
        train_size = int(len(X) * 0.8)
        X_val = X[train_size:]
        y_val = y[train_size:]
        val_dates = df.index[train_size + sequence_length:]

        # Get predictions for n_samples from validation set
        predictions, actuals = get_validation_predictions(
            loaded_model, X_val, y_val, loaded_scaler, n_samples
        )

        # Create the plot
        historical_data = df['Adj Close'].values
        fig = create_prediction_plot(
            historical_data,
            predictions,
            actuals,
            list(df.index)[-len(historical_data):]
        )

        # Calculate error metrics
        mse = np.mean((actuals - predictions) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # Create summary
        summary = pd.DataFrame({
            'Date': val_dates[:n_samples],
            'Actual Price': actuals,
            'Predicted Price': predictions,
            'Absolute Error': np.abs(actuals - predictions),
            'Percentage Error': np.abs((actuals - predictions) / actuals) * 100
        })

        metrics_summary = f"\nError Metrics:\nRMSE: ${rmse:.2f}\nMAPE: {mape:.2f}%\n\n"
        summary_str = metrics_summary + summary.to_string()

        return fig, summary_str

    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def main():
    # Create Gradio interface
    iface = gr.Interface(
        fn=predict_stock_price,
        inputs=[
            gr.Dropdown(
                choices=["AAPL", "GOOG", "MSFT", "BABA", "BAC", "D", "FB", "PCLN", "T", "AMZN"],
                label="Stock Symbol"
            ),
            gr.Slider(
                minimum=5,
                maximum=50,
                step=5,
                value=20,
                label="Number of Validation Samples"
            )
        ],
        outputs=[
            gr.Plot(label="Stock Price Prediction Plot"),
            gr.Textbox(label="Prediction Summary and Metrics", lines=15)
        ],
        title="Stock Price Prediction with LSTM (Validation Testing)",
        description="Compare predicted prices with actual values from validation set.",
    )
    
    # Launch the app
    iface.launch()

if __name__ == "__main__":
    main()
