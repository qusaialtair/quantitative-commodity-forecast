import yfinance as yf
import json
import sys

def main():
    try:
        # Fetch data for Gold (ticker: GC=F)
        ticker = yf.Ticker('GC=F')
        
        # Pull today's live market data (1d period)
        todays_data = ticker.history(period='1d')
        
        if todays_data.empty:
            print(json.dumps({"error": "No market data available for Gold (GC=F)"}, indent=4))
            sys.exit(1)
            
        # Extract the current price (using the last close price)
        current_price = todays_data['Close'].iloc[-1]
        
        # Format as JSON
        output = {
            "asset": "Gold",
            "ticker": "GC=F",
            "current_price": round(float(current_price), 4),
            "currency": "USD"
        }
        
        # Print cleanly to the console
        print(json.dumps(output, indent=4))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=4))
        sys.exit(1)

if __name__ == "__main__":
    main()