from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
from sabr.curves.yield_curve import YieldCurve
from sabr.models.hagan_lognormal import HaganSABR

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    try:
        # Get files from request
        swaption_file = request.files['swaption_data']
        libor_file = request.files['libor_data']
        
        # Save files temporarily
        swaption_path = os.path.join(UPLOAD_FOLDER, 'swaption.xlsx')
        libor_path = os.path.join(UPLOAD_FOLDER, 'libor.xlsx')
        swaption_file.save(swaption_path)
        libor_file.save(libor_path)
        
        try:
            # Read market data
            swaption_data = pd.read_excel(swaption_path, index_col=0)
            libor_data = pd.read_excel(libor_path)
            
            print("Initial data:")
            print("Swaption columns:", swaption_data.columns)
            print("Sample data:\n", swaption_data.head())
            
            # Initialize yield curve
            curve = YieldCurve(libor_data, shift=0.025)
            
            # Calculate forward rates
            forward_rates = curve.get_forward_rates_for_swaptions(swaption_data)
            forward_rates = np.array(forward_rates)
            print("Forward rates:", forward_rates)
            
            # Convert market data to volatility matrix
            # First convert ATM volatilities
            market_vols = pd.DataFrame(index=swaption_data.index)
            market_vols['ATM'] = swaption_data['ATM'].astype(float) / 100
            
            # Add other strikes as spreads to ATM
            for col in swaption_data.columns:
                if col != 'ATM':
                    # Convert basis point spreads to absolute volatilities
                    market_vols[col] = market_vols['ATM'] + swaption_data[col].astype(float) / 10000
            
            # Sort columns to match strike order
            cols = sorted([col for col in market_vols.columns if col != 'ATM'])
            market_vols = market_vols[['ATM'] + cols]
            
            print("Market vols sample:\n", market_vols.head())
            
            # Create strike grid
            strikes = np.zeros((len(forward_rates), len(market_vols.columns)))
            for i in range(len(forward_rates)):
                for j, col in enumerate(market_vols.columns):
                    if col == 'ATM':
                        strikes[i,j] = forward_rates[i]
                    else:
                        # Ensure strikes are positive by using absolute value of basis points
                        bps = abs(float(col)) if isinstance(col, (int, float)) else 0
                        sign = 1 if (isinstance(col, (int, float)) and col > 0) else -1
                        strikes[i,j] = forward_rates[i] + sign * bps * 0.0001
            
            # Ensure all strikes are positive
            min_strike = 0.0001  # Set minimum strike to avoid log(0) issues
            strikes = np.maximum(strikes, min_strike)
            
            print("Strikes shape:", strikes.shape)
            print("Strike range:", np.min(strikes), "to", np.max(strikes))
            print("Sample strikes:", strikes[0])
            
            # Extract option tenors
            tenors = []
            for idx in swaption_data.index:
                months = float(idx.split('M')[0])  # Extract months from index (e.g., "1M2Y" -> 1)
                tenors.append(months/12.0)  # Convert to years
            
            tenors = np.array(tenors)
            print("Tenors:", tenors)
            
            # Convert market vols to numpy array
            market_vols_array = market_vols.values
            
            # Calibrate SABR model
            model = HaganSABR(beta=0.5)
            result = model.calibrate(forward_rates, strikes, tenors, market_vols_array)
            
            # Clean up temporary files
            os.remove(swaption_path
            os.remove(libor_path)
            
            return jsonify({
                'success': True,
                'forward_rates': forward_rates.tolist(),
                'calibrated_parameters': {
                    'alpha': float(model.alpha),
                    'sigma': float(model.sigma),
                    'rho': float(model.rho),
                    'beta': float(model.beta)
                },
                'optimization_success': bool(result.success),
                'optimization_message': str(result.message)
            })
            
        except Exception as e:
            print(f"Error during calibration: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def main():
    app.run(debug=True)

if __name__ == '__main__':
    main() 