import pytest
import numpy as np
from sabr.models.hagan_lognormal import HaganSABR

ERROR_TOLERANCE = 0.001  # 0.1% error tolerance

test_data = {
    'Beta=1 flat lognormal': {
        'inputs': {
            'atm_vol': 0.60,
            'forward': 0.02,
            'time': 1.5,
            'beta': 1.0,
            'rho': 0.0,
            'sigma': 0.0
        },
        'target_alpha': 0.60
    },
    'Beta=0 flat normal': {
        'inputs': {
            'atm_vol': 0.60,
            'forward': 2.0,
            'time': 1.5,
            'beta': 0.0,
            'rho': 0.0,
            'sigma': 0.0
        },
        'target_alpha': 1.1746
    },
    'Beta=0.5 10y': {
        'inputs': {
            'atm_vol': 0.20,
            'forward': 0.015,
            'time': 10.0,
            'beta': 0.5,
            'rho': -0.2,
            'sigma': 0.3
        },
        'target_alpha': 0.02310713
    }
}

@pytest.fixture(params=test_data.items(), ids=test_data.keys())
def sabr_test_case(request):
    return request.param[1]

def test_calibration(sabr_test_case):
    # Extract test data
    inputs = sabr_test_case['inputs']
    target_alpha = sabr_test_case['target_alpha']
    
    # Create model
    model = HaganSABR(beta=inputs['beta'])
    
    # Set initial parameters
    model.sigma = inputs['sigma']
    model.rho = inputs['rho']
    
    # Create test data arrays
    forward = inputs['forward']
    strike = inputs['forward']  # ATM strike
    time = inputs['time']
    market_vol = inputs['atm_vol']
    
    # Calibrate
    result = model.calibrate_atm(forward, time, market_vol)
    
    # Check result
    assert abs(model.alpha - target_alpha) <= ERROR_TOLERANCE * target_alpha, \
        f"Alpha mismatch: got {model.alpha}, expected {target_alpha}"

def test_atm_vol_repricing(sabr_test_case):
    # Extract test data
    inputs = sabr_test_case['inputs']
    
    # Create model
    model = HaganSABR(beta=inputs['beta'])
    
    # Set parameters
    model.sigma = inputs['sigma']
    model.rho = inputs['rho']
    model.alpha = inputs['atm_vol'] if inputs['beta'] == 1.0 else inputs['atm_vol'] * (inputs['forward']**(1-inputs['beta']))
    
    # Calculate ATM vol
    test_vol = model.implied_volatility(inputs['forward'], inputs['forward'], inputs['time'])
    
    # Check result
    assert abs(test_vol - inputs['atm_vol']) <= ERROR_TOLERANCE * inputs['atm_vol'], \
        f"Vol mismatch: got {test_vol}, expected {inputs['atm_vol']}"

def test_full_smile():
    """Test calibration and repricing of a full volatility smile"""
    # Test data
    forward = 0.03
    strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    time = 5.0
    market_vols = np.array([0.25, 0.23, 0.22, 0.23, 0.25])
    
    # Create model
    model = HaganSABR(beta=0.5)
    
    # Calibrate
    result = model.calibrate_smile(forward, strikes, time, market_vols)
    assert result.success, "Smile calibration failed"
    
    # Test repricing
    model_vols = model.implied_volatility(forward, strikes, time)
    
    # Check each strike point
    for i, (market_vol, model_vol) in enumerate(zip(market_vols, model_vols)):
        assert abs(model_vol - market_vol) <= ERROR_TOLERANCE * market_vol, \
            f"Vol mismatch at strike {strikes[i]}: got {model_vol}, expected {market_vol}"