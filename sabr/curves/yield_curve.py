import numpy as np
import pandas as pd

class YieldCurve:
    def __init__(self, libor_data, shift=0.025):
        """
        Initialize yield curve from Libor data
        
        Args:
            libor_data: DataFrame with columns ['Year', 'Ask', 'Bid']
            shift: Shift to add to rates (default 0.025 = 2.5%)
        """
        self.libor_data = libor_data
        self.shift = shift
        
        # Calculate mid rates and discount factors
        self.libor_data['Mid'] = 0.5 * (self.libor_data['Ask'] + self.libor_data['Bid']) + self.shift
        self.libor_data['Disc_Factor'] = 1 / (1 + self.libor_data['Year'] * self.libor_data['Mid'] / 100)
        
        # Store discount factors for interpolation
        self.years = np.array([0] + list(self.libor_data['Year']))
        self.discount_factors = np.array([1.0] + list(self.libor_data['Disc_Factor']))

    def interpolate_df(self, time, time1):
        """Linear interpolation of discount factors between time points"""
        time = max(0, time)
        time1 = max(0, time1)
        return float(time1-time) * self.get_df(time1-1) + float(time-(time1-1)) * self.get_df(time1)

    def get_df(self, time):
        """Get discount factor for a given time"""
        time = max(0, time)
        if time == 0:
            return 1.0
        idx = np.searchsorted(self.years, time)
        if idx >= len(self.years):
            return self.discount_factors[-1]
        if self.years[idx] == time:
            return self.discount_factors[idx]
        # Linear interpolation
        t0, t1 = self.years[idx-1], self.years[idx]
        d0, d1 = self.discount_factors[idx-1], self.discount_factors[idx]
        return d0 + (d1 - d0) * (time - t0)/(t1 - t0)

    def get_zero_disc_matrix(self, tenors, maturities):
        """
        Create matrix of discount factors for given tenors and maturities
        """
        matrix = np.zeros((len(tenors), len(maturities)))
        for i, tenor in enumerate(tenors):
            for j, maturity in enumerate(maturities):
                time = maturity + tenor
                matrix[i,j] = self.interpolate_df(time, int(time)+1)
        return matrix

    def calculate_pvbp(self, year, discount_factors):
        """Calculate present value of a basis point"""
        denom = 0.0
        for n in range(year+1):
            for m in range(len(discount_factors)):  # Use actual matrix size
                denom += discount_factors[m][n]
        denom = denom - discount_factors[0][0] + discount_factors[0][year+1]
        return denom

    def calculate_forward_rate(self, year, discount_factors):
        """Calculate forward rate"""
        pvbp = self.calculate_pvbp(year, discount_factors)
        if pvbp == 0:
            return 0
        forward = (discount_factors[0][0] - discount_factors[0][year+1])/(0.25*pvbp)
        return forward 
    
    def get_forward_rates_for_swaptions(self, swaption_data):
        """Calculate forward rates for swaption structure"""
        forward_rates = []
        
        # 1M tenors
        disc_1m = self.get_zero_disc_matrix([1/12, 4/12, 7/12, 10/12], range(11))
        for year in [0, 1, 2, 4, 9]:
            fwd = self.calculate_forward_rate(year, disc_1m)
            forward_rates.append(fwd)
            
        # 3M tenors
        disc_3m = self.get_zero_disc_matrix([3/12, 6/12, 9/12, 12/12], range(11))
        for year in [0, 1, 2, 4, 9]:
            fwd = self.calculate_forward_rate(year, disc_3m)
            forward_rates.append(fwd)
            
        # 6M tenors
        disc_6m = self.get_zero_disc_matrix([6/12, 9/12, 12/12, 15/12], range(11))
        for year in [0, 1, 2, 4, 9]:
            fwd = self.calculate_forward_rate(year, disc_6m)
            forward_rates.append(fwd)
            
        # 9M tenors
        disc_9m = self.get_zero_disc_matrix([9/12, 12/12, 15/12, 18/12], range(11))
        for year in [1, 4, 9]:
            fwd = self.calculate_forward_rate(year, disc_9m)
            forward_rates.append(fwd)
            
        return forward_rates