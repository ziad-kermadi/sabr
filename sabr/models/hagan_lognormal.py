import numpy as np
from scipy.optimize import minimize
from sabr.models.base_sabr import BaseSABR

class HaganSABR(BaseSABR):
    def implied_volatility(self, F, K, T):
        """
        Implementation of Hagan's lognormal SABR formula
        """
        # Convert inputs to numpy arrays
        F = np.asarray(F, dtype=float).reshape(-1)
        K = np.asarray(K, dtype=float).reshape(-1)
        T = np.asarray(T, dtype=float).reshape(-1)
        
        # Broadcast F and T to match K's shape if needed
        if F.size == 1:
            F = np.full_like(K, F[0])
        if T.size == 1:
            T = np.full_like(K, T[0])
        
        # Ensure minimum values
        F = np.maximum(F, 1e-10)
        K = np.maximum(K, 1e-10)
        T = np.maximum(T, 1e-10)
        
        # ATM case
        atm_mask = np.abs(F - K) < 1e-10
        result = np.zeros_like(K)
        result[atm_mask] = self.alpha / (F[atm_mask]**(1-self.beta))
        
        # Non-ATM case
        non_atm = ~atm_mask
        if np.any(non_atm):
            F_non_atm = F[non_atm]
            K_non_atm = K[non_atm]
            T_non_atm = T[non_atm]
            
            S0K = F_non_atm * K_non_atm
            lS0K = np.log(F_non_atm/K_non_atm)
            
            z = (self.sigma/self.alpha) * (S0K**((1-self.beta)/2)) * lS0K
            x = np.log((np.sqrt(1-2*self.rho*z+z**2)+z-self.rho)/(1-self.rho))
            
            denom = 1 + ((1-self.beta)*lS0K)**2/24 + ((1-self.beta)*lS0K)**4/1920
            
            term1 = ((1-self.beta)*self.alpha)**2/(24*(S0K**(1-self.beta)))
            term2 = (self.rho*self.beta*self.sigma*self.alpha)/(4*(S0K**((1-self.beta)/2)))
            term3 = (self.sigma**2)*(2-3*(self.rho**2))/24
            numer = 1 + T_non_atm*(term1 + term2 + term3)
            
            result[non_atm] = (self.alpha * numer * (z/x))/(denom * (S0K**((1-self.beta)/2)))
        
        # Handle any remaining numerical issues
        bad_values = ~np.isfinite(result)
        if np.any(bad_values):
            result[bad_values] = self.alpha / (F[bad_values]**(1-self.beta))
        
        return result[0] if K.size == 1 else result

    def calibrate_atm(self, F, T, market_vol):
        """
        Calibrate to single ATM vol using standard formula
        alpha = market_vol * F^(1-beta)
        This formula works for all beta values (0, 0.5, 1)
        """
        F = float(F)
        T = float(T)
        
        # Standard formula for all beta values
        self.alpha = market_vol * (F**(1-self.beta))
        
        return True

    def calibrate_smile(self, F, K, T, market_vols):
        """Calibrate to full volatility smile with precise optimization"""
        def objective(params):
            self.alpha = params[0]
            self.sigma = params[1]
            self.rho = params[2]
            
            model_vols = self.implied_volatility(F, K, T)
            
            # Calculate relative error to handle different vol levels
            rel_diff = (model_vols - market_vols) / market_vols
            
            # Weight ATM points more heavily
            weights = np.ones_like(market_vols)
            atm_mask = np.abs(K - F) < 1e-10
            weights[atm_mask] = 5.0  # Higher weight for ATM
            
            # Add penalty for extreme parameter values
            param_penalty = (
                0.1 * max(0, abs(self.rho) - 0.9)**2 +  # Penalize rho near ±1
                0.1 * max(0, self.sigma - 1.0)**2 +     # Penalize large sigma
                0.1 * max(0, 0.001 - self.alpha)**2     # Penalize small alpha
            )
            
            return np.sum((rel_diff * weights)**2) + param_penalty
        
        # Calculate initial guesses based on market data
        atm_idx = np.argmin(np.abs(K - F))
        atm_vol = market_vols[atm_idx]
        vol_slope = (market_vols[-1] - market_vols[0]) / (K[-1] - K[0])
        
        initial_guesses = [
            # Base guess using ATM vol
            [atm_vol * (F**(1-self.beta)), 0.3, np.sign(vol_slope) * 0.3],
            # Alternative guesses
            [atm_vol * (F**(1-self.beta)), 0.5, 0.0],
            [atm_vol * (F**(1-self.beta)), 0.2, -0.5],
            [atm_vol * (F**(1-self.beta)), 0.2, 0.5]
        ]
        
        bounds = [
            (0.0001, None),    # alpha > 0
            (0.0001, 2.0),     # 0 < sigma < 2
            (-0.95, 0.95)      # -0.95 < rho < 0.95
        ]
        
        best_result = None
        best_error = np.inf
        
        for guess in initial_guesses:
            result = minimize(
                objective,
                guess,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'eps': 1e-8
                }
            )
            
            if result.fun < best_error:
                best_error = result.fun
                best_result = result
                
            # If we get a good enough fit, stop
            if best_error < 1e-10:
                break
        
        if best_result is not None:
            self.alpha = best_result.x[0]
            self.sigma = best_result.x[1]
            self.rho = best_result.x[2]
            
            # Fine-tune alpha to match ATM vol exactly
            model_vols = self.implied_volatility(F, K, T)
            atm_error = model_vols[atm_idx] - market_vols[atm_idx]
            self.alpha *= (1 - atm_error/model_vols[atm_idx])
        
        return best_result

    def calibrate(self, F, K, T, market_vols):
        """
        General calibration method that handles both ATM and smile cases
        """
        market_vols = np.asarray(market_vols)
        if market_vols.ndim == 1 and len(market_vols) == 1:
            # Single ATM vol case
            return self.calibrate_atm(F[0], T[0], market_vols[0])
        else:
            # Full smile case
            return self.calibrate_smile(F[0], K.flatten(), T[0], market_vols.flatten())