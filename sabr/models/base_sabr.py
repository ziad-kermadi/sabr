from abc import ABC, abstractmethod
import numpy as np

class BaseSABR(ABC):
    def __init__(self, beta=0.5):
        self.beta = beta
        self.alpha = None
        self.rho = None
        self.sigma = None  # This is nu in some implementations
        
    @abstractmethod
    def implied_volatility(self, F, K, T):
        """Calculate implied volatility for given forward, strike and time"""
        pass