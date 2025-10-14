import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from itertools import combinations


def pairwise_cointegration_check(instrument_1, instrument_2, p_thresh=0.05):
    """
    Perform cointegration test with proper time alignment using masking
    
    Parameters:
    -----------
    instrument_1, instrument_2 : array-like
        Time series data (should be time-aligned)
    p_thresh : float
        P-value threshold for significance
        
    Returns:
    --------
    hedge_ratio : float
        Hedge ratio if cointegrated, 0 otherwise
    """
    # Ensure both arrays are numpy arrays
    instrument_1 = np.array(instrument_1)
    instrument_2 = np.array(instrument_2)
    
    # Check if arrays have the same length
    if len(instrument_1) != len(instrument_2):
        print(f"Error: Arrays have different lengths: {len(instrument_1)} vs {len(instrument_2)}")
        return 0
    
    # Create mask for valid (non-NaN, non-infinite) data points
    mask = ~(np.isnan(instrument_1) | np.isnan(instrument_2) | 
             np.isinf(instrument_1) | np.isinf(instrument_2))
    
    # Apply mask to get valid data points
    valid_1 = instrument_1[mask]
    valid_2 = instrument_2[mask]
    
    # Check for sufficient valid data points (need at least 10 for reliable cointegration)
    if len(valid_1) < 10:
        print(f"Error: Insufficient valid data points ({len(valid_1)}) for cointegration test")
        return 0
    
    # Check if we lost too much data
    data_loss = (len(instrument_1) - len(valid_1)) / len(instrument_1)
    if data_loss > 0.5:  # More than 50% data loss
        print(f"Warning: High data loss ({data_loss:.1%}), results may be unreliable")
    
    try:
        # Perform cointegration test on valid data points
        t_stat, p_val, crit_val = coint(valid_1, valid_2)
        
        if p_val < p_thresh:
            # Calculate hedge ratio using OLS regression
            valid_1_const = sm.add_constant(valid_1)
            model = sm.OLS(valid_2, valid_1_const).fit()
            hedge_ratio = model.params[1]  # the slope
            print(f"Cointegrated! p-value: {p_val:.4f}, hedge ratio: {hedge_ratio:.4f}, valid points: {len(valid_1)}")
            return hedge_ratio
        else:
            return 0
            
    except Exception as e:
        print(f"Error in cointegration test: {e}")
        return 0


def pairwise_cointegration_check_time_aligned(instrument_1, instrument_2, timestamps, p_thresh=0.05):
    """
    Perform cointegration test with explicit time alignment
    
    Parameters:
    -----------
    instrument_1, instrument_2 : array-like
        Time series data aligned to common timestamps
    timestamps : pd.DatetimeIndex
        Common timestamps for both series
    p_thresh : float
        P-value threshold for significance
        
    Returns:
    --------
    hedge_ratio : float
        Hedge ratio if cointegrated, 0 otherwise
    """
    return pairwise_cointegration_check(instrument_1, instrument_2, p_thresh)
    


def find_all_pairs(inst_arrays,p_thresh=0.05):
    """
    inst_array: a list of numpy arrays. Each array is a different instrument's time series (all series are time-aligned)
    Output: a matrix of hedge-ratios
    """
    pairs = combinations(inst_arrays,r=2)
    ratios = [pairwise_cointegration_check(pair[0],pair[1]) for pair in pairs]
    return pairs, ratios



def geometric_walk(n_steps=100, start_price=1.0, mu=0.0, sigma=0.1, seed=None):
    """
    Generate a random geometric walk (geometric Brownian motion).

    Parameters:
    - n_steps: int, number of steps in the walk
    - start_price: float, initial value of the walk
    - mu: float, expected return (drift)
    - sigma: float, volatility (standard deviation of returns)
    - seed: int or None, random seed for reproducibility

    Returns:
    - numpy array of length n_steps representing the walk
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate standard normal random variables for each step
    Z = np.random.randn(n_steps)
    
    # Compute log returns
    log_returns = mu + sigma * Z
    
    # Convert log returns to prices
    walk = start_price * np.exp(np.cumsum(log_returns))
    
    return walk