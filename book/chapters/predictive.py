import numpy as np
import statsmodels.api as sm

def calculate_out_of_sample_r2(fitted_model, test_data):
    """
    Calculate out-of-sample R-squared using only the fitted model and test data.
    
    Parameters:
    fitted_model: statsmodels fitted regression model
    test_data: pandas DataFrame with test observations
    
    Returns:
    R²: 1 - (SS_res / SS_tot)
    """
    try:
        # Extract model variables from the fitted model
        formula_terms = fitted_model.model.exog_names
        dependent_var = fitted_model.model.endog_names
        
        # Get predictor variables (exclude intercept)
        predictor_vars = [var for var in formula_terms if var != 'Intercept']
        
        # Create test subset with complete data for this model
        required_vars = predictor_vars + [dependent_var]
        test_subset = test_data[required_vars].dropna()
        
        if len(test_subset) == 0:
            return np.nan, 0
        
        # Generate predictions
        y_pred = fitted_model.predict(test_subset[predictor_vars])
        y_true = test_subset[dependent_var].values
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return np.nan, 0
        
        # Calculate R²
        ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
        ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
        
        if ss_tot == 0:
            return np.nan, len(y_true_clean)
        
        r2 = 1 - (ss_res / ss_tot)
        return r2, len(y_true_clean)
        
    except Exception as e:
        print(f"    Error in R² calculation: {e}")
        return np.nan, 0