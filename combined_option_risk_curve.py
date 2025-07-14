#               > cd C:\Users\zhiro\Documents\Option Risk Curve
#               > combined_option_risk_curve.py


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

# --- Black-Scholes Option Pricing Model ---
def black_scholes(S, K, T, r, sigma, option_type):
    """
    Calculates the price of a European option using the Black-Scholes model.

    Args:
        S (float): Current price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to expiration in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Implied volatility of the underlying asset (annualized).
        option_type (str): 'call' for a call option, 'put' for a put option.

    Returns:
        float: The theoretical price of the option.
    """
    if T <= 0: # Handle options at or past expiration
        if option_type == 'call':
            return max(0, S - K)
        else: # put
            return max(0, K - S)

    # Avoid division by zero or log of non-positive numbers
    if sigma <= 0 or T <= 0:
        return 0.0 # Or handle as an error/specific case

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

# --- Function to Calculate P&L for a Single Option at a Future Date ---
def calculate_single_option_pnl_curve(
    hypothetical_underlying_prices,
    initial_underlying_price,
    strike_price,
    expiration_date_obj,
    current_date_obj, # This is the analysis start date, not necessarily option's current_date_str
    target_date_obj,  # This is the date for which we want the P&L curve
    implied_volatility,
    risk_free_rate,
    option_type,
    num_contracts,
    multiplier=100
):
    """
    Calculates the P&L curve for a single option at a specified target date.

    Args:
        hypothetical_underlying_prices (np.array): Range of underlying prices for the curve.
        initial_underlying_price (float): Current market price of the underlying asset.
        strike_price (float): Strike price of the option.
        expiration_date_obj (datetime.datetime): Expiration date of the option.
        current_date_obj (datetime.datetime): The date from which the option's initial premium is calculated.
        target_date_obj (datetime.datetime): The specific future date for which to calculate the P&L curve.
        implied_volatility (float): Annualized implied volatility.
        risk_free_rate (float): Annualized risk-free interest rate.
        option_type (str): 'call' or 'put'.
        num_contracts (int): Number of option contracts.
        multiplier (int): Multiplier for the option contract.

    Returns:
        np.array: Corresponding P&L values at the target date for this single option.
        float: Initial cost of this single option.
    """
    T_initial = (expiration_date_obj - current_date_obj).days / 365.0
    T_target = (expiration_date_obj - target_date_obj).days / 365.0

    # Ensure time to expiration is not negative for calculations
    if T_initial < 0:
        T_initial = 0.0
    if T_target < 0:
        T_target = 0.0

    initial_option_price = black_scholes(initial_underlying_price, strike_price, T_initial, risk_free_rate, implied_volatility, option_type)

    pnl_values = []
    for S_target in hypothetical_underlying_prices:
        future_option_price = black_scholes(S_target, strike_price, T_target, risk_free_rate, implied_volatility, option_type)
        pnl = (future_option_price - initial_option_price) * num_contracts * multiplier
        pnl_values.append(pnl)

    return np.array(pnl_values), initial_option_price * num_contracts * multiplier

# --- Function to Calculate P&L for a Stock Position ---
def calculate_stock_pnl_curve(
    hypothetical_underlying_prices,
    initial_stock_price,
    num_shares
):
    """
    Calculates the P&L curve for a stock position.

    Args:
        hypothetical_underlying_prices (np.array): Range of underlying prices for the curve.
        initial_stock_price (float): The price at which the stock was acquired.
        num_shares (int): Number of shares (positive for long, negative for short).

    Returns:
        np.array: Corresponding P&L values.
        float: Initial cost of the stock position.
    """
    pnl_values = []
    for S_future in hypothetical_underlying_prices:
        pnl = (S_future - initial_stock_price) * num_shares
        pnl_values.append(pnl)
    return np.array(pnl_values), initial_stock_price * num_shares

# --- Main execution block ---
if __name__ == "__main__":
    print("--- Combined Option and Stock Risk Curve Calculator ---")

    # Provided data for 'DVN' stock and options, parsed and populated
    # Note: Risk-Free Rate is assumed to be 0.05 (5%) as it was not provided in your data.
    # 'future_date_str' for options are set to a reasonable date before their expiration.
    dvn_portfolio_data = [
        {
            "local_symbol": "DVN",
            "sec_type": "STK",
            "current_underlying_price": 33.95, # From 'Option Close' or 'Underlying Close'
            "initial_stock_price": 33.95, # Assuming acquired at current price for P&L calculation from today
            "num_shares": 100
        },
        {
            "local_symbol": "DVN 250718C00040000",
            "sec_type": "OPT",
            "current_underlying_price": 33.97, # From 'Underlying Close'
            "strike_price": 40.0,
            "expiration_date_str": "2025-07-18",
            "current_date_str": "2025-07-12",
            "future_date_str": "2025-07-17", # 1 day before expiry
            "implied_volatility": 0.4330, # 43.3%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "call",
            "num_contracts": -2,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 250815P00032500",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 32.5,
            "expiration_date_str": "2025-08-15",
            "current_date_str": "2025-07-12",
            "future_date_str": "2025-08-14", # 1 day before expiry
            "implied_volatility": 0.3791, # 37.91%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -4,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 250815P00035000",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 35.0,
            "expiration_date_str": "2025-08-15",
            "current_date_str": "2025-07-12",
            "future_date_str": "2025-08-14", # 1 day before expiry
            "implied_volatility": 0.3633, # 36.33%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -4,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 251017P00035000",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 35.0,
            "expiration_date_str": "2025-10-17",
            "current_date_str": "2025-07-12",
            "future_date_str": "2025-10-16", # 1 day before expiry
            "implied_volatility": 0.3422, # 34.22%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -3,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 260116P00032500",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 32.5,
            "expiration_date_str": "2026-01-16",
            "current_date_str": "2025-07-12",
            "future_date_str": "2026-01-15", # 1 day before expiry
            "implied_volatility": 0.3595, # 35.95%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -4,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 260116P00047500",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 47.5,
            "expiration_date_str": "2026-01-16",
            "current_date_str": "2025-07-12",
            "future_date_str": "2026-01-15", # 1 day before expiry
            "implied_volatility": 0.3289, # 32.89%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -1,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 260618C00045000",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 45.0,
            "expiration_date_str": "2026-06-18",
            "current_date_str": "2025-07-12",
            "future_date_str": "2026-06-17", # 1 day before expiry
            "implied_volatility": 0.3181, # 31.81%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "call",
            "num_contracts": -2,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 270115P00032500",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 32.5,
            "expiration_date_str": "2027-01-15",
            "current_date_str": "2025-07-12",
            "future_date_str": "2027-01-14", # 1 day before expiry
            "implied_volatility": 0.3463, # 34.63%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -4,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 270115C00040000",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 40.0,
            "expiration_date_str": "2027-01-15",
            "current_date_str": "2025-07-12",
            "future_date_str": "2027-01-14", # 1 day before expiry
            "implied_volatility": 0.3268, # 32.68%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "call",
            "num_contracts": -2,
            "multiplier": 100
        },
        {
            "local_symbol": "DVN 270115P00040000",
            "sec_type": "OPT",
            "current_underlying_price": 33.97,
            "strike_price": 40.0,
            "expiration_date_str": "2027-01-15",
            "current_date_str": "2025-07-12",
            "future_date_str": "2027-01-14", # 1 day before expiry
            "implied_volatility": 0.3268, # 32.68%
            "risk_free_rate": 0.05, # Assumed
            "option_type": "put",
            "num_contracts": -2,
            "multiplier": 100
        }
    ]

    # Set the analysis start date to today's date
    analysis_current_date_obj = datetime.datetime.now()
    analysis_current_date_str = analysis_current_date_obj.strftime('%Y-%m-%d')

    # Find the earliest option expiration date and its strike for the second curve and SD calculation
    first_option_expiration_date_obj = None
    first_option_sigma_for_sd = None
    initial_underlying_price_for_sd = None
    first_option_strike_price = None # Store the strike price of the first expiring option

    options_only = [item for item in dvn_portfolio_data if item["sec_type"] == "OPT"]
    if options_only:
        # Sort options by expiration date to find the earliest
        options_only.sort(key=lambda x: datetime.datetime.strptime(x["expiration_date_str"], '%Y-%m-%d'))
        first_option_to_expire = options_only[0]
        
        first_option_expiration_date_str = first_option_to_expire["expiration_date_str"]
        first_option_expiration_date_obj = datetime.datetime.strptime(first_option_expiration_date_str, '%Y-%m-%d')
        first_option_sigma_for_sd = first_option_to_expire["implied_volatility"]
        initial_underlying_price_for_sd = first_option_to_expire["current_underlying_price"] # Assuming this is representative
        first_option_strike_price = first_option_to_expire["strike_price"] # Get the strike price

    if first_option_expiration_date_obj is None:
        print("Warning: No option found in the portfolio. Only plotting 'as of today'. SD shading will not be applied.")
        # If no options, set a dummy date far in future so only one curve is plotted
        first_option_expiration_date_obj = analysis_current_date_obj + datetime.timedelta(days=3650) 
        first_option_expiration_date_str = first_option_expiration_date_obj.strftime('%Y-%m-%d')


    # Determine overall min/max for hypothetical underlying prices across all positions
    all_current_prices = [item["current_underlying_price"] for item in dvn_portfolio_data]
    all_strike_prices = [item["strike_price"] for item in dvn_portfolio_data if item["sec_type"] == "OPT"]

    # Ensure the price range is meaningful even if only stocks are present or strikes are limited
    if not all_strike_prices: # If no options, base range solely on stock price
        overall_price_range_min = min(all_current_prices) * 0.8
        overall_price_range_max = max(all_current_prices) * 1.2
    else:
        overall_price_range_min = min(all_current_prices + all_strike_prices) * 0.8
        overall_price_range_max = max(all_current_prices + all_strike_prices) * 1.2

    hypothetical_underlying_prices = np.linspace(overall_price_range_min, overall_price_range_max, 200)

    # Initialize combined P&L arrays for both curves
    combined_pnl_as_of_today = np.zeros_like(hypothetical_underlying_prices)
    combined_pnl_at_first_option_expiry = np.zeros_like(hypothetical_underlying_prices)
    
    total_initial_cost_at_analysis_start = 0

    print(f"\nCalculating P&L for each position and combining (Analysis Date: {analysis_current_date_str})...\n")

    for i, position_params in enumerate(dvn_portfolio_data):
        try:
            if position_params["sec_type"] == "OPT":
                expiration_date_obj = datetime.datetime.strptime(position_params["expiration_date_str"], '%Y-%m-%d')
                
                # Skip options that are already expired relative to the analysis start date
                if analysis_current_date_obj > expiration_date_obj:
                     print(f"Skipping Option {i+1} ({position_params['local_symbol']}): Analysis date is past its expiration.")
                     continue

                # Calculate P&L for the first curve (as of today)
                pnl_today, initial_cost_today = calculate_single_option_pnl_curve(
                    hypothetical_underlying_prices,
                    position_params["current_underlying_price"],
                    position_params["strike_price"],
                    expiration_date_obj,
                    analysis_current_date_obj, # Use the common analysis start date
                    analysis_current_date_obj, # Target date is also today for this curve
                    position_params["implied_volatility"],
                    position_params["risk_free_rate"],
                    position_params["option_type"],
                    position_params["num_contracts"],
                    position_params["multiplier"]
                )
                combined_pnl_as_of_today += pnl_today
                total_initial_cost_at_analysis_start += initial_cost_today

                # Calculate P&L for the second curve (at the first option's expiration date)
                # Only include options that are still active at the first option's expiration date
                if first_option_expiration_date_obj <= expiration_date_obj:
                    pnl_first_expiry, _ = calculate_single_option_pnl_curve( # Initial cost already added
                        hypothetical_underlying_prices,
                        position_params["current_underlying_price"],
                        position_params["strike_price"],
                        expiration_date_obj,
                        analysis_current_date_obj, # Use the common analysis start date
                        first_option_expiration_date_obj,
                        position_params["implied_volatility"],
                        position_params["risk_free_rate"],
                        position_params["option_type"],
                        position_params["num_contracts"],
                        position_params["multiplier"]
                    )
                    combined_pnl_at_first_option_expiry += pnl_first_expiry
                else:
                    print(f"  - Option {i+1} ({position_params['local_symbol']}): Will be expired by {first_option_expiration_date_str}, not included in that curve.")

                print(f"  - Option {i+1} ({position_params['local_symbol']}): P&L calculated for both curves.")

            elif position_params["sec_type"] == "STK":
                # For stocks, P&L doesn't change with time or implied volatility
                # It's simply (Future_Price - Initial_Price) * Num_Shares
                pnl_stock, initial_cost_stock = calculate_stock_pnl_curve(
                    hypothetical_underlying_prices,
                    position_params["initial_stock_price"],
                    position_params["num_shares"]
                )
                combined_pnl_as_of_today += pnl_stock
                combined_pnl_at_first_option_expiry += pnl_stock # Stock P&L is the same regardless of future date
                total_initial_cost_at_analysis_start += initial_cost_stock
                print(f"  - Stock {i+1} ({position_params['local_symbol']}): P&L calculated for both curves.")

            else:
                print(f"Warning: Unknown security type for {position_params['local_symbol']}. Skipping.")

        except ValueError as e:
            print(f"Error processing position {i+1} ({position_params['local_symbol']}): {e}")
        except Exception as e:
            print(f"An unexpected error occurred for position {i+1} ({position_params['local_symbol']}): {e}")


    print(f"\nTotal Initial Cost of Portfolio (at {analysis_current_date_str}): ${total_initial_cost_at_analysis_start:.2f}\n")

    # Plotting the combined P&L curves
    plt.figure(figsize=(14, 8))
    
    # --- Add SD Shading ---
    if first_option_expiration_date_obj and first_option_sigma_for_sd is not None and initial_underlying_price_for_sd is not None:
        T_sd = (first_option_expiration_date_obj - analysis_current_date_obj).days / 365.0
        if T_sd > 0 and first_option_sigma_for_sd > 0:
            log_return_sd = first_option_sigma_for_sd * math.sqrt(T_sd)

            # Calculate price boundaries for +/- 1, 2, 3 standard deviations
            # These represent the approximate price ranges within which the underlying is expected to fall
            # by the expiration of the first option, based on its implied volatility.
            
            # 3 Standard Deviations (outermost)
            lower_3sd = initial_underlying_price_for_sd * math.exp(-3 * log_return_sd)
            upper_3sd = initial_underlying_price_for_sd * math.exp(3 * log_return_sd)

            # 2 Standard Deviations
            lower_2sd = initial_underlying_price_for_sd * math.exp(-2 * log_return_sd)
            upper_2sd = initial_underlying_price_for_sd * math.exp(2 * log_return_sd)

            # 1 Standard Deviation (innermost)
            lower_1sd = initial_underlying_price_for_sd * math.exp(-1 * log_return_sd)
            upper_1sd = initial_underlying_price_for_sd * math.exp(1 * log_return_sd)

            # Add shading to the plot (from lightest to darkest)
            plt.axvspan(lower_3sd, upper_3sd, color='skyblue', alpha=0.15, label='±3 SD Range (First Expiry)')
            plt.axvspan(lower_2sd, upper_2sd, color='lightgreen', alpha=0.25, label='±2 SD Range')
            plt.axvspan(lower_1sd, upper_1sd, color='lightcoral', alpha=0.35, label='±1 SD Range')
            print(f"✅ SD Ranges calculated and added based on first option expiry ({first_option_expiration_date_str}).")
            print(f"   1 SD Range: ${lower_1sd:.2f} - ${upper_1sd:.2f}")
            print(f"   2 SD Range: ${lower_2sd:.2f} - ${upper_2sd:.2f}")
            print(f"   3 SD Range: ${lower_3sd:.2f} - ${upper_3sd:.2f}")
        else:
            print("Warning: Cannot calculate SD ranges. First option expired or has zero volatility.")

    # Swapped line types and colors for P&L curves
    plt.plot(hypothetical_underlying_prices, combined_pnl_as_of_today, label=f'P&L as of Today ({analysis_current_date_str})', color='orange', linestyle='--', linewidth=2)
    
    # Only plot the second curve if it's a meaningful date (i.e., different from today)
    if first_option_expiration_date_obj > analysis_current_date_obj:
        plt.plot(hypothetical_underlying_prices, combined_pnl_at_first_option_expiry, label=f'P&L at {first_option_expiration_date_str} (First Option Expiry)', color='purple', linestyle='-', linewidth=2)


    # Add initial underlying price line (assuming it's the same for all positions)
    # Using the first position's current underlying price as a reference
    initial_S_ref = dvn_portfolio_data[0]["current_underlying_price"]
    plt.axvline(x=initial_S_ref, color='gray', linestyle='--', label=f'Current Underlying Price (${initial_S_ref})')

    # Add individual strike prices for reference
    for item in dvn_portfolio_data:
        if item["sec_type"] == "OPT":
            # Determine color for strike line: red if it's the first expiring option's strike, else green
            line_color = 'red' if item["strike_price"] == first_option_strike_price and item["expiration_date_str"] == first_option_expiration_date_str else 'green'

            # Include position and expiration date in the label
            label_text = f'Strike: ${item["strike_price"]} ({item["option_type"].upper()}) (Pos: {item["num_contracts"]}) (Exp: {item["expiration_date_str"]})'
            # Check if label already exists to avoid duplicates in legend
            if label_text not in plt.gca().get_legend_handles_labels()[1]:
                plt.axvline(x=item["strike_price"], color=line_color, linestyle=':', alpha=0.6, label=label_text)
            else:
                plt.axvline(x=item["strike_price"], color=line_color, linestyle=':', alpha=0.6)
        elif item["sec_type"] == "STK":
            # Add label for stock position
            label_text = f'Stock (Pos: {item["num_shares"]} shares)'
            # Check if label already exists to avoid duplicates in legend
            if label_text not in plt.gca().get_legend_handles_labels()[1]:
                plt.axvline(x=item["initial_stock_price"], color='blue', linestyle='--', alpha=0.7, label=label_text)
            else:
                plt.axvline(x=item["initial_stock_price"], color='blue', linestyle='--', alpha=0.7)


    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8) # Zero P&L line
    plt.title('Combined Option and Stock Portfolio P&L Curves (DVN Positions)')
    plt.xlabel('Underlying Price at Future Date')
    plt.ylabel('Combined Profit / Loss ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
