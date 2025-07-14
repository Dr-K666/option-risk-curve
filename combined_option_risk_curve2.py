import os
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm

# --- Helper Function to Clean Data ---
def clean(val):
    """
    Cleans and converts a value to float.
    Returns 0.0 if conversion fails or value is None/empty/NaN.
    """
    if val is None or (isinstance(val, str) and val.strip() == ''):
        return 0.0
    try:
        # Handle cases where 'IV (%)' might be a string like "29.64"
        if isinstance(val, str) and '%' in val:
            return float(val.replace('%', '').strip()) / 100
        return float(val)
    except (ValueError, TypeError):
        return 0.0
    if isinstance(val, float) and math.isnan(val):
        return 0.0
    return val # Should already be float if it reached here without error


# --- Black-Scholes Option Pricing Model ---
def black_scholes(S, K, T, r, sigma, option_type):
    """
    Calculates the price of a European option using the Black-Scholes model.
    """
    if T <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    if sigma <= 0: # Volatility must be positive
        return 0.0
    if T <= 0: # Time must be positive for log and sqrt
        return 0.0

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
    current_date_obj,
    target_date_obj,
    implied_volatility,
    risk_free_rate,
    option_type,
    num_contracts,
    multiplier=100
):
    """Calculates the P&L curve for a single option at a specified target date."""
    T_initial = (expiration_date_obj - current_date_obj).days / 365.0
    T_target = (expiration_date_obj - target_date_obj).days / 365.0

    if T_initial < 0: T_initial = 0.0
    if T_target < 0: T_target = 0.0

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
    """Calculates the P&L curve for a stock position."""
    pnl_values = []
    for S_future in hypothetical_underlying_prices:
        pnl = (S_future - initial_stock_price) * num_shares
        pnl_values.append(pnl)
    return np.array(pnl_values), initial_stock_price * num_shares

# --- Main function to generate the risk curve ---
def generate_risk_curve(
    selected_symbol: str,
    portfolio_data: list, # This will be the list of dictionaries from Google Sheet
    analysis_date_str: str = None, # Optional: override today's date for analysis
    output_dir: str = "/tmp" # Use /tmp for Render's ephemeral storage
) -> str:
    """
    Generates and saves the combined option and stock risk curve for a given symbol.

    Args:
        selected_symbol (str): The underlying stock symbol to analyze.
        portfolio_data (list): A list of dictionaries, where each dict represents a position
                                (row from Google Sheet).
        analysis_date_str (str, optional): The date from which to start the analysis (YYYY-MM-DD).
                                            Defaults to today's date.
        output_dir (str, optional): Directory to save the generated plot. Defaults to "/tmp".

    Returns:
        str: The file path to the saved PNG image of the risk curve.
    """
    print(f"--- Generating Risk Curve for {selected_symbol} ---")

    # Set the analysis start date
    if analysis_date_str:
        analysis_current_date_obj = datetime.datetime.strptime(analysis_date_str, '%Y-%m-%d')
    else:
        analysis_current_date_obj = datetime.datetime.now()
    analysis_current_date_str = analysis_current_date_obj.strftime('%Y-%m-%d')

    # --- Parse and filter data from the provided portfolio_data list ---
    parsed_portfolio_data = []
    
    # Use a set to track unique strike labels to avoid duplicates in legend
    legend_strike_labels = set()

    for i, record in enumerate(portfolio_data):
        # Ensure keys match your Google Sheet headers
        # Use .get() with a default to avoid KeyError if a column is truly missing
        
        symbol_from_record = str(record.get('Symbol', '')).strip().upper()
        sec_type = str(record.get('Security Type', '')).strip().upper()

        if symbol_from_record != selected_symbol.upper(): # Filter here
            continue

        try:
            item = {
                "local_symbol": str(record.get('Local Symbol', '')),
                "symbol": symbol_from_record,
                "sec_type": sec_type,
                "current_underlying_price": clean(record.get('Underlying Close')),
                "position": clean(record.get('Position'))
            }

            if sec_type == 'OPT':
                item.update({
                    "strike_price": clean(record.get('Strike')),
                    "expiration_date_str": str(record.get('Expiry')), # Keep as YYYYMMDD string from sheet
                    "current_date_str": str(record.get('Date')), # Keep as YYYY-MM-DD string from sheet
                    "implied_volatility": clean(record.get('IV (%)')) / 100 if record.get('IV (%)') else 0.0,
                    "risk_free_rate": 0.05, # Assumed, or pass as arg
                    "option_type": str(record.get('Right', '')).lower(),
                    "multiplier": clean(record.get('Multiplier')) if record.get('Multiplier') else 100
                })
                # Set future_date_str for individual option P&L curve calculation
                exp_date_obj = datetime.datetime.strptime(item["expiration_date_str"], '%Y%m%d')
                item["future_date_str"] = (exp_date_obj - datetime.timedelta(days=1)).strftime('%Y-%m-%d') # Default to 1 day before expiry

            elif sec_type == 'STK':
                item.update({
                    "initial_stock_price": clean(record.get('Option Close')) if record.get('Option Close') else clean(record.get('Underlying Close')),
                    "num_shares": clean(record.get('Position'))
                })
            
            parsed_portfolio_data.append(item)

        except Exception as ex:
            print(f"⚠️ Error parsing record ({record.get('Local Symbol', 'N/A')}): {ex}")

    if not parsed_portfolio_data:
        raise ValueError(f"No valid positions found for symbol '{selected_symbol}' after parsing.")

    # Find the earliest option expiration date and its strike for the second curve and SD calculation
    first_option_expiration_date_obj = None
    first_option_sigma_for_sd = None
    initial_underlying_price_for_sd = None
    first_option_strike_price = None

    options_only_in_filtered = [item for item in parsed_portfolio_data if item["sec_type"] == "OPT"]
    if options_only_in_filtered:
        options_only_in_filtered.sort(key=lambda x: datetime.datetime.strptime(x["expiration_date_str"], '%Y%m%d'))
        first_option_to_expire = options_only_in_filtered[0]
        
        first_option_expiration_date_str_raw = first_option_to_expire["expiration_date_str"]
        first_option_expiration_date_obj = datetime.datetime.strptime(first_option_expiration_date_str_raw, '%Y%m%d')
        first_option_sigma_for_sd = first_option_to_expire["implied_volatility"]
        initial_underlying_price_for_sd = first_option_to_expire["current_underlying_price"]
        first_option_strike_price = first_option_to_expire["strike_price"]

    if first_option_expiration_date_obj is None:
        print(f"Warning: No option found for '{selected_symbol}'. Only plotting 'as of today'. SD shading will not be applied.")
        first_option_expiration_date_obj = analysis_current_date_obj + datetime.timedelta(days=3650) # Set a far future date to effectively disable second curve
        first_option_expiration_date_str_raw = first_option_expiration_date_obj.strftime('%Y%m%d') # For label consistency


    # Determine overall min/max for hypothetical underlying prices across filtered positions
    all_current_prices_filtered = [item["current_underlying_price"] for item in parsed_portfolio_data]
    all_strike_prices_filtered = [item["strike_price"] for item in parsed_portfolio_data if item["sec_type"] == "OPT"]

    if not all_strike_prices_filtered:
        if not all_current_prices_filtered: # Handle case with no data at all
            raise ValueError("No price data found to determine plot range.")
        overall_price_range_min = min(all_current_prices_filtered) * 0.8
        overall_price_range_max = max(all_current_prices_filtered) * 1.2
    else:
        overall_price_range_min = min(all_current_prices_filtered + all_strike_prices_filtered) * 0.8
        overall_price_range_max = max(all_current_prices_filtered + all_strike_prices_filtered) * 1.2

    hypothetical_underlying_prices = np.linspace(overall_price_range_min, overall_price_range_max, 200)

    # Initialize combined P&L arrays for both curves
    combined_pnl_as_of_today = np.zeros_like(hypothetical_underlying_prices)
    combined_pnl_at_first_option_expiry = np.zeros_like(hypothetical_underlying_prices)
    
    total_initial_cost_at_analysis_start = 0

    print(f"\nCalculating P&L for '{selected_symbol}' positions (Analysis Date: {analysis_current_date_str})...\n")

    for i, position_params in enumerate(parsed_portfolio_data):
        try:
            # Parse current_date_str from the position_params for initial premium calculation
            # This ensures the initial premium calculation uses the date the data was recorded
            position_current_date_obj = datetime.datetime.strptime(position_params["current_date_str"], '%Y-%m-%d')

            if position_params["sec_type"] == "OPT":
                expiration_date_obj = datetime.datetime.strptime(position_params["expiration_date_str"], '%Y%m%d') # Use YYYYMMDD for sheet expiry
                
                if analysis_current_date_obj > expiration_date_obj:
                     print(f"Skipping Option {i+1} ({position_params['local_symbol']}): Analysis date is past its expiration.")
                     continue

                pnl_today, initial_cost_today = calculate_single_option_pnl_curve(
                    hypothetical_underlying_prices,
                    position_params["current_underlying_price"],
                    position_params["strike_price"],
                    expiration_date_obj,
                    position_current_date_obj, # Use the position's recorded current date for initial cost
                    analysis_current_date_obj, # Target date is today for this curve
                    position_params["implied_volatility"],
                    position_params["risk_free_rate"],
                    position_params["option_type"],
                    position_params["position"],
                    position_params["multiplier"]
                )
                combined_pnl_as_of_today += pnl_today
                total_initial_cost_at_analysis_start += initial_cost_today

                if first_option_expiration_date_obj <= expiration_date_obj:
                    pnl_first_expiry, _ = calculate_single_option_pnl_curve(
                        hypothetical_underlying_prices,
                        position_params["current_underlying_price"],
                        position_params["strike_price"],
                        expiration_date_obj,
                        position_current_date_obj, # Use the position's recorded current date for initial cost
                        first_option_expiration_date_obj,
                        position_params["implied_volatility"],
                        position_params["risk_free_rate"],
                        position_params["option_type"],
                        position_params["position"],
                        position_params["multiplier"]
                    )
                    combined_pnl_at_first_option_expiry += pnl_first_expiry
                else:
                    print(f"  - Option {i+1} ({position_params['local_symbol']}): Will be expired by {first_option_expiration_date_str_raw}, not included in that curve.")

                print(f"  - Option {i+1} ({position_params['local_symbol']}): P&L calculated for both curves.")

            elif position_params["sec_type"] == "STK":
                pnl_stock, initial_cost_stock = calculate_stock_pnl_curve(
                    hypothetical_underlying_prices,
                    position_params["initial_stock_price"],
                    position_params["position"]
                )
                combined_pnl_as_of_today += pnl_stock
                combined_pnl_at_first_option_expiry += pnl_stock
                total_initial_cost_at_analysis_start += initial_stock_stock
                print(f"  - Stock {i+1} ({position_params['local_symbol']}): P&L calculated for both curves.")

            else:
                print(f"Warning: Unknown security type for {position_params['local_symbol']}. Skipping.")

        except KeyError as ke:
            print(f"Error processing position {i+1} ({position_params.get('local_symbol', 'N/A')}) due to missing key: {ke}. Check sheet headers.")
        except ValueError as ve:
            print(f"Error processing position {i+1} ({position_params.get('local_symbol', 'N/A')}) due to data type conversion: {ve}. Check data format in sheet.")
        except Exception as e:
            print(f"An unexpected error occurred for position {i+1} ({position_params.get('local_symbol', 'N/A')}): {e}")


    print(f"\nTotal Initial Cost of Portfolio (at {analysis_current_date_str}): ${total_initial_cost_at_analysis_start:.2f}\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_image_filename = f"{selected_symbol}_risk_curve_{analysis_current_date_obj.strftime('%Y%m%d_%H%M%S')}.png"
    output_image_filepath = os.path.join(output_dir, output_image_filename)

    # Plotting the combined P&L curves
    plt.figure(figsize=(14, 8))
    
    # --- Add SD Shading ---
    if first_option_expiration_date_obj and first_option_sigma_for_sd is not None and initial_underlying_price_for_sd is not None:
        T_sd = (first_option_expiration_date_obj - analysis_current_date_obj).days / 365.0
        if T_sd > 0 and first_option_sigma_for_sd > 0:
            log_return_sd = first_option_sigma_for_sd * math.sqrt(T_sd)

            lower_3sd = initial_underlying_price_for_sd * math.exp(-3 * log_return_sd)
            upper_3sd = initial_underlying_price_for_sd * math.exp(3 * log_return_sd)
            lower_2sd = initial_underlying_price_for_sd * math.exp(-2 * log_return_sd)
            upper_2sd = initial_underlying_price_for_sd * math.exp(2 * log_return_sd)
            lower_1sd = initial_underlying_price_for_sd * math.exp(-1 * log_return_sd)
            upper_1sd = initial_underlying_price_for_sd * math.exp(1 * log_return_sd)

            plt.axvspan(lower_3sd, upper_3sd, color='skyblue', alpha=0.15, label='±3 SD Range (First Expiry)')
            plt.axvspan(lower_2sd, upper_2sd, color='lightgreen', alpha=0.25, label='±2 SD Range')
            plt.axvspan(lower_1sd, upper_1sd, color='lightcoral', alpha=0.35, label='±1 SD Range')
            print(f"✅ SD Ranges calculated and added based on first option expiry ({first_option_expiration_date_str_raw}).")
            print(f"   1 SD Range: ${lower_1sd:.2f} - ${upper_1sd:.2f}")
            print(f"   2 SD Range: ${lower_2sd:.2f} - ${upper_2sd:.2f}")
            print(f"   3 SD Range: ${lower_3sd:.2f} - ${upper_3sd:.2f}")
        else:
            print("Warning: Cannot calculate SD ranges. First option expired or has zero volatility.")

    plt.plot(hypothetical_underlying_prices, combined_pnl_as_of_today, label=f'P&L as of Today ({analysis_current_date_str})', color='orange', linestyle='--', linewidth=2)
    
    if first_option_expiration_date_obj > analysis_current_date_obj:
        plt.plot(hypothetical_underlying_prices, combined_pnl_at_first_option_expiry, label=f'P&L at {first_option_expiration_date_str_raw} (First Option Expiry)', color='purple', linestyle='-', linewidth=2)


    initial_S_ref = parsed_portfolio_data[0]["current_underlying_price"]
    plt.axvline(x=initial_S_ref, color='gray', linestyle='--', label=f'Current Underlying Price (${initial_S_ref})')

    for item in parsed_portfolio_data:
        if item["sec_type"] == "OPT":
            is_first_expiring_option = (
                item["strike_price"] == first_option_strike_price and
                item["expiration_date_str"] == first_option_expiration_date_str_raw # Compare string formats
            )
            line_color = 'red' if is_first_expiring_option else 'green'

            label_text = f'Strike: ${item["strike_price"]} ({item["option_type"].upper()}) (Pos: {item["position"]}) (Exp: {item["expiration_date_str"]})'
            if label_text not in plt.gca().get_legend_handles_labels()[1]:
                plt.axvline(x=item["strike_price"], color=line_color, linestyle=':', alpha=0.6, label=label_text)
            else:
                plt.axvline(x=item["strike_price"], color=line_color, linestyle=':', alpha=0.6)
        elif item["sec_type"] == "STK":
            label_text = f'Stock (Pos: {item["position"]} shares)'
            if label_text not in plt.gca().get_legend_handles_labels()[1]:
                plt.axvline(x=item["initial_stock_price"], color='blue', linestyle='--', alpha=0.7, label=label_text)
            else:
                plt.axvline(x=item["initial_stock_price"], color='blue', linestyle='--', alpha=0.7)


    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.title(f'Combined Option and Stock Portfolio P&L Curves ({selected_symbol} Positions)')
    plt.xlabel('Underlying Price at Future Date')
    plt.ylabel('Combined Profit / Loss ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image_filepath) # Save the figure
    plt.close() # Close the figure to free up memory

    return output_image_filepath # Return the path to the saved image

# --- Local Testing Example (This block will NOT be used by FastAPI/n8n) ---
if __name__ == "__main__":
    print("--- Running local test of generate_risk_curve function ---")
    
    # Dummy portfolio data for local testing (replace with actual data from your sheet)
    # This data structure must match what gspread.get_all_records() returns
    # and what your parsing logic expects.
    test_portfolio_data = [
        {
            "Date": "2025-07-12", "Local Symbol": "B", "Symbol": "B", "Expiry": "", "Strike": "", "Right": "", "Currency": "USD", "Exchange": "SMART", "Position": 300, "Security Type": "STK", "Multiplier": 1, "Delta": "", "Gamma": "", "Vega": "", "Theta": "", "IV (%)": "", "Option Close": 21.07, "Underlying Close": 21.07, "Days to Maturity": "", "Market Value": 6363.0, "Intrinsic Value": "", "Time Value": "", "Daily Theta ($)": "", "Effective Shares": 300, "Delta Dollars": 6363
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 250718C00021000", "Symbol": "B", "Expiry": "20250718", "Strike": 21.0, "Right": "C", "Currency": "USD", "Exchange": "SMART", "Position": -3, "Security Type": "OPT", "Multiplier": 100, "Delta": 0.607, "Gamma": 0.4441, "Vega": 0.0111, "Theta": -0.0259, "IV (%)": 29.64, "Option Close": 0.4, "Underlying Close": 21.22, "Days to Maturity": 6, "Market Value": -141.45, "Intrinsic Value": 66.0, "Time Value": 75.45, "Daily Theta ($)": 7.78, "Effective Shares": -182.11, "Delta Dollars": -3864.37
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 250725P00021500", "Symbol": "B", "Expiry": "20250725", "Strike": 21.5, "Right": "P", "Currency": "USD", "Exchange": "SMART", "Position": -3, "Security Type": "OPT", "Multiplier": 100, "Delta": -0.5785, "Gamma": 0.3278, "Vega": 0.016, "Theta": -0.0158, "IV (%)": 29.3, "Option Close": 0.73, "Underlying Close": 21.22, "Days to Maturity": 13, "Market Value": -188.8, "Intrinsic Value": 84.0, "Time Value": 104.8, "Daily Theta ($)": 4.75, "Effective Shares": 173.54, "Delta Dollars": 3682.52
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 250919P00019000", "Symbol": "B", "Expiry": "20250919", "Strike": 19.0, "Right": "P", "Currency": "USD", "Exchange": "SMART", "Position": -3, "Security Type": "OPT", "Multiplier": 100, "Delta": -0.1896, "Gamma": 0.0905, "Vega": 0.0262, "Theta": -0.0056, "IV (%)": 32.6, "Option Close": 0.36, "Underlying Close": 21.22, "Days to Maturity": 69, "Market Value": -100.83, "Intrinsic Value": 0.0, "Time Value": 100.83, "Daily Theta ($)": 1.68, "Effective Shares": 56.87, "Delta Dollars": 1206.78
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 251219C00022000", "Symbol": "B", "Expiry": "20251219", "Strike": 22.0, "Right": "C", "Currency": "USD", "Exchange": "SMART", "Position": -2, "Security Type": "OPT", "Multiplier": 100, "Delta": 0.4992, "Gamma": 0.0896, "Vega": 0.0563, "Theta": -0.0066, "IV (%)": 31.87, "Option Close": 1.51, "Underlying Close": 21.22, "Days to Maturity": 160, "Market Value": -314.44, "Intrinsic Value": 0.0, "Time Value": 314.44, "Daily Theta ($)": 1.32, "Effective Shares": -99.84, "Delta Dollars": -2118.6
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 270115C00020000", "Symbol": "B", "Expiry": "20270115", "Strike": 20.0, "Right": "C", "Currency": "USD", "Exchange": "SMART", "Position": -1, "Security Type": "OPT", "Multiplier": 100, "Delta": 0.6739, "Gamma": 0.0432, "Vega": 0.0932, "Theta": -0.0038, "IV (%)": 32.47, "Option Close": 4.46, "Underlying Close": 21.22, "Days to Maturity": 552, "Market Value": -424.07, "Intrinsic Value": 122.0, "Time Value": 302.07, "Daily Theta ($)": 0.38, "Effective Shares": -67.39, "Delta Dollars": -1430.02
        }
    ]

    # Example: Generate for 'B'
    try:
        output_path = generate_risk_curve(selected_symbol="B", portfolio_data=test_portfolio_data)
        print(f"\nRisk curve saved to: {output_path}")
    except ValueError as e:
        print(f"Error during local test: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during local test: {e}")