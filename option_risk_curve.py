import os
import math
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants ---
CONFIG = {
    "DAYS_IN_YEAR": 365.25,
    "DEFAULT_OPTION_MULTIPLIER": 100,
    "DEFAULT_RISK_FREE_RATE": 0.05
}

# --- Helper Function to Clean Data ---
def clean(val: Any, is_percentage: bool = False) -> Optional[float]:
    """
    Cleans and converts a value to float.
    Handles None, empty strings, and NaN values, returning None if conversion fails.
    Converts percentage values (e.g., "29.64%" or 29.64) to decimal floats if is_percentage=True.
    """
    if val is None or (isinstance(val, str) and val.strip() == ''):
        return None
    
    if isinstance(val, float) and math.isnan(val):
        return None

    try:
        if isinstance(val, str):
            val = val.replace('%', '').strip()
        value = float(val)
        if is_percentage:
            value /= 100.0
        return value
    except (ValueError, TypeError):
        return None

# --- Black-Scholes Option Pricing Model (Vectorized) ---
def black_scholes_vectorized(S: Union[float, np.ndarray], K: float, T: float, r: float, sigma: float, option_type: str) -> Union[float, np.ndarray]:
    """
    Calculates the price of a European option using the Black-Scholes model,
    supporting vectorized S (underlying price).
    """
    if T <= 0:
        if option_type.lower() in ['call', 'c']:
            return np.maximum(0, S - K) if isinstance(S, np.ndarray) else max(0, S - K)
        else: # put
            return np.maximum(0, K - S) if isinstance(S, np.ndarray) else max(0, K - S)

    if sigma <= 0:
        return np.zeros_like(S) if isinstance(S, np.ndarray) else 0.0

    S_val = np.array(S) if isinstance(S, np.ndarray) else S
    K_val = np.array(K) if isinstance(S, np.ndarray) else K

    d1 = (np.log(S_val / K_val) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() in ['call', 'c']:
        price = S_val * norm.cdf(d1) - K_val * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() in ['put', 'p']:
        price = K_val * math.exp(-r * T) * norm.cdf(-d2) - S_val * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call', 'put', 'C', or 'P'")
    return price

# --- Function to Calculate P&L for a Single Option at a Future Date ---
def calculate_single_option_pnl_curve(
    hypothetical_underlying_prices: np.ndarray,
    initial_underlying_price: float,
    strike_price: float,
    expiration_date_obj: datetime.datetime,
    current_date_obj: datetime.datetime,
    target_date_obj: datetime.datetime,
    implied_volatility: float,
    risk_free_rate: float,
    option_type: str,
    num_contracts: float,
    multiplier: float = CONFIG["DEFAULT_OPTION_MULTIPLIER"]
) -> Tuple[np.ndarray, float]:
    """Calculates the P&L curve for a single option at a specified target date."""
    T_initial = (expiration_date_obj - current_date_obj).days / CONFIG["DAYS_IN_YEAR"]
    T_target = (expiration_date_obj - target_date_obj).days / CONFIG["DAYS_IN_YEAR"]

    if T_initial < 0: T_initial = 0.0
    if T_target < 0: T_target = 0.0

    initial_option_price = black_scholes_vectorized(initial_underlying_price, strike_price, T_initial, risk_free_rate, implied_volatility, option_type)
    
    future_option_prices = black_scholes_vectorized(hypothetical_underlying_prices, strike_price, T_target, risk_free_rate, implied_volatility, option_type)
    
    pnl_values = (future_option_prices - initial_option_price) * num_contracts * multiplier

    return pnl_values, initial_option_price * num_contracts * multiplier

# --- Function to Calculate P&L for a Stock Position ---
def calculate_stock_pnl_curve(
    hypothetical_underlying_prices: np.ndarray,
    initial_stock_price: float,
    num_shares: float
) -> Tuple[np.ndarray, float]:
    """Calculates the P&L curve for a stock position."""
    pnl_values = (hypothetical_underlying_prices - initial_stock_price) * num_shares
    return pnl_values, initial_stock_price * num_shares

# --- Main function to generate the risk curve ---
def generate_risk_curve(
    selected_symbol: str,
    portfolio_data: List[Dict[str, Any]],
    analysis_date_str: Optional[str] = None,
    risk_free_rate: float = CONFIG["DEFAULT_RISK_FREE_RATE"],
    output_dir: str = "/tmp" # Changed output_dir to /tmp for Render.com
) -> str:
    """
    Generates and saves the combined option and stock risk curve for a given symbol.
    """
    logging.info(f"--- Generating Risk Curve for {selected_symbol} ---")

    if analysis_date_str:
        analysis_current_date_obj = datetime.datetime.strptime(analysis_date_str, '%Y-%m-%d')
    else:
        # For Render.com, datetime.datetime.now() is appropriate for current date.
        # For local testing, if you want consistent output with test_portfolio_data,
        # you'll need to set analysis_date_str in the __main__ block.
        analysis_current_date_obj = datetime.datetime.now()
    analysis_current_date_str = analysis_current_date_obj.strftime('%Y-%m-%d')

    parsed_portfolio_data = []
    required_base_keys = ['Symbol', 'Security Type', 'Position', 'Underlying Close', 'Date']
    required_opt_keys = ['Expiry', 'Strike', 'Right', 'IV (%)']

    for i, record in enumerate(portfolio_data):
        if not all(key in record for key in required_base_keys):
            logging.warning(f"Skipping record {i+1} ({record.get('Local Symbol', 'N/A')}) due to missing essential base keys.")
            continue

        symbol_from_record = str(record.get('Symbol', '')).strip().upper()
        sec_type = str(record.get('Security Type', '')).strip().upper()

        if symbol_from_record != selected_symbol.upper():
            continue

        try:
            date_str_raw = str(record.get('Date', ''))
            pos_current_date_obj = datetime.datetime.strptime(date_str_raw, '%Y-%m-%d')

            item = {
                "local_symbol": str(record.get('Local Symbol', '')),
                "symbol": symbol_from_record,
                "sec_type": sec_type,
                "current_underlying_price": clean(record.get('Underlying Close')),
                "position": clean(record.get('Position')),
                "current_date_str": pos_current_date_obj.strftime('%Y-%m-%d')
            }

            if sec_type == 'OPT':
                if not all(key in record for key in required_opt_keys):
                    logging.warning(f"Skipping option record {i+1} ({record.get('Local Symbol', 'N/A')}) due to missing essential option keys.")
                    continue

                expiry_str_raw = str(record.get('Expiry', ''))
                exp_date_obj = datetime.datetime.strptime(expiry_str_raw, '%Y%m%d')

                # Use the clean function, which now handles is_percentage=True implicitly if '%' is in string
                implied_vol = clean(record.get('IV (%)'), is_percentage=True) 
                if implied_vol is None:
                    logging.warning(f"Skipping option {record.get('Local Symbol', 'N/A')}: Invalid implied volatility.")
                    continue

                item.update({
                    "strike_price": clean(record.get('Strike')),
                    "expiration_date_str": exp_date_obj.strftime('%Y-%m-%d'),
                    "implied_volatility": implied_vol,
                    "risk_free_rate": risk_free_rate,
                    "option_type": str(record.get('Right', '')).lower(),
                    "multiplier": clean(record.get('Multiplier')) if record.get('Multiplier') is not None else CONFIG["DEFAULT_OPTION_MULTIPLIER"]
                })
                item["future_date_str"] = (exp_date_obj - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                logging.info(f"Parsed option {item['local_symbol']}: IV = {item['implied_volatility']}")

            elif sec_type == 'STK':
                initial_price = clean(record.get('Underlying Close'))
                if initial_price is None or initial_price == 0.0:
                    initial_price = clean(record.get('Option Close'))
                
                if initial_price is None or initial_price == 0.0:
                    raise ValueError("Stock initial price (Underlying Close or Option Close) is missing or zero.")

                item.update({
                    "initial_stock_price": initial_price,
                    "num_shares": clean(record.get('Position'))
                })
            
            if item["current_underlying_price"] is None:
                raise ValueError("Current underlying price is missing for this position.")

            parsed_portfolio_data.append(item)

        except ValueError as ve:
            logging.error(f"Error parsing data for record ({record.get('Local Symbol', 'N/A')}) at index {i+1}: {ve}. Skipping record.")
        except Exception as ex:
            logging.exception(f"An unexpected error occurred during parsing record ({record.get('Local Symbol', 'N/A')}) at index {i+1}.")

    if not parsed_portfolio_data:
        raise ValueError(f"No valid positions found for symbol '{selected_symbol}' after parsing.")

    first_option_expiration_date_obj = None
    first_option_sigma_for_sd = None
    initial_underlying_price_for_sd = None
    first_option_strike_price = None
    first_option_expiration_date_str_raw = None

    options_only_in_filtered = [item for item in parsed_portfolio_data if item["sec_type"] == "OPT"]
    if options_only_in_filtered:
        options_only_in_filtered.sort(key=lambda x: datetime.datetime.strptime(x["expiration_date_str"], '%Y-%m-%d'))
        first_option_to_expire = options_only_in_filtered[0]
        
        first_option_expiration_date_str_raw = first_option_to_expire["expiration_date_str"]
        first_option_expiration_date_obj = datetime.datetime.strptime(first_option_expiration_date_str_raw, '%Y-%m-%d')
        first_option_sigma_for_sd = first_option_to_expire["implied_volatility"]
        initial_underlying_price_for_sd = first_option_to_expire["current_underlying_price"]
        first_option_strike_price = first_option_to_expire["strike_price"]
        logging.info(f"First option for SD: {first_option_to_expire['local_symbol']}, IV = {first_option_sigma_for_sd}, Price = {initial_underlying_price_for_sd}")

    if first_option_expiration_date_obj is None:
        logging.warning(f"No option found for '{selected_symbol}'. Only plotting 'as of today'. SD shading will not be applied.")
        # If no options, set a dummy future expiry far out for 'first expiry' curve, though it won't be used for SD calculations.
        # The 'P&L at first option expiry' curve will just default to the 'as of today' curve if no options exist to define a future expiry.
        first_option_expiration_date_obj = analysis_current_date_obj + datetime.timedelta(days=CONFIG["DAYS_IN_YEAR"] * 10)
        first_option_expiration_date_str_raw = first_option_expiration_date_obj.strftime('%Y-%m-%d')

    all_current_prices = [item["current_underlying_price"] for item in parsed_portfolio_data]
    all_strike_prices = [item["strike_price"] for item in parsed_portfolio_data if item["sec_type"] == "OPT"]

    valid_current_prices = [p for p in all_current_prices if p is not None and p > 0]
    valid_strike_prices = [p for p in all_strike_prices if p is not None and p > 0]

    if not valid_current_prices and not valid_strike_prices:
        raise ValueError(f"No valid price data found to determine plot range for symbol '{selected_symbol}'.")

    if not valid_strike_prices:
        overall_price_range_min = min(valid_current_prices) * 0.8
        overall_price_range_max = max(valid_current_prices) * 1.2
    else:
        all_valid_prices = valid_current_prices + valid_strike_prices
        overall_price_range_min = min(all_valid_prices) * 0.8
        overall_price_range_max = max(all_valid_prices) * 1.2

    if overall_price_range_min <= 0:
        overall_price_range_min = 0.1 * overall_price_range_max

    hypothetical_underlying_prices = np.linspace(overall_price_range_min, overall_price_range_max, 200)

    combined_pnl_as_of_today = np.zeros_like(hypothetical_underlying_prices)
    combined_pnl_at_first_option_expiry = np.zeros_like(hypothetical_underlying_prices)
    
    total_initial_cost_at_analysis_start = 0

    logging.info(f"Calculating P&L for '{selected_symbol}' positions (Analysis Date: {analysis_current_date_str})...")

    for i, position_params in enumerate(parsed_portfolio_data):
        try:
            position_current_date_obj = datetime.datetime.strptime(position_params["current_date_str"], '%Y-%m-%d')

            if position_params["sec_type"] == "OPT":
                expiration_date_obj = datetime.datetime.strptime(position_params["expiration_date_str"], '%Y-%m-%d')
                
                if analysis_current_date_obj > expiration_date_obj:
                    logging.info(f"Skipping Option {i+1} ({position_params['local_symbol']}): Analysis date is past its expiration.")
                    continue

                if any(x is None for x in [position_params["strike_price"], position_params["implied_volatility"], position_params["position"]]):
                    logging.warning(f"Skipping option {position_params['local_symbol']}: Missing strike, IV, or position.")
                    continue
                if position_params["option_type"] not in ["call", "put", "c", "p"]:
                    logging.warning(f"Skipping option {position_params['local_symbol']}: Invalid option type {position_params['option_type']}.")
                    continue

                pnl_today, initial_cost_today = calculate_single_option_pnl_curve(
                    hypothetical_underlying_prices,
                    position_params["current_underlying_price"],
                    position_params["strike_price"],
                    expiration_date_obj,
                    position_current_date_obj,
                    analysis_current_date_obj,
                    position_params["implied_volatility"],
                    position_params["risk_free_rate"],
                    position_params["option_type"],
                    position_params["position"],
                    position_params["multiplier"]
                )
                combined_pnl_as_of_today += pnl_today
                total_initial_cost_at_analysis_start += initial_cost_today

                if first_option_expiration_date_obj <= expiration_date_obj: # Only include options that haven't expired by the first expiry date
                    pnl_first_expiry, _ = calculate_single_option_pnl_curve(
                        hypothetical_underlying_prices,
                        position_params["current_underlying_price"],
                        position_params["strike_price"],
                        expiration_date_obj,
                        position_current_date_obj,
                        first_option_expiration_date_obj,
                        position_params["implied_volatility"],
                        position_params["risk_free_rate"],
                        position_params["option_type"],
                        position_params["position"],
                        position_params["multiplier"]
                    )
                    combined_pnl_at_first_option_expiry += pnl_first_expiry
                else:
                    logging.info(f"  - Option {i+1} ({position_params['local_symbol']}): Will be expired by {first_option_expiration_date_str_raw}, not included in that curve.")

                logging.info(f"  - Option {i+1} ({position_params['local_symbol']}): P&L calculated for both curves.")

            elif position_params["sec_type"] == "STK":
                if position_params["initial_stock_price"] is None or position_params["position"] is None:
                    logging.warning(f"Skipping stock {position_params['local_symbol']}: Missing initial price or position.")
                    continue

                pnl_stock, initial_cost_stock = calculate_stock_pnl_curve(
                    hypothetical_underlying_prices,
                    position_params["initial_stock_price"],
                    position_params["position"]
                )
                combined_pnl_as_of_today += pnl_stock
                combined_pnl_at_first_option_expiry += pnl_stock
                total_initial_cost_at_analysis_start += initial_cost_stock
                logging.info(f"  - Stock {i+1} ({position_params['local_symbol']}): P&L calculated for both curves.")

            else:
                logging.warning(f"Unknown security type for {position_params['local_symbol']}. Skipping.")

        except KeyError as ke:
            logging.error(f"Error processing position {i+1} ({position_params.get('local_symbol', 'N/A')}) due to missing key: {ke}. Check sheet headers.")
        except ValueError as ve:
            logging.error(f"Error processing position {i+1} ({position_params.get('local_symbol', 'N/A')}) due to data type conversion: {ve}. Check data format in sheet.")
        except Exception as e:
            logging.exception(f"An unexpected error occurred for position {i+1} ({position_params.get('local_symbol', 'N/A')}): {e}")

    logging.info(f"\nTotal Initial Cost of Portfolio (at {analysis_current_date_str}): ${total_initial_cost_at_analysis_start:.2f}\n")

    os.makedirs(output_dir, exist_ok=True)
    output_image_filename = f"{selected_symbol}_risk_curve_{analysis_current_date_obj.strftime('%Y%m%d_%H%M%S')}.png"
    output_image_filepath = os.path.join(output_dir, output_image_filename)

    plt.figure(figsize=(14, 8))
    
    # Initialize effective X-axis limits based on the initial linspace range
    effective_x_min = overall_price_range_min
    effective_x_max = overall_price_range_max

    if first_option_expiration_date_obj and first_option_sigma_for_sd is not None and initial_underlying_price_for_sd is not None and options_only_in_filtered:
        T_sd = (first_option_expiration_date_obj - analysis_current_date_obj).days / CONFIG["DAYS_IN_YEAR"]
        if T_sd > 0 and first_option_sigma_for_sd > 0:
            log_return_sd = first_option_sigma_for_sd * math.sqrt(T_sd)

            lower_4sd = initial_underlying_price_for_sd * math.exp(-4 * log_return_sd)
            upper_4sd = initial_underlying_price_for_sd * math.exp(4 * log_return_sd)
            lower_3sd = initial_underlying_price_for_sd * math.exp(-3 * log_return_sd)
            upper_3sd = initial_underlying_price_for_sd * math.exp(3 * log_return_sd)
            lower_2sd = initial_underlying_price_for_sd * math.exp(-2 * log_return_sd)
            upper_2sd = initial_underlying_price_for_sd * math.exp(2 * log_return_sd)
            lower_1sd = initial_underlying_price_for_sd * math.exp(-1 * log_return_sd)
            upper_1sd = initial_underlying_price_for_sd * math.exp(1 * log_return_sd)

            plt.axvspan(lower_3sd, upper_3sd, color='skyblue', alpha=0.15, label='±3 SD Range (First Expiry)')
            plt.axvspan(lower_2sd, upper_2sd, color='lightgreen', alpha=0.25, label='±2 SD Range')
            plt.axvspan(lower_1sd, upper_1sd, color='lightcoral', alpha=0.35, label='±1 SD Range')
            logging.info(f"SD Ranges calculated and added based on first option expiry ({first_option_expiration_date_str_raw}).")
            logging.info(f"    1 SD Range: ${lower_1sd:.2f} - ${upper_1sd:.2f}")
            logging.info(f"    2 SD Range: ${lower_2sd:.2f} - ${upper_2sd:.2f}")
            logging.info(f"    3 SD Range: ${lower_3sd:.2f} - ${upper_3sd:.2f}")
            logging.info(f"    4 SD Range: ${lower_4sd:.2f} - ${upper_4sd:.2f}")
            
            # Apply 4 SD limit to X-axis
            plt.xlim(lower_4sd, upper_4sd)
            effective_x_min = lower_4sd # Update effective X-axis limits for Y-axis calculation
            effective_x_max = upper_4sd # Update effective X-axis limits for Y-axis calculation
        else:
            logging.warning("Cannot calculate SD ranges. First option expired or has zero volatility.")

    colors = plt.get_cmap('tab10')(np.linspace(0, 1, 10))

    plt.plot(hypothetical_underlying_prices, combined_pnl_as_of_today, label=f'P&L as of Today ({analysis_current_date_str})', color=colors[0], linestyle='--', linewidth=2)
    
    if first_option_expiration_date_obj > analysis_current_date_obj and options_only_in_filtered:
        plt.plot(hypothetical_underlying_prices, combined_pnl_at_first_option_expiry, label=f'P&L at {first_option_expiration_date_str_raw} (First Option Expiry)', color=colors[1], linestyle='-', linewidth=2)

    # --- Start of Modified Y-axis Range Logic ---
    # Filter the P&L arrays to only include values within the effective X-axis range
    mask = (hypothetical_underlying_prices >= effective_x_min) & \
           (hypothetical_underlying_prices <= effective_x_max)

    filtered_pnl_as_of_today = combined_pnl_as_of_today[mask]
    filtered_pnl_at_first_option_expiry = combined_pnl_at_first_option_expiry[mask]

    # Calculate pnl_min and pnl_max from the filtered P&L values
    pnl_values_to_consider = []
    if filtered_pnl_as_of_today.size > 0:
        pnl_values_to_consider.extend(filtered_pnl_as_of_today.tolist())
    if filtered_pnl_at_first_option_expiry.size > 0 and first_option_expiration_date_obj > analysis_current_date_obj and options_only_in_filtered:
        # Only consider the second curve if it's actually plotted
        pnl_values_to_consider.extend(filtered_pnl_at_first_option_expiry.tolist())
        
    if pnl_values_to_consider: # Check if the list is not empty
        pnl_min = min(pnl_values_to_consider)
        pnl_max = max(pnl_values_to_consider)
    else:
        # Fallback to the full P&L range if no data within the specified X-axis range (unlikely with current setup)
        logging.warning("No P&L data found within the calculated X-axis range for Y-axis limits. Using full range.")
        pnl_min = np.min(combined_pnl_as_of_today)
        pnl_max = np.max(combined_pnl_as_of_today)
        if options_only_in_filtered and first_option_expiration_date_obj > analysis_current_date_obj:
            pnl_min = min(pnl_min, np.min(combined_pnl_at_first_option_expiry))
            pnl_max = max(pnl_max, np.max(combined_pnl_at_first_option_expiry))

    y_buffer_min = 0.1 * abs(pnl_min) if pnl_min < 0 else 0.1 * pnl_min
    y_buffer_max = 0.2 * abs(pnl_max) if pnl_max > 0 else 0.2 * pnl_max # Increased top buffer to 20%
    plt.ylim(pnl_min - y_buffer_min, pnl_max + y_buffer_max)
    # --- End of Modified Y-axis Range Logic ---


    initial_S_ref = parsed_portfolio_data[0]["current_underlying_price"]
    plt.axvline(x=initial_S_ref, color='gray', linestyle='--', label=f'Current Underlying Price (${initial_S_ref})')

    # --- Start of Modified Breakeven Annotation Logic (based on First Option Expiry) ---
    # Find zero crossings based on the 'P&L at First Option Expiry' curve
    # Ensure this curve is actually being plotted (i.e., there are options and the expiry is in the future)
    if first_option_expiration_date_obj > analysis_current_date_obj and options_only_in_filtered:
        # Re-filter hypothetical_underlying_prices and combined_pnl_at_first_option_expiry for breakeven calculations
        # to ensure breakevens are within the displayed X-axis range
        plot_hypothetical_underlying_prices_for_be = hypothetical_underlying_prices[mask]
        plot_combined_pnl_at_first_option_expiry_for_be = combined_pnl_at_first_option_expiry[mask]

        zero_crossings_indices = np.where(np.diff(np.sign(plot_combined_pnl_at_first_option_expiry_for_be)))[0]

        for idx in zero_crossings_indices:
            x1, y1 = plot_hypothetical_underlying_prices_for_be[idx], plot_combined_pnl_at_first_option_expiry_for_be[idx]
            x2, y2 = plot_hypothetical_underlying_prices_for_be[idx + 1], plot_combined_pnl_at_first_option_expiry_for_be[idx + 1]

            if (y2 - y1) != 0:
                breakeven_price = x1 - y1 * (x2 - x1) / (y2 - y1)
                offset_y = 50  # Fixed upward offset for annotation

                plt.annotate(f'BE: ${breakeven_price:.2f}',
                             xy=(breakeven_price, 0),
                             xytext=(0, offset_y), # Use offset from xy point
                             textcoords="offset points",
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5, headlength=8),
                             fontsize=9, color='darkred',
                             ha='center', va='bottom') # Align text above arrow
    else:
        logging.info("No 'P&L at First Option Expiry' curve available or relevant for breakeven annotations.")
    # --- End of Modified Breakeven Annotation Logic ---


    # Plot strike price and stock lines
    for item in parsed_portfolio_data:
        # Check if the strike/price falls within the plotted X-axis range before drawing the line
        if item["sec_type"] == "OPT":
            if item["strike_price"] is not None and \
               effective_x_min <= item["strike_price"] <= effective_x_max: # For option strike price
                is_first_expiring_option = (
                    item["strike_price"] == first_option_strike_price and
                    item["expiration_date_str"] == first_option_expiration_date_str_raw
                )
                line_color = 'red' if is_first_expiring_option else 'green'

                label_text = f'Strike: ${item["strike_price"]} ({item["option_type"].upper()}) (Pos: {item["position"]}) (Exp: {item["expiration_date_str"]})'
                # Check if label already exists to avoid duplicate legend entries for identical strikes
                if label_text not in plt.gca().get_legend_handles_labels()[1]:
                    plt.axvline(x=item["strike_price"], color=line_color, linestyle=':', alpha=0.6, label=label_text)
                else:
                    plt.axvline(x=item["strike_price"], color=line_color, linestyle=':', alpha=0.6)
        elif item["sec_type"] == "STK":
            # For stock, check its initial price against the effective X-axis range
            if item["initial_stock_price"] is not None and \
               effective_x_min <= item["initial_stock_price"] <= effective_x_max:
                label_text = f'Stock (Pos: {item["position"]} shares)'
                if label_text not in plt.gca().get_legend_handles_labels()[1]:
                    plt.axvline(x=item["initial_stock_price"], color='blue', linestyle='--', alpha=0.7, label=label_text)
                else:
                    plt.axvline(x=item["initial_stock_price"], color='blue', linestyle='--', alpha=0.7)


    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.title(f'Combined Option and Stock Portfolio P&L Curves ({selected_symbol} Positions)')
    plt.subplots_adjust(top=0.92)  # Increased top margin to avoid overlap
    plt.xlabel('Underlying Price at Future Date')
    plt.ylabel('Combined Profit / Loss ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    try:
        plt.savefig(output_image_filepath)
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save plot to {output_image_filepath}: {e}")
        raise

    return output_image_filepath

# --- Local Testing Example ---
if __name__ == "__main__":
    logging.info("--- Running local test of generate_risk_curve function ---")
    
    test_portfolio_data = [
        {
            "Date": "2025-07-12", "Local Symbol": "B", "Symbol": "B", "Expiry": "", "Strike": "", "Right": "", 
            "Currency": "USD", "Exchange": "SMART", "Position": 300, "Security Type": "STK", "Multiplier": 1, 
            "Delta": "", "Gamma": "", "Vega": "", "Theta": "", "IV (%)": "", "Option Close": 21.07, 
            "Underlying Close": 21.07, "Days to Maturity": "", "Market Value": 6363.0, "Intrinsic Value": "", 
            "Time Value": "", "Daily Theta ($)": "", "Effective Shares": 300, "Delta Dollars": 6363
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 250718C00021000", "Symbol": "B", "Expiry": "20250718", 
            "Strike": 21.0, "Right": "C", "Currency": "USD", "Exchange": "SMART", "Position": -3, 
            "Security Type": "OPT", "Multiplier": 100, "Delta": 0.607, "Gamma": 0.4441, "Vega": 0.0111, 
            "Theta": -0.0259, "IV (%)": 29.64, "Option Close": 0.4, "Underlying Close": 21.22, 
            "Days to Maturity": 6, "Market Value": -141.45, "Intrinsic Value": 66.0, "Time Value": 75.45, 
            "Daily Theta ($)": 7.78, "Effective Shares": -182.11, "Delta Dollars": -3864.37
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 250725P00021500", "Symbol": "B", "Expiry": "20250725", 
            "Strike": 21.5, "Right": "P", "Currency": "USD", "Exchange": "SMART", "Position": -3, 
            "Security Type": "OPT", "Multiplier": 100, "Delta": -0.5785, "Gamma": 0.3278, "Vega": 0.016, 
            "Theta": -0.0158, "IV (%)": 29.3, "Option Close": 0.73, "Underlying Close": 21.22, 
            "Days to Maturity": 13, "Market Value": -188.8, "Intrinsic Value": 84.0, "Time Value": 104.8, 
            "Daily Theta ($)": 4.75, "Effective Shares": 173.54, "Delta Dollars": 3682.52
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 250919P00019000", "Symbol": "B", "Expiry": "20250919", 
            "Strike": 19.0, "Right": "P", "Currency": "USD", "Exchange": "SMART", "Position": -3, 
            "Security Type": "OPT", "Multiplier": 100, "Delta": -0.1896, "Gamma": 0.0905, "Vega": 0.0262, 
            "Theta": -0.0056, "IV (%)": 32.6, "Option Close": 0.36, "Underlying Close": 21.22, 
            "Days to Maturity": 69, "Market Value": -100.83, "Intrinsic Value": 0.0, "Time Value": 100.83, 
            "Daily Theta ($)": 1.68, "Effective Shares": 56.87, "Delta Dollars": 1206.78
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 251219C00022000", "Symbol": "B", "Expiry": "20251219", 
            "Strike": 22.0, "Right": "C", "Currency": "USD", "Exchange": "SMART", "Position": -2, 
            "Security Type": "OPT", "Multiplier": 100, "Delta": 0.4992, "Gamma": 0.0896, "Vega": 0.0563, 
            "Theta": -0.0066, "IV (%)": 31.87, "Option Close": 1.51, "Underlying Close": 21.22, 
            "Days to Maturity": 160, "Market Value": -314.44, "Intrinsic Value": 0.0, "Time Value": 314.44, 
            "Daily Theta ($)": 1.32, "Effective Shares": -99.84, "Delta Dollars": -2118.6
        },
        {
            "Date": "2025-07-12", "Local Symbol": "B 270115C00020000", "Symbol": "B", "Expiry": "20270115", 
            "Strike": 20.0, "Right": "C", "Currency": "USD", "Exchange": "SMART", "Position": -1, 
            "Security Type": "OPT", "Multiplier": 100, "Delta": 0.6739, "Gamma": 0.0432, "Vega": 0.0932, 
            "Theta": -0.0038, "IV (%)": 32.47, "Option Close": 4.46, "Underlying Close": 21.22, 
            "Days to Maturity": 552, "Market Value": -424.07, "Intrinsic Value": 122.0, "Time Value": 302.07, 
            "Daily Theta ($)": 0.38, "Effective Shares": -67.39, "Delta Dollars": -1430.02
        }
    ]

    selected_symbol = "B"
    portfolio_data = test_portfolio_data

    try:
        # Use the specific analysis_date_str from the test data for consistent plotting
        # For production use with Render.com, you would typically remove this if you want it to use the current date
        output_path = generate_risk_curve(selected_symbol=selected_symbol, portfolio_data=portfolio_data, analysis_date_str="2025-07-12")
        logging.info(f"\nRisk curve saved to: {output_path}")
        # Display the image (optional, requires matplotlib backend)
        # This part will not run on Render.com but is useful for local debugging
        img = plt.imread(output_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    except ValueError as e:
        logging.error(f"Error during local test: {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during local test: {e}")

```