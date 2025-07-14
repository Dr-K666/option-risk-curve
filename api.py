# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, root_validator
from pydantic.functional_validators import BeforeValidator # For Pydantic v2 BeforeValidator
from typing import List, Dict, Any, Optional, Union, Annotated # Import Annotated for Pydantic v2
import os
import json
import base64
import traceback
import uuid
import logging
import datetime
import tempfile # Added for tempfile.gettempdir()

# Configure logging for the API
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import your risk curve generation logic from option_risk_curve.py
# Ensure option_risk_curve.py is in the same directory as api.py
try:
    from option_risk_curve import generate_risk_curve, clean # Import clean function too
except ImportError:
    logging.error("Error: Could not import generate_risk_curve from option_risk_curve.py")
    logging.error("Please ensure 'option_risk_curve.py' is in the same directory and all its dependencies are installed.")
    raise # Re-raise to stop the application startup if core logic is missing

app = FastAPI(
    title="Option Risk Curve Generator API",
    description="API to generate combined option and stock risk curves from portfolio data.",
    version="1.0.0",
)

# Define a custom Pydantic type using Annotated and BeforeValidator
# This will apply the clean function to the input value before Pydantic validates its type
# This is for Pydantic v2. If you're on Pydantic v1, the root_validator approach is used.
# Assuming Pydantic v2, as it's the latest. If you're on v1, this will need adjustment.
CleanFloat = Annotated[float, BeforeValidator(clean)]
CleanOptionalFloat = Annotated[Optional[float], BeforeValidator(clean)]


# Define Pydantic model for a single row from your Google Sheet
class PortfolioItem(BaseModel):
    Date: str = Field(..., alias="Date") # Required, string format YYYY-MM-DD
    Local_Symbol: Optional[str] = Field(None, alias="Local Symbol")
    Symbol: str = Field(..., alias="Symbol") # Required
    Expiry: Optional[str] = Field(None, alias="Expiry") # YYYYMMDD string
    Strike: CleanOptionalFloat = Field(None, alias="Strike") # Can be float or None
    Right: Optional[str] = Field(None, alias="Right") # 'C', 'P', or None
    Currency: Optional[str] = Field(None, alias="Currency")
    Exchange: Optional[str] = Field(None, alias="Exchange")
    
    # Use CleanOptionalFloat for fields that should be floats but might come as strings/empty
    Position: CleanOptionalFloat = Field(None, alias="Position")
    Security_Type: str = Field(..., alias="Security Type") # Required, 'STK' or 'OPT'
    Multiplier: CleanOptionalFloat = Field(None, alias="Multiplier")
    
    Delta: CleanOptionalFloat = Field(None, alias="Delta")
    Gamma: CleanOptionalFloat = Field(None, alias="Gamma")
    Vega: CleanOptionalFloat = Field(None, alias="Vega")
    Theta: CleanOptionalFloat = Field(None, alias="Theta")
    IV_percent: CleanOptionalFloat = Field(None, alias="IV (%)") # Mapped from 'IV (%)'
    Option_Close: CleanOptionalFloat = Field(None, alias="Option Close")
    Underlying_Close: CleanOptionalFloat = Field(None, alias="Underlying Close")
    Days_to_Maturity: CleanOptionalFloat = Field(None, alias="Days to Maturity")
    Market_Value: CleanOptionalFloat = Field(None, alias="Market Value")
    Intrinsic_Value: CleanOptionalFloat = Field(None, alias="Intrinsic Value")
    Time_Value: CleanOptionalFloat = Field(None, alias="Time Value")
    Daily_Theta_dollars: CleanOptionalFloat = Field(None, alias="Daily Theta ($)")
    Effective_Shares: CleanOptionalFloat = Field(None, alias="Effective Shares")
    Delta_Dollars: CleanOptionalFloat = Field(None, alias="Delta Dollars")

    # The root_validator is still useful for cross-field validation or complex logic
    # that BeforeValidator on individual fields can't handle (like date formats).
    @root_validator(pre=True)
    def validate_types_and_values(cls, values):
        # Basic validation for Security_Type
        sec_type = values.get('Security Type')
        if sec_type and sec_type.upper() not in ['STK', 'OPT']:
            raise ValueError(f"Invalid Security Type: {sec_type}. Must be 'STK' or 'OPT'.")
        
        # Basic validation for Option Right
        option_right = values.get('Right')
        if sec_type and sec_type.upper() == 'OPT' and option_right and option_right.upper() not in ['C', 'P']:
            raise ValueError(f"Invalid Option Right: {option_right}. Must be 'C' or 'P' for options.")

        # Basic date format validation (YYYY-MM-DD for Date, YYYYMMDD for Expiry)
        date_str = values.get('Date')
        if date_str:
            try:
                datetime.datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                raise ValueError(f"Invalid 'Date' format: {date_str}. Expected YYYY-MM-DD.")
        
        expiry_str = values.get('Expiry')
        if expiry_str and sec_type and sec_type.upper() == 'OPT':
            try:
                datetime.datetime.strptime(expiry_str, '%Y%m%d')
            except ValueError:
                raise ValueError(f"Invalid 'Expiry' format: {expiry_str}. Expected YYYYMMDD for options.")

        # The IV (%) conversion from string "XX.YY%" to float is now handled by clean() via BeforeValidator
        # but if it's just a number string (e.g., "29.64"), clean() will also handle it.
        
        return values

    class Config:
        allow_population_by_field_name = True
        extra = "allow" # Allows additional fields not explicitly defined in the model


# Health check endpoint
@app.get("/")
async def health_check():
    """
    Health check endpoint to confirm the API is running.
    """
    return {"status": "ok", "message": "Option Risk Curve API is running."}

@app.post("/risk-curve/{symbol}")
async def get_risk_curve(symbol: str, portfolio_rows: List[PortfolioItem]):
    """
    Generates an option risk curve for a specified stock symbol.

    Args:
        symbol (str): The underlying stock symbol (e.g., 'B', 'DVN').
        portfolio_rows (List[PortfolioItem]): A list of portfolio data rows
                                             from your Google Sheet.

    Returns:
        Dict: A dictionary containing the base64 encoded image of the risk curve.
    """
    # Validate symbol input
    if not symbol or not isinstance(symbol, str) or symbol.strip() == "":
        raise HTTPException(status_code=400, detail="Stock symbol cannot be empty.")
    
    # Validate portfolio_rows input
    if not portfolio_rows:
        raise HTTPException(status_code=400, detail="Portfolio data cannot be empty.")

    # Convert Pydantic models back to a list of plain dictionaries
    # using by_alias=True to get the original Google Sheet header names
    # Pydantic has already applied the 'clean' function and type conversions here
    raw_portfolio_data = [row.dict(by_alias=True) for row in portfolio_rows]

    # --- Securely handle Google credentials from environment variable ---
    google_credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not google_credentials_json_str:
        logging.error("GOOGLE_CREDENTIALS_JSON environment variable is not set.")
        raise HTTPException(status_code=500, detail="Google credentials (GOOGLE_CREDENTIALS_JSON) not found in environment variables. Please configure Render.com environment variables.")
    
    # Validate JSON format of credentials string
    try:
        json.loads(google_credentials_json_str)
    except json.JSONDecodeError:
        logging.error("GOOGLE_CREDENTIALS_JSON is not a valid JSON string.")
        raise HTTPException(status_code=500, detail="Google credentials JSON is malformed.")

    # Generate unique filenames for concurrency safety
    unique_id = uuid.uuid4()
    # Use tempfile.gettempdir() for a robust temporary directory path
    credentials_file_path = os.path.join(tempfile.gettempdir(), f"google_credentials_{unique_id}.json")
    image_path = None # Initialize image_path for finally block

    try:
        with open(credentials_file_path, "w") as f:
            f.write(google_credentials_json_str)
        
        # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file_path
        logging.info(f"Temporary credentials file created: {credentials_file_path}")

        # Call your main risk curve generation function
        image_path = generate_risk_curve(
            selected_symbol=symbol.upper(), # Ensure symbol is uppercase for filtering
            portfolio_data=raw_portfolio_data,
            output_dir=tempfile.gettempdir() # Ensure output is saved to the system temp dir
        )

        # Check if image was actually generated
        if not image_path or not os.path.exists(image_path):
            logging.error(f"Risk curve image was not generated at expected path: {image_path}")
            raise HTTPException(status_code=500, detail="Risk curve image could not be generated.")

        # Read the generated image file and encode it as base64
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        return {"image_base64": img_base64}

    except ValueError as ve:
        # Catch specific validation errors from generate_risk_curve
        logging.warning(f"Bad request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Catch any other unexpected errors during generation
        logging.exception("An unhandled internal server error occurred during risk curve generation.") # Logs full traceback
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}. Check server logs for details.")
    finally:
        # Clean up temporary files, even if an error occurred
        if os.path.exists(credentials_file_path):
            os.remove(credentials_file_path)
            logging.info(f"Cleaned up temporary credentials file: {credentials_file_path}")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"Cleaned up temporary image file: {image_path}")

