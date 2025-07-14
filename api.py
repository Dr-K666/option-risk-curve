from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError, root_validator
from typing import List, Optional
import os
import json
import base64
import traceback
import uuid
import logging
import datetime
import tempfile  # <-- Added

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from option_risk_curve import generate_risk_curve, clean
except ImportError:
    logging.error("Error: Could not import generate_risk_curve from option_risk_curve.py")
    raise

app = FastAPI(
    title="Option Risk Curve Generator API",
    description="API to generate combined option and stock risk curves from portfolio data.",
    version="1.0.0",
)

class PortfolioItem(BaseModel):
    # Your fields as before...
    Date: str = Field(..., alias="Date")
    Local_Symbol: Optional[str] = Field(None, alias="Local Symbol")
    Symbol: str = Field(..., alias="Symbol")
    Expiry: Optional[str] = Field(None, alias="Expiry")
    Strike: Optional[float] = Field(None, alias="Strike")
    Right: Optional[str] = Field(None, alias="Right")
    Currency: Optional[str] = Field(None, alias="Currency")
    Exchange: Optional[str] = Field(None, alias="Exchange")
    Position: Optional[float] = Field(None, alias="Position")
    Security_Type: str = Field(..., alias="Security Type")
    Multiplier: Optional[float] = Field(None, alias="Multiplier")
    Delta: Optional[float] = Field(None, alias="Delta")
    Gamma: Optional[float] = Field(None, alias="Gamma")
    Vega: Optional[float] = Field(None, alias="Vega")
    Theta: Optional[float] = Field(None, alias="Theta")
    IV_percent: Optional[float] = Field(None, alias="IV (%)")
    Option_Close: Optional[float] = Field(None, alias="Option Close")
    Underlying_Close: Optional[float] = Field(None, alias="Underlying Close")
    Days_to_Maturity: Optional[float] = Field(None, alias="Days to Maturity")
    Market_Value: Optional[float] = Field(None, alias="Market Value")
    Intrinsic_Value: Optional[float] = Field(None, alias="Intrinsic Value")
    Time_Value: Optional[float] = Field(None, alias="Time Value")
    Daily_Theta_dollars: Optional[float] = Field(None, alias="Daily Theta ($)")
    Effective_Shares: Optional[float] = Field(None, alias="Effective Shares")
    Delta_Dollars: Optional[float] = Field(None, alias="Delta Dollars")

    @root_validator(pre=True)
    def validate_types_and_values(cls, values):
        sec_type = values.get('Security Type')
        if sec_type and sec_type.upper() not in ['STK', 'OPT']:
            raise ValueError(f"Invalid Security Type: {sec_type}. Must be 'STK' or 'OPT'.")
        
        option_right = values.get('Right')
        if sec_type and sec_type.upper() == 'OPT' and option_right and option_right.upper() not in ['C', 'P']:
            raise ValueError(f"Invalid Option Right: {option_right}. Must be 'C' or 'P' for options.")

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

        iv_percent_raw = values.get('IV (%)')
        if isinstance(iv_percent_raw, str) and '%' in iv_percent_raw:
            try:
                values['IV (%)'] = float(iv_percent_raw.replace('%', '').strip())
            except ValueError:
                raise ValueError(f"Invalid 'IV (%)' format: {iv_percent_raw}. Expected a number or percentage string.")
        
        return values

    class Config:
        allow_population_by_field_name = True
        extra = "allow"

@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Option Risk Curve API is running."}

@app.post("/risk-curve/{symbol}")
async def get_risk_curve(symbol: str, portfolio_rows: List[PortfolioItem]):
    if not symbol or not isinstance(symbol, str) or symbol.strip() == "":
        raise HTTPException(status_code=400, detail="Stock symbol cannot be empty.")
    if not portfolio_rows:
        raise HTTPException(status_code=400, detail="Portfolio data cannot be empty.")

    raw_portfolio_data = [row.dict(by_alias=True) for row in portfolio_rows]

    google_credentials_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not google_credentials_json_str:
        logging.error("GOOGLE_CREDENTIALS_JSON environment variable is not set.")
        raise HTTPException(status_code=500, detail="Google credentials (GOOGLE_CREDENTIALS_JSON) not found in environment variables.")
    
    try:
        json.loads(google_credentials_json_str)
    except json.JSONDecodeError:
        logging.error("GOOGLE_CREDENTIALS_JSON is not a valid JSON string.")
        raise HTTPException(status_code=500, detail="Google credentials JSON is malformed.")

    unique_id = uuid.uuid4()
    temp_dir = tempfile.gettempdir()  # Use system temp directory
    credentials_file_path = os.path.join(temp_dir, f"google_credentials_{unique_id}.json")
    image_path = None

    try:
        with open(credentials_file_path, "w") as f:
            f.write(google_credentials_json_str)
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file_path
        logging.info(f"Temporary credentials file created: {credentials_file_path}")

        image_path = generate_risk_curve(
            selected_symbol=symbol.upper(),
            portfolio_data=raw_portfolio_data,
            output_dir=temp_dir
        )

        if not image_path or not os.path.exists(image_path):
            logging.error(f"Risk curve image was not generated at expected path: {image_path}")
            raise HTTPException(status_code=500, detail="Risk curve image could not be generated.")

        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"image_base64": img_base64}

    except ValueError as ve:
        logging.warning(f"Bad request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.exception("An unhandled internal server error occurred during risk curve generation.")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}. Check server logs for details.")
    finally:
        if os.path.exists(credentials_file_path):
            os.remove(credentials_file_path)
            logging.info(f"Cleaned up temporary credentials file: {credentials_file_path}")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"Cleaned up temporary image file: {image_path}")