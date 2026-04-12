# Defines the data contract for the /predict endpoint.
# Pydantic validates every incoming request before it touches the model. 
# (Bassically, it checks that the JSON body has the right fields and data types. And rejects the request with a clear error message if anything's wrong.)

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Represents a single credit card transaction.
    V1-V28 are PCA-anonymized features from the original dataset.
    We enforce realistic bounds to reject obviously corrupted inputs.
    """
    V1:  float = Field(..., ge=-30, le=30)
    V2:  float = Field(..., ge=-30, le=30)
    V3:  float = Field(..., ge=-30, le=30)
    V4:  float = Field(..., ge=-30, le=30)
    V5:  float = Field(..., ge=-30, le=30)
    V6:  float = Field(..., ge=-30, le=30)
    V7:  float = Field(..., ge=-30, le=30)
    V8:  float = Field(..., ge=-30, le=30)
    V9:  float = Field(..., ge=-30, le=30)
    V10: float = Field(..., ge=-30, le=30)
    V11: float = Field(..., ge=-30, le=30)
    V12: float = Field(..., ge=-30, le=30)
    V13: float = Field(..., ge=-30, le=30)
    V14: float = Field(..., ge=-30, le=30)
    V15: float = Field(..., ge=-30, le=30)
    V16: float = Field(..., ge=-30, le=30)
    V17: float = Field(..., ge=-30, le=30)
    V18: float = Field(..., ge=-30, le=30)
    V19: float = Field(..., ge=-30, le=30)
    V20: float = Field(..., ge=-30, le=30)
    V21: float = Field(..., ge=-30, le=30)
    V22: float = Field(..., ge=-30, le=30)
    V23: float = Field(..., ge=-30, le=30)
    V24: float = Field(..., ge=-30, le=30)
    V25: float = Field(..., ge=-30, le=30)
    V26: float = Field(..., ge=-30, le=30)
    V27: float = Field(..., ge=-30, le=30)
    V28: float = Field(..., ge=-30, le=30)
    Amount: float = Field(..., ge=0, le=50000)

    @field_validator("Amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        # Transactions of exactly $0 are suspicious — flag them but allow through
        # A real system might route these to a separate review queue
        if v == 0:
            pass  # allowed, but worth knowing about
        return round(v, 2)  # normalize to 2 decimal places like real currency

    model_config = {
        # Reject any fields not defined above (prevents parameter pollution attacks)
        "extra": "forbid",
        # Example shown in /docs (FastAPI auto-generates this)
        "json_schema_extra": {
            "example": {
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061257,
                "V8": 0.0986979012610507,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762348,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525259478,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114843888824,
                "V27": 0.133558376740387,
                "V28": -0.0210530534538215,
                "Amount": 149.62
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    What the API returns after evaluating a transaction.
    Keeping the response explicit prevents accidentally leaking
    internal model details in the future.
    """
    is_fraud: bool                    # binary decision
    fraud_probability: float          # model confidence (0.0 - 1.0)
    risk_level: str                   # human-readable: LOW / MEDIUM / HIGH
    model_version: str                # tracks which model artifact made this call


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ErrorResponse(BaseModel):
    detail: str
    code: str