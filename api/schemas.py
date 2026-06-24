"""Pydantic schemas for request and response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class DiabetesFeatures(BaseModel):
    """Input features for diabetes prediction (Pima Indians dataset).

    All fields correspond to the original dataset columns used during training.
    Validation rules reflect clinically plausible ranges.
    """

    Pregnancies: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of times pregnant (0–20)",
    )
    Glucose: float = Field(
        ...,
        gt=0,
        le=300,
        description="Plasma glucose concentration (mg/dL), must be > 0",
    )
    BloodPressure: float = Field(
        ...,
        ge=0,
        le=200,
        description="Diastolic blood pressure (mm Hg)",
    )
    SkinThickness: float = Field(
        ...,
        ge=0,
        le=100,
        description="Triceps skin fold thickness (mm)",
    )
    Insulin: float = Field(
        ...,
        ge=0,
        le=1000,
        description="2-Hour serum insulin (mu U/ml)",
    )
    BMI: float = Field(
        ...,
        gt=0,
        le=80,
        description="Body mass index (weight in kg / height in m^2), must be > 0",
    )
    DiabetesPedigreeFunction: float = Field(
        ...,
        ge=0.0,
        le=3.0,
        description="Diabetes pedigree function score (0.0–3.0)",
    )
    Age: int = Field(
        ...,
        ge=1,
        le=120,
        description="Age in years (1–120)",
    )

    @field_validator("Glucose")
    @classmethod
    def glucose_positive(cls, v: float) -> float:
        """Ensure glucose is a positive, clinically meaningful value."""
        if v <= 0:
            raise ValueError("Glucose must be greater than 0")
        return v

    @field_validator("BMI")
    @classmethod
    def bmi_positive(cls, v: float) -> float:
        """Ensure BMI is a positive value."""
        if v <= 0:
            raise ValueError("BMI must be greater than 0")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 2,
                "Glucose": 120,
                "BloodPressure": 70,
                "SkinThickness": 25,
                "Insulin": 80,
                "BMI": 28.5,
                "DiabetesPedigreeFunction": 0.35,
                "Age": 33,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for a successful prediction.

    Attributes:
        prediction: Predicted class (0 = no diabetes, 1 = diabetes).
        probability: Probability of positive class (if available).
        model: Name / type of the model that produced the prediction.
    """

    prediction: int = Field(..., description="Predicted outcome (0 or 1)")
    probability: float | None = Field(
        default=None,
        description="Probability of positive class (0.0–1.0), if model supports it",
    )
    model: str = Field(..., description="Model name used for prediction")


class HealthResponse(BaseModel):
    """Response schema for the health-check endpoint.

    Attributes:
        status: Always 'ok' when the service is reachable.
        baseline_model_loaded: Whether the sklearn baseline model is loaded.
        automl_model_loaded: Whether the AutoGluon model is loaded.
    """

    status: str = Field(default="ok")
    baseline_model_loaded: bool
    automl_model_loaded: bool