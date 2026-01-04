"""
Catalogue of available Darts model instances for the project for easy import in training pipeline.

The dictionary MODEL_CATALOGUE maps string names to model factory functions. It can be found at the bottom
of this file.

Each model instance represents a model with given set of hyperparameters.
Models can be swapped in and out of the training pipeline by changing the
imported model instance here.

MODEL NAMING CONVENTION:
Model names are constructed as follows:
<model_class>_<lags target><lags past covariates><lags future covariates (tuple)>_<ENCODER>

For more information on available models and hyperparameters, see:
https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html

>> NOTE: It is important each MODEL_CATALOGUE value is a function reference (without '()')
    and not a function call (with '()').
   Why?
    1) If it is a function call, the model will be instantiated upon import of the catalogue,
        and become mutable. and changes to the model instance during runtime would be persisted here.
    2) This avoids instantiating all models when importing the catalogue.
"""

from darts.models import RandomForest
from darts.dataprocessing.transformers import Scaler

RANDOM_FOREST_DEFAULT_ESTIMATORS = 500
RANDOM_FOREST_DEFAULT_MAX_DEPTH = 10
ENCODERS = {
    "cyclic_day": {"cyclic": {"future": ["day_of_week"]}},
    "cyclic_day_scaled": {
        "cyclic": {"future": ["day_of_week"]},
        "transformer": Scaler(),
    },
    "cyclic_day_month": {"cyclic": {"future": ["day_of_week", "month"]}},
    "cyclic_day_month_scaled": {
        "cyclic": {"future": ["day_of_week", "month"]},
        "transformer": Scaler(),
    },
    "dt_attribute_day": {"datetime_attribute": {"future": ["day_of_week"]}},
    "dt_attribute_day_scaled": {
        "datetime_attribute": {"future": ["day_of_week"]},
        "transformer": Scaler(),
    },
    "dt_attribute_day_month": {
        "datetime_attribute": {"future": ["day_of_week", "month"]}
    },
    "dt_attribute_day_month_scaled": {
        "datetime_attribute": {"future": ["day_of_week", "month"]},
        "transformer": Scaler(),
    },
}


def random_forest_1111() -> RandomForest:
    model = RandomForest(
        n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,
        max_depth=RANDOM_FOREST_DEFAULT_MAX_DEPTH,
        lags=1,
        lags_past_covariates=1,
        lags_future_covariates=(1, 1),
        add_encoders=None,
    )
    return model


def random_forest_7777_cyclic_day_scaled() -> RandomForest:
    model = RandomForest(
        n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,
        max_depth=RANDOM_FOREST_DEFAULT_MAX_DEPTH,
        lags=7,
        lags_past_covariates=7,
        lags_future_covariates=(7, 7),
        add_encoders=ENCODERS["cyclic_day_scaled"],
    )
    return model


def random_forest_7777_cyclic_day_month_scaled() -> RandomForest:
    model = RandomForest(
        n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,
        max_depth=RANDOM_FOREST_DEFAULT_MAX_DEPTH,
        lags=7,
        lags_past_covariates=7,
        lags_future_covariates=(7, 7),
        add_encoders=ENCODERS["cyclic_day_month_scaled"],
    )
    return model


def random_forest_7777_dt_attribute_day_scaled() -> RandomForest:
    model = RandomForest(
        n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,
        max_depth=RANDOM_FOREST_DEFAULT_MAX_DEPTH,
        lags=7,
        lags_past_covariates=7,
        lags_future_covariates=(7, 7),
        add_encoders=ENCODERS["dt_attribute_day_scaled"],
    )
    return model


def random_forest_7777_dt_attribute_day_month_scaled() -> RandomForest:
    model = RandomForest(
        n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,
        max_depth=RANDOM_FOREST_DEFAULT_MAX_DEPTH,
        lags=7,
        lags_past_covariates=7,
        lags_future_covariates=(7, 7),
        add_encoders=ENCODERS["dt_attribute_day_month_scaled"],
    )
    return model


def random_forest_7777_dt_attribute_day_month() -> RandomForest:
    model = RandomForest(
        n_estimators=RANDOM_FOREST_DEFAULT_ESTIMATORS,
        max_depth=RANDOM_FOREST_DEFAULT_MAX_DEPTH,
        lags=7,
        lags_past_covariates=7,
        lags_future_covariates=(7, 7),
        add_encoders=ENCODERS["dt_attribute_day_month"],
    )
    return model


MODEL_CATALOGUE = {
    "random_forest_1111": random_forest_1111,
    "random_forest_7777_cyclic_day_scaled": random_forest_7777_cyclic_day_scaled,
    "random_forest_7777_cyclic_day_month_scaled": random_forest_7777_cyclic_day_month_scaled,
    "random_forest_7777_dt_attribute_day_scaled": random_forest_7777_dt_attribute_day_scaled,
    "random_forest_7777_dt_attribute_day_month_scaled": random_forest_7777_dt_attribute_day_month_scaled,
    "random_forest_7777_dt_attribute_day_month": random_forest_7777_dt_attribute_day_month,
}
