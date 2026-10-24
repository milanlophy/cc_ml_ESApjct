
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "disaster_data.csv"

FEATURES = [
    "rainfall_mm",
    "elevation_m",
    "slope_degrees",
    "river_distance_km",
    "historical_events",
    "population_density",
    "economic_vulnerability",
]

RISK_LABELS = ["Low", "Medium", "High"]


def generate_dataset():
    """Create an illustrative dataset for learning purposes."""
    rng = np.random.default_rng(42)
    n = 2500

    rainfall = rng.uniform(20, 500, n)
    elevation = rng.uniform(0, 1200, n)
    slope = rng.uniform(0, 45, n)
    river_distance = rng.uniform(0.1, 15, n)
    history = rng.integers(0, 8, n)
    population = rng.uniform(100, 12000, n)
    vulnerability = rng.uniform(1, 10, n)

    # Illustrative scoring rule, NOT a validated disaster model.
    score = (
        0.27 * rainfall / 500
        + 0.17 * (1 - elevation / 1200)
        + 0.15 * slope / 45
        + 0.17 * (1 - river_distance / 15)
        + 0.12 * history / 7
        + 0.05 * population / 12000
        + 0.07 * vulnerability / 10
        + rng.normal(0, 0.035, n)
    )

    risk = np.where(
        score < 0.34, "Low",
        np.where(score < 0.55, "Medium", "High")
    )

    return pd.DataFrame({
        "rainfall_mm": rainfall.round(1),
        "elevation_m": elevation.round(1),
        "slope_degrees": slope.round(1),
        "river_distance_km": river_distance.round(2),
        "historical_events": history,
        "population_density": population.round(0).astype(int),
        "economic_vulnerability": vulnerability.round(1),
        "risk_level": risk,
    })


def load_dataset():
    if not DATA_FILE.exists():
        generate_dataset().to_csv(DATA_FILE, index=False)

    data = pd.read_csv(DATA_FILE)
    required = FEATURES + ["risk_level"]

    if not set(required).issubset(data.columns):
        raise ValueError(
            "The dataset has missing columns. "
            "Restore the generated disaster_data.csv file."
        )

    data = data[required].dropna()
    data["risk_level"] = data["risk_level"].astype(str).str.title()
    data = data[data["risk_level"].isin(RISK_LABELS)]

    if len(data) < 30 or data["risk_level"].nunique() < 2:
        raise ValueError(
            "The dataset needs at least 30 valid rows and two risk classes."
        )

    return data


data = load_dataset()

X = data[FEATURES]
y = data["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)

# This measures agreement with synthetic labels, not real-world accuracy.
test_accuracy = accuracy_score(y_test, model.predict(X_test))


def get_risk_description(risk):
    descriptions = {
        "Low": (
            "The submitted indicators fall into the model's low-risk "
            "category. Continue routine monitoring and follow official advice."
        ),
        "Medium": (
            "The submitted indicators fall into the model's medium-risk "
            "category. Review local hazards and preparedness measures."
        ),
        "High": (
            "The submitted indicators fall into the model's high-risk "
            "category. Check official warnings and local emergency guidance."
        ),
    }
    return descriptions[risk]


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    defaults = {
        "area_name": "",
        "rainfall_mm": "250",
        "elevation_m": "100",
        "slope_degrees": "15",
        "river_distance_km": "2",
        "historical_events": "2",
        "population_density": "3000",
        "economic_vulnerability": "5",
    }

    form_data = defaults.copy()

    if request.method == "POST":
        for key in form_data:
            form_data[key] = request.form.get(key, "").strip()

        try:
            area_name = form_data["area_name"] or "Unnamed area"

            numeric_values = {
                "rainfall_mm": (20, 500),
                "elevation_m": (0, 1200),
                "slope_degrees": (0, 45),
                "river_distance_km": (0.1, 15),
                "historical_events": (0, 7),
                "population_density": (100, 12000),
                "economic_vulnerability": (1, 10),
            }

            values = {}

            for key, (minimum, maximum) in numeric_values.items():
                raw_value = form_data[key]

                if not raw_value:
                    raise ValueError(f"Please enter {key.replace('_', ' ')}.")

                value = float(raw_value)

                if not np.isfinite(value):
                    raise ValueError("Please enter valid numeric values.")

                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"{key.replace('_', ' ').title()} must be "
                        f"between {minimum} and {maximum}."
                    )

                if key == "historical_events" and not value.is_integer():
                    raise ValueError(
                        "Historical events must be a whole number."
                    )

                if key == "population_density" and not value.is_integer():
                    raise ValueError(
                        "Population density must be a whole number."
                    )

                values[key] = value

            input_frame = pd.DataFrame(
                [[values[feature] for feature in FEATURES]],
                columns=FEATURES,
            )

            prediction = model.predict(input_frame)[0]
            probabilities = model.predict_proba(input_frame)[0]
            probability_map = dict(zip(model.classes_, probabilities))

            result = {
                "area_name": area_name,
                "risk": prediction,
                "description": get_risk_description(prediction),
                "probability": round(
                    float(probability_map.get(prediction, 0)) * 100, 1
                ),
                "probabilities": [
                    {
                        "label": label,
                        "value": round(
                            float(probability_map.get(label, 0)) * 100, 1
                        ),
                    }
                    for label in RISK_LABELS
                ],
                "values": values,
            }

        except ValueError as exc:
            error = str(exc)
        except Exception:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Check the inputs and try again."

    # Distribution of the dataset's illustrative labels.
    distribution = (
        data["risk_level"]
        .value_counts()
        .reindex(RISK_LABELS, fill_value=0)
        .to_dict()
    )

    # Recent rows are examples, not actual visitor or disaster observations.
    sample_records = data.head(6)[FEATURES + ["risk_level"]].round(2)
    sample_records = sample_records.to_dict(orient="records")

    return render_template(
        "index.html",
        result=result,
        error=error,
        form_data=form_data,
        distribution=distribution,
        sample_records=sample_records,
        total_records=len(data),
        test_accuracy=round(test_accuracy * 100, 1),
    )


if __name__ == "__main__":
    app.run(debug=True)
