
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2500

rainfall = rng.uniform(20, 500, n)
elevation = rng.uniform(0, 1200, n)
slope = rng.uniform(0, 45, n)
river_distance = rng.uniform(0.1, 15, n)
history = rng.integers(0, 8, n)
population = rng.uniform(100, 12000, n)
vulnerability = rng.uniform(1, 10, n)

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

df = pd.DataFrame({
    "rainfall_mm": rainfall.round(1),
    "elevation_m": elevation.round(1),
    "slope_degrees": slope.round(1),
    "river_distance_km": river_distance.round(2),
    "historical_events": history,
    "population_density": population.round(0).astype(int),
    "economic_vulnerability": vulnerability.round(1),
    "risk_level": risk
})

df.to_csv("disaster_data.csv", index=False)

print("Dataset created successfully!")
print("Total records:", len(df))
print(df.head())
