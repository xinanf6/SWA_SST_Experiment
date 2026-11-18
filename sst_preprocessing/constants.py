"""Variable contracts and physical constants shared by all pipeline stages."""

BASE_VARIABLES = ("sst", "u10", "v10", "uo", "vo", "t2m", "msl")
FRONT_VARIABLES = ("gradient_x", "gradient_y", "gradient_magnitude")
THERMAL_VARIABLES = ("net_heat_flux", "mixed_layer_depth", "heat_increment")
ENERGY_VARIABLES = ("ssr", "str", "slhf", "sshf")
FEATURE_GROUPS = {
    "base": BASE_VARIABLES,
    "front": FRONT_VARIABLES,
    "thermal": THERMAL_VARIABLES,
}
SECONDS_PER_DAY = 86400.0
EARTH_RADIUS_METERS = 6371000.0
KELVIN_OFFSET = 273.15
SCHEMA_VERSION = 2
STANDARD_DEVIATION_EPSILON = 1e-12

VARIABLE_UNITS = {
    "sst": "degree_Celsius", "t2m": "degree_Celsius",
    "u10": "m s-1", "v10": "m s-1", "uo": "m s-1", "vo": "m s-1",
    "msl": "Pa", "mixed_layer_depth": "m", "net_heat_flux": "W m-2",
    "heat_increment": "degree_Celsius day-1",
    "gradient_x": "degree_Celsius km-1",
    "gradient_y": "degree_Celsius km-1",
    "gradient_magnitude": "degree_Celsius km-1",
    "minimum_ocean_interpolation_weight": "1",
    "minimum_air_interpolation_weight": "1",
    **{name: "J m-2" for name in ENERGY_VARIABLES},
}
