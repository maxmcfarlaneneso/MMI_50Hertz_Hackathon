import glob
import shutil
import os
import europeanpower.const as const
from europeanpower.const import Countries, TargetCountries, WeatherParameterAlias, CountryAbb, Targets

# Base directory for the project
DIR = os.path.dirname(__file__)

# Paths to certificates and API keys
cert_path = os.path.join(DIR, 'neso-zscaler-issuing-ca.neso.energy.crt')
entsoe_api_key = os.path.join(DIR, 'entsoe_api_key.json')
odre_api_key = os.path.join(DIR, 'odre_api_key.json')
blob_storage_account_credentials = os.path.join(DIR, 'neso_blob_api_key.json')
nuts_2024_boundary_file = os.path.join(DIR, "NUTS_RG_03M_2024_3035.gpkg")
nuts_2016_boundary_file = os.path.join(DIR, "NUTS_RG_60M_2016_3035.gpkg")
nuts_2021_boundary_file = os.path.join(DIR, "NUTS_RG_03M_2021_3035.gpkg")
nuts_boundary_file = nuts_2021_boundary_file
nuts_demographics_gpd_file = os.path.join(DIR, "Europe_NUTS_2_Demographics_and_Boundaries.geojson")
nuts_target_demographics_gpd_file = os.path.join(DIR, "Targets_Europe_NUTS_2_Demographics_and_Boundaries.geojson")
enapsys_api_key_path = os.path.join(DIR, "enapsys_api_key.json")

pv_capacity_path = os.path.join(DIR, 'solar_pv_capacity', 'installed-solar-pv-capacity.csv')
wind_capacity_path = os.path.join(DIR, 'wind_capacity', 'cumulative-installed-wind-energy-capacity-gigawatts.csv')

# Folders
results_dir = os.path.join(DIR, 'results')
os.makedirs(results_dir, exist_ok=True)

gam_dir = os.path.join(DIR, 'r_gams')
os.makedirs(gam_dir, exist_ok=True)

gam_tmp_dir = os.path.join(gam_dir, 'tmp')
os.makedirs(gam_tmp_dir, exist_ok=True)

# -------------------------------------------------------------------
# Define base and raw directories for each country
# -------------------------------------------------------------------
country_base_dirs = {country: os.path.join(DIR, country) for country in Countries}
country_raw_dirs = {country: os.path.join(base, "raw") for country, base in country_base_dirs.items()}

era5_dir = os.path.join(DIR, 'ERA5')
era5_combined_dir = os.path.join(era5_dir, 'combined')

covid_dir = os.path.join(DIR, 'covid')
covid_paths = {}
covid_paths['stay_at_home'] = os.path.join(covid_dir, 'stay-at-home-covid.csv')
covid_paths['stringency_index'] = os.path.join(covid_dir, 'covid-19-stringency-index.csv')

# -------------------------------------------------------------------
# # Define raw data paths manually per country
# # EXAMPLES: Access France's raw data
# -------------------------------------------------------------------
hardcoded_paths_ = {
    const.Targets.solar: None,
    const.Targets.wind: None,
    const.Targets.demand: None,
    'demand': None,
    'generation': None,
    'capacity': None,
    'demand_forecast_enappsys': None,
    'demand_enappsys': None,
    'wind_meta': None,
    'solar_meta': None,
    'wind_capacity_enappsys': None,
    'solar_capacity_enappsys': None,
    'wind_capacity_ourworldindata': None,
    'solar_capacity_ourworldindata': None,
    'renewables_capacity_ourworldindata': None,
}

programmatic_paths_ = [
    WeatherParameterAlias.surface_solar_radiation_downwards,
    WeatherParameterAlias.temperature_2m,
    WeatherParameterAlias.wind_10m_speed,
    'sun_times',
    'ERA5',
]

_init_standard_keys = lambda c: {k: os.path.join(country_raw_dirs[c], f"{c}_{k}_raw.parquet") for k in programmatic_paths_}

raw_paths = {c: {**hardcoded_paths_, **_init_standard_keys(c)} for c in Countries}

for country in raw_paths.keys():
    country = str(country)
    base_dir = country_base_dirs[country]
    raw_dir = country_raw_dirs[country]
    regional_raw_dir = os.path.join(raw_dir, "regional")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(regional_raw_dir, exist_ok=True)

    raw_paths[country]['_raw_path'] = raw_dir

    raw_paths[country]['capacity'] = os.path.join(country_raw_dirs[country], f"capacity_enappsys-{CountryAbb[country]}.parquet")

    raw_paths[country]['renewables'] = os.path.join(
        country_raw_dirs[country], f"renewables_enappsys-{CountryAbb[country]}.parquet"
    )

    raw_paths[country]['curtailment'] = os.path.join(
        country_raw_dirs[country], f"curtailment_enappsys-{CountryAbb[country]}.parquet"
    )

    raw_paths[country][const.Targets.demand] = os.path.join(country_raw_dirs[country], f"demand-{CountryAbb[country]}.csv")
    raw_paths[country]['da_demand_forecast'] = os.path.join(
        country_raw_dirs[country], f"dayahead-forecast-demand-{CountryAbb[country]}.csv"
    )

    raw_paths[country]['ninja_pv_generation'] = os.path.join(
        raw_dir, f"ninja-pv-country-{CountryAbb[country]}-national-merra2.csv"
    )
    raw_paths[country]['ninja_t2m_area_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-temperature_area_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_t2m_pop_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-temperature_pop_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_tpp_pop_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-precipitation_pop_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_cc_area_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-cloud_cover_area_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_si_area_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-irradiance_surface_area_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_ws_area_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-wind_speed_area_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_ws_pop_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-wind_speed_pop_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_si_pop_weighted'] = os.path.join(
        raw_dir, f"ninja-weather-country-{CountryAbb[country]}-irradiance_surface_pop_wtd-merra2.csv"
    )
    raw_paths[country]['ninja_BAIT'] = os.path.join(raw_dir, f"ninja-demand-country-{CountryAbb[country]}-BAIT-merra2.csv")
    raw_paths[country]['ninja_CDD'] = os.path.join(raw_dir, f"ninja-demand-country-{CountryAbb[country]}-CDD-merra2.csv")
    raw_paths[country]['ninja_HDD'] = os.path.join(raw_dir, f"ninja-demand-country-{CountryAbb[country]}-HDD-merra2.csv")

    raw_paths[country]['_ERA5_NUTS0_gridbox_indexed'] = os.path.join(raw_dir, f"{country}_ERA5_0_NUTS0_indexed")
    os.makedirs(raw_paths[country]['_ERA5_NUTS0_gridbox_indexed'], exist_ok=True)

    raw_paths[country]['ERA5'] = os.path.join(raw_dir, f"{country}_ERA5_raw.parquet")
    raw_paths[country]['ERA5(region)'] = os.path.join(regional_raw_dir, f"{country}_{'{}'}_ERA5_raw.parquet")

    raw_paths[country]['wind_outturn'] = os.path.join(raw_dir, f"{country}_wind_outturn.csv")
    raw_paths[country]['wind_cap'] = os.path.join(raw_dir, f"{country}_installed_wind_cap.csv")
    raw_paths[country]['weather_sim'] = os.path.join(raw_dir, "10day_weather_simulations.parquet")
    raw_paths[country]['gam_model'] = os.path.join(gam_dir, country, "fitted_model.RDS")
    os.makedirs(os.path.join(gam_dir, country), exist_ok=True)

# Add other countries and their raw paths here...

raw_paths[Countries.France].update(
    {
        "odre_generation": os.path.join(country_raw_dirs[Countries.France], "eco2mix-national-cons-def.parquet"),
        "odre_capacity": os.path.join(country_raw_dirs[Countries.France], "parc-national-annuel-prod-eolien-solaire.parquet"),
        "odre_demand": os.path.join(country_raw_dirs[Countries.France], "eco2mix-regional-cons-def.parquet"),
        # "wind_bmu_gridpoints": fra_data.wind_bmu_locations_path,
    }
)

# -------------------------------------------------------------------
# # Create directories and define processed paths based on raw keys
# # EXAMPLE: Access France's processed demand data
# fr_demand_processed = country_data_paths[COUNTRIES.France]["processed"]["demand"]
# -------------------------------------------------------------------
processed_paths = {}
for country, raw_paths_ in raw_paths.items():
    country = str(country)
    base_dir = country_base_dirs[country]
    processed_dir = os.path.join(base_dir, "processed")
    models_dir = os.path.join(processed_dir, "models")
    gridpoints_processed_dir = os.path.join(processed_dir, "gridpoints")
    regional_processed_dir = os.path.join(processed_dir, "regional")

    os.makedirs(gridpoints_processed_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(regional_processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Store both raw and processed paths
    processed_paths[country] = {key: os.path.join(processed_dir, f"{country}_{key}_processed.parquet") for key in raw_paths_}

    processed_paths[country]['_processed_path'] = processed_dir
    processed_paths[country]['_models_dir'] = models_dir
    processed_paths[country]['_regional_dir'] = regional_processed_dir

    for supply_target in const.SupplyTargets:
        processed_paths[country][f'{supply_target}_location_scale_data'] = os.path.join(
            processed_dir, f"{country}_{supply_target}_location_scale_data.parquet"
        )

    processed_paths[country]['demand_location_scale_data'] = os.path.join(
        processed_dir, f"{country}_demand_location_scale_data.geojson"
    )

    processed_paths[country]['_ERA5_NUTS0_boundary_indexed'] = os.path.join(processed_dir, f"{country}_ERA5_1_NUTS0_indexed")
    os.makedirs(processed_paths[country]['_ERA5_NUTS0_boundary_indexed'], exist_ok=True)

    processed_paths[country]['ERA5(region)_indexed'] = os.path.join(
        regional_processed_dir, f"{country}_{'{}'}_ERA5_1_NUTS1_indexed.parquet"
    )
    processed_paths[country]['ERA5_NUTS2_1_indexed'] = os.path.join(
        regional_processed_dir, f"{country}_{'{}'}_ERA5_1_NUTS2_indexed.parquet"
    )

    processed_paths[country]['ERA5_NUTS0'] = os.path.join(processed_dir, f"{country}_ERA5_2_NUTS0_processed.parquet")
    processed_paths[country]['ERA5_NUTS2_unweighted'] = os.path.join(
        regional_processed_dir, f"{country}_{'{}'}_ERA5_2_NUTS2_unweighted.parquet"
    )
    processed_paths[country]['ERA5_NUTS2_weighted'] = os.path.join(processed_dir, f"{country}_ERA5_2_NUTS2_weighted.parquet")

    for target in const.Targets:

        processed_paths[country][f'ERA5_1_gridpoints_{target}_indexed'] = os.path.join(
            gridpoints_processed_dir, f"{country}_ERA5_1_gridpoints_{target}_indexed.parquet"
        )
        processed_paths[country][f'ERA5_2_gridpoints_{target}_split'] = os.path.join(
            gridpoints_processed_dir, f"{country}_ERA5_2_gridpoints_{target}_split.parquet"
        )
        processed_paths[country][f'ERA5_2_gridpoints_{target}_weighted'] = os.path.join(
            gridpoints_processed_dir, f"{country}_ERA5_2_gridpoints_{target}_weighted.parquet"
        )
        processed_paths[country][f'{target}_training_testing_set'] = os.path.join(
            processed_dir, f"{country}-{target}_training_testing_set.parquet"
        )

    processed_paths[country]['regional_demand'] = os.path.join(processed_dir, f"{country}-regional_demand_processed.parquet")
    # processed_paths[country]['national_demand'] = os.path.join(processed_dir, f"{country}-national_demand_processed.parquet")
    processed_paths[country]['outturn_era5'] = os.path.join(processed_dir, f"{country}_outturn_era5.csv")

    processed_paths[country][f'{Targets.solar}_projections'] = os.path.join(
        processed_dir, f"{country}_{Targets.solar}_projections.parquet"
    )
    processed_paths[country][f'{Targets.wind}_projections'] = os.path.join(
        processed_dir, f"{country}_{Targets.wind}_projections.parquet"
    )
    processed_paths[country][f'{Targets.demand}_projections'] = os.path.join(
        processed_dir, f"{country}_{Targets.demand}_projections.parquet"
    )

processed_paths['holidays'] = os.path.join(DIR, 'holidays.csv')

results_paths = {}
for country in Countries:
    country = str(country)
    results_paths[country] = {}
    results_paths[country]["gam_prediction"] = os.path.join(results_dir, country, "gam_results.csv")
    results_paths[country]["gam_quantiles"] = os.path.join(results_dir, country, "gam_quantiles.csv")

export_for_onedrive = False
if export_for_onedrive:
    __raw_files = list(
        filter(
            lambda f: '.py' not in f and '.md' not in f and 'key.json' not in f and 'oneDrive' not in f,
            glob.glob(f'{DIR}/*.*', recursive=False),
        )
    )
    [os.makedirs(os.path.dirname(f.replace(DIR, f'{DIR}\\oneDrive\\')), exist_ok=True) for f in __raw_files]
    [shutil.copy(f, f.replace(DIR, f'{DIR}\\oneDrive')) for f in __raw_files]

    __raw_files = glob.glob(f'{DIR}/**/raw/**.*', recursive=True)
    __raw_files = list(filter(lambda f: 'oneDrive' not in f, __raw_files))
    [os.makedirs(d, exist_ok=True) for d in [os.path.dirname(f.replace(DIR, f'{DIR}\\oneDrive')) for f in __raw_files]]
    [shutil.copy(f, os.path.dirname(f.replace(DIR, f'{DIR}\\oneDrive'))) for f in __raw_files]

    __raw_files = list(
        filter(
            lambda f: '.py' not in f and '.md' not in f and 'key.json' not in f,
            glob.glob(f'{DIR}/datetime_/*.*', recursive=False),
        )
    )
    [os.makedirs(d, exist_ok=True) for d in [os.path.dirname(f.replace(DIR, f'{DIR}\\oneDrive')) for f in __raw_files]]
    [shutil.copy(f, os.path.dirname(f.replace(DIR, f'{DIR}\\oneDrive'))) for f in __raw_files]

    for target in [const.Targets.wind, const.Targets.solar]:
        for country in const.TargetCountries:
            gridpoint_output_key = const.dict_target_era5_2_gridpoints_weighted_key[target]
            era5_gridpoints_path = processed_paths[country][gridpoint_output_key]
            os.makedirs(os.path.dirname(era5_gridpoints_path.replace(DIR, f'{DIR}\\oneDrive')), exist_ok=True)
            shutil.copy(era5_gridpoints_path, os.path.dirname(era5_gridpoints_path.replace(DIR, f'{DIR}\\oneDrive')))
