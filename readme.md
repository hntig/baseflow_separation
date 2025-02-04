# baseflow_separation

## Description

This repository contains scripts developed for baseflow separation, as part of my PhD and the methods are described in Section 4.1.3 of my thesis titled "Urban hydrogeological modelling and uncertainty characterisation to evaluate the risk of groundwater flooding" (link to be added).

The scripts were adapted from <https://github.com/samzipper/GlobalBaseflow> and re-written in Python.

## Data sources and format

Daily river flow data were obtained from <https://environment.data.gov.uk/hydrology/explore> (last accessed: 27 December 2024) and followed the format displayed in `river_flow_daily/sample_file.csv`. The station ID was manually added to file names and the final file names containing the daily river flow data followed the format: `STATION_ID-STATION_NAME-flow-daily-Qualified.csv`
