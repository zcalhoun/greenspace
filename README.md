# Greenspace Analysis
This repository contains code needed to assess the predictive benefit of including high-resolution vegetation data on air temperature.



# About the data
We use the greenspace data, traversals from NOAA's urban heat island campaign, and compare with NDVI data from Sentinel.


## Greenspace data
This is the data from the paper [add link]

No. | greenspace_class |
|:--:|:--|
0   | Not vegetated
2	| Evergreen broadleaved forest    
4	| Deciduous broadleaved forest
6	| Evergreen needleleaved forest
8	| Deciduous needleleaved forest
10	| Mixed-leaf forest
12	| Evergreen shrubland
13	| Deciduous shrubland
14	| Grassland


## NDVI data
We pull 10-m Sentinel-2 data to collect NDVI and albedo.

## Method
We adapt the method from Calhoun et al. [add link].