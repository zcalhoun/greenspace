# Greenspace Analysis
This repository contains code needed to assess the predictive benefit of including high-resolution vegetation data on air temperature. Ultimately, there are two experiments contained within this repository:
1. [Model comparison](experiments/01_Bayesian_Optimization/): This experiment determines the extent to which including the additional greenspace information improves our ability to estimate temperature.
2. [Causal estimation](experiments/02_Causal_Estimates/): This experiment fits the causal model on a reduced set of covariates, using reasonable hyperparameters so we can compare results across cities. This is the experiment ultimately used to produce maps for the final paper.


# Datasets used
We use the greenspace data, traversals from NOAA's urban heat island campaign, NDVI/derived Albedo from Sentinel, and the National Land Cover Database.

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


## Method
We adapt the method from [Calhoun et al](https://www.nature.com/articles/s41598-023-50981-w).

The only significant change from the original approach is that we re-wrote the code to use GPyTorch. The benefit of this approach is that we can constrain the lengthscales for the unobserved confound term. Aside from this change, the data used is different.