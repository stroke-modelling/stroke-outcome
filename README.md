# Stroke outcome


[![GitHub Badge][github-img]][github-link] [![Jupyter Book Badge][jupyterbooks-img]][jupyterbooks-link] 

Toolkit for calculating patient outcomes after stroke depending on time to treatment.

+ __Source code:__ https://github.com/stroke-modelling/stroke-outcome/
+ __Full background and methods:__ https://samuel-book.github.io/stroke_outcome/intro.html
+ __Parent project:__ https://github.com/stroke-modelling

## Motivation in brief:


Disability levels may be measured in various ways. In this project we are using the modified Rankin Scale (mRS). It is a commonly used scale for measuring the degree of disability or dependence in the daily activities of people who have suffered a stroke. The scale runs from 0-6, running from perfect health without symptoms to death.

In addition to mRS, we may calculate utility-weighted mRS (UW-mRS). UW-mRS incorporates both treatment effect and patient perceived quality of life as a single outcome measure for stroke trials.

Patients with ischaemic stroke can be defined by the location of the clot: those with a large vessel occlusion (LVO); and those not with a large vessel occlusion (nLVO). Patients with an nLVO can be treated with thrombolysis (IVT), a clot-busting medication. Patients with an LVO can be treated with IVT and/or thrombectomy (MT), which physically removes the clot. 

This method calculates disability outcome estimates for three patient-treatment cohorts: 
1) nLVO-IVT (patients with an nLVO that are treated with IVT), 
2) LVO-IVT (patients with an LVO that are treated with IVT), 
3) LVO-MT (patients with an LVO that are treated with MT). 

The result is provided as a distribution of disability (with six levels) following reperfusion treatment at any point between these two time stages: 
1) receiving reperfusion treatment as soon as their stroke began (this will be referred to as time of stroke onset, and we will use the terminology “t = 0”), and 
2) receiving reperfusion treatment at the duration after stroke onset where the treatment has no effect (this will be referred to as time of no effect, and we will use the terminology “t = No Effect”).

The method is built by synthesising data from multiple sources to define the distribution of disability for each of the three patient-treatment cohorts at the two time stages (t = 0 & t = No Effect), and we use interpolation to determine the disability distribution at any point in between.

For more details, please see [the online book][jupyterbooks-link].

## Package details:
The package includes the following data:
+ mRS cumulative probability distributions as derived in [the online book][jupyterbooks-link].
+ A selection of utility scores for each mRS level.

Optionally, other data can be used instead of these provided files. The required formats are described in [the continuous outcome demo][jupyternotebook-continuous-link].

The package includes the following processes:
+ __Continuous outcomes:__ Each "patient" uses the population mRS probability distribution rather than being assigned a single mRS score.


## External resources

The following resources are not included within the package files and are accessible on the GitHub repository.

A conda environment file, `environment.yml`, is provided in the [GitHub repository][github-link] for use with the demonstration Jupyter notebooks.

__Continuous outcomes:__
  + [![Jupyter Notebook][jupyternotebook-img]][jupyternotebook-continuous-link] [Demonstration Jupyter notebook][jupyternotebook-continuous-link] 
  + [TO DO]: A worked example of how the code actually works.



[github-img]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[github-link]: https://github.com/stroke-modelling/stroke-outcome/

[jupyterbooks-img]: https://jupyterbook.org/badge.svg
[jupyterbooks-link]: https://samuel-book.github.io/stroke_outcome/intro.html

[jupyternotebook-img]: https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white
[jupyternotebook-continuous-link]: https://github.com/stroke-modelling/stroke-outcome/blob/main/demo/demo_continuous_outcomes.ipynb