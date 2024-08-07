# Stroke outcome


[![GitHub Badge][github-img]][github-link] [![Jupyter Book Badge][jupyterbooks-img]][jupyterbooks-link] [![PyPI][pypi-img]][pypi-link] [![DOI][doi-img]][doi-link]

Toolkit for calculating patient outcomes after stroke depending on time to treatment.

+ __Source code:__ https://github.com/stroke-modelling/stroke-outcome/
+ __Full background and methods:__ https://samuel-book.github.io/stroke_outcome/intro.html
+ __PyPI package:__ https://pypi.org/project/stroke-outcome/
+ __Parent project:__ https://github.com/stroke-modelling

## ➡️ Get started

This toolkit works with Python versions 3.8 and up.

Install the package with:

    pip install stroke-outcome

And follow the links to the code demonstrations in the "External resources" section below.

## 🏥 Motivation in brief:

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

## 📦 Package details:
The package includes the following data:
+ mRS cumulative probability distributions as derived in [the online book][jupyterbooks-link].
+ A selection of utility scores for each mRS level.

Optionally, other data can be used instead of these provided files. The required formats are described in [the continuous outcome demo][continuous-jupyternotebook-link].

The package includes the following processes:
+ __Continuous outcomes:__ Each "patient" uses the average mRS across a population mRS probability distribution. The average mRS score may be any number between 0 and 6, for example 1.2.
+ __Discrete outcomes:__ Each patient is given a single mRS score out of the population mRS probability distribution. The score must be a whole number from 0 to 6.

The following images summarise the differences between the methods:

![Summary of continuous method. There is an mRS distribution when treated and an mRS distribution when not treated. The patient's mRS is the mean across the distribution.](https://raw.githubusercontent.com/stroke-modelling/stroke-outcome/main/docs/images/continuous_example.png) ![Summary of discrete method. There is an mRS distribution when treated and an mRS distribution when not treated. The patient's mRS is selected from whichever part of the distribution contains a fixed cumulative probability score.](https://raw.githubusercontent.com/stroke-modelling/stroke-outcome/main/docs/images/discrete_example.png)


## 📚 Resources

### Data

The data can be downloaded without needing to use python or to install the `stroke-outcome` package.

To find the data:
1. Go to the [GitHub repository](https://github.com/stroke-modelling/stroke-outcome/)
2. Enter the folder `stroke_outcome`
3. Enter the folder `data` 
4. Click on the name of the file you would like to download
5. On the file preview page, click on the "Download raw file" button 📥

### Docs

The following resources are not included within the package files and are accessible on the GitHub repository.

Each major aspect of the package has a __demonstration__ and a __documentation__ file. The demonstration is a minimal example of running the code, and the documentation uses a simple example to show the concepts behind the methods.

A conda environment file, `environment.yml`, is provided in the GitHub repository for use with the demonstration Jupyter notebooks.

| | **Continuous outcomes** | **Discrete outcomes** | 
|:--|:---------------|:-------------|
| Demo | [![Jupyter Notebook][jupyternotebook-img]][continuous-jupyternotebook-link]  | [![Jupyter Notebook][jupyternotebook-img]][discrete-jupyternotebook-link]  | 
| Docs | [![Jupyter Notebook][jupyternotebook-img]][continuous-documentation-link] | [![Jupyter Notebook][jupyternotebook-img]][discrete-documentation-link] | 



[github-img]: https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white
[github-link]: https://github.com/stroke-modelling/stroke-outcome/

[pypi-img]: https://img.shields.io/pypi/v/stroke-outcome?label=pypi%20package
[pypi-link]: https://pypi.org/project/stroke-outcome/

[doi-img]: https://zenodo.org/badge/673336639.svg
[doi-link]: https://zenodo.org/badge/latestdoi/673336639

[jupyterbooks-img]: https://jupyterbook.org/badge.svg
[jupyterbooks-link]: https://samuel-book.github.io/stroke_outcome/intro.html

[jupyternotebook-img]: https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white

[continuous-jupyternotebook-link]: https://github.com/stroke-modelling/stroke-outcome/blob/main/demo/demo_continuous_outcomes.ipynb
[continuous-documentation-link]: https://github.com/stroke-modelling/stroke-outcome/blob/main/docs/docs_continuous_outcome.ipynb

[discrete-jupyternotebook-link]: https://github.com/stroke-modelling/stroke-outcome/blob/main/demo/demo_discrete_outcomes.ipynb
[discrete-documentation-link]: https://github.com/stroke-modelling/stroke-outcome/blob/main/docs/docs_discrete_outcome.ipynb