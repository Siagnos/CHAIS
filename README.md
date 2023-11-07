# CHAIS
CHAIS is a method for the non-invasive inference of central hemodynamics from wearable ECG data.

We developed CHAIS using the procedure described in the figure below. (a) CHAIS is trained on an internal MGH development dataset, using 10s, single lead recordings (Lead I) extracted from the 12-lead electrocardiogram from patients with known cardiac hemodynamic measurements; (b) CHAIS is evaluated on an internal holdout dataset and an external validation dataset containing patients with ECG data (only lead I is used) and known cardiac hemodynamics; (c) the model is then prospectively evaluated using ECG data acquired from patients who wore a commercially available wearable ECG monitor. 
![Diagram of model development workflow](https://github.com/mit-ccrg/CHAIS/blob/main/figures/figure1.png)
