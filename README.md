# Towards Practical Federated Learning and Evaluation for Medical Prediction Models

This repository contains the data and code associated with the paper: [URL]().

## Research Papers

To ensure full reproducibility, we provide the complete list of datasets used in this paper:

1. **Usefulness of Bi-Parametric Magnetic Resonance Imaging with b=1,800 s/mm2 Diffusion-Weighted Imaging for Diagnosing Clinically Significant Prostate Cancer**  
   [Korea1](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/R35JV9)
   

2. **Impact of Ultrasonographic Findings on Cancer Detection Rate during Magnetic Resonance Image/ Ultrasonography Fusion-Targeted Prostate Biopsy**  
   [Korea2](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/P5LVV9)

3. **Lesion volume predicts prostate cancer risk and aggressiveness: validation of its value alone and matched with prostate imaging reporting and data system score**  
   [Italy](https://data.mendeley.com/datasets/9x87km32n6/1)

4. **The Role of Digital Rectal Examination Prostate Volume Category in the Early Detection of Prostate Cancer: Its Correlation with the Magnetic Resonance Imaging Prostate Volume**  
   [Spain](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DAZPCX)

5. **Data on the detection of clinically significant prostate cancer by magnetic resonance imaging (MRI)-guided targeted and systematic biopsy**  
   [Germany1](https://zenodo.org/records/6834906)

6. **Assessing the accuracy of multiparametric MRI to predict clinically significant prostate cancer in biopsy naïve men across racial/ethnic groups**  
   [USA1](https://figshare.com/articles/dataset/Additional_file_1_of_Assessing_the_accuracy_of_multiparametric_MRI_to_predict_clinically_significant_prostate_cancer_in_biopsy_na_ve_men_across_racial_ethnic_groups/20336747)

7. **Distribution of Prostate Imaging Reporting and Data System score and diagnostic accuracy of magnetic resonance imaging–targeted biopsy: comparison of an Asian and European cohort**  
   [China1 + Netherlands](https://data.mendeley.com/datasets/5sxfpyzmx4/1)

7. **Modified Predictive Model and Nomogram by Incorporating Prebiopsy Biparametric Magnetic Resonance Imaging With Clinical Indicators for Prostate Biopsy Decision Making**  
   [China2](https://figshare.com/articles/dataset/DataSheet_1_Modified_Predictive_Model_and_Nomogram_by_Incorporating_Prebiopsy_Biparametric_Magnetic_Resonance_Imaging_With_Clinical_Indicators_for_Prostate_Biopsy_Decision_Making_zip/16609681?backTo=%2Fcollections%2FModified_Predictive_Model_and_Nomogram_by_Incorporating_Prebiopsy_Biparametric_Magnetic_Resonance_Imaging_With_Clinical_Indicators_for_Prostate_Biopsy_Decision_Making%2F5615185&file=30741169)

9. **Prediction of significant prostate cancer in biopsy-naïve men: Validation of a novel risk model combining MRI and clinical parameters and comparison to an ERSPC risk calculator and PI-RADS**  
   [Germany2 + UK](https://plos.figshare.com/articles/dataset/Prediction_of_significant_prostate_cancer_in_biopsy-na_ve_men_Validation_of_a_novel_risk_model_combining_MRI_and_clinical_parameters_and_comparison_to_an_ERSPC_risk_calculator_and_PI-RADS/9733748?file=17431067)

10. **Comparison of Multiparametric MRI Scoring Systems and the Impact on Cancer Detection in Patients Undergoing MR US Fusion Guided Prostate Biopsies**  
   [USA2](https://plos.figshare.com/articles/dataset/_Comparison_of_Multiparametric_MRI_Scoring_Systems_and_the_Impact_on_Cancer_Detection_in_Patients_Undergoing_MR_US_Fusion_Guided_Prostate_Biopsies_/1612831?file=2578419)

11. **Repeat multiparametric MRI in prostate cancer patients on active surveillance**  
   [Finland](https://plos.figshare.com/articles/dataset/Repeat_multiparametric_MRI_in_prostate_cancer_patients_on_active_surveillance/5736456?file=10103526)

12. **Efficacy of plasma atherogenic index in predicting malignancy in the presence of Prostate Imaging–Reporting and Data System 3 (PI-RADS 3) prostate lesions**  
   [Turkey](https://figshare.com/articles/dataset/Efficacy_of_plasma_atherogenic_index_in_predicting_malignancy_in_the_presence_of_Prostate_Imaging-Reporting_and_Data_System_3_PI-RADS_3_prostate_lesions_in_multiparametric_magnetic_resonance_imaging/20739085/1?file=36971380)

We do not directly provide the raw data from the corresponding papers to avoid potentially violating user licenses.
The extracted and preprocessed data is stored in `data/data.pkl`.
To automatically download all datasets, run the script below. Note that some of the download URLs may change in the future. If that happens, we recommend manually downloading the data by cross-referencing the original papers.

```
chmod+x download_datasets.sh
```

and

```
./download_datasets.sh
```

In order to make the data compatible with the `dataset_preprocess.py` file. The following changes need to be done to the files: 
1. `Korea1.pdf` needs to be converted to `Korea1.csv`
2. `China1.sav` needs to be converted to `China1.csv`
3. `Turkey.sav` needs to be converted to `Turkey.csv`
4. `China2.zip` needs to be unpacked. `Supplementary Table 1.xlsx` and `Supplementary Table 2.xlsx` need to be extracted and merged together into `China2.csv`.

For `sav` to `csv` conversion, we recommend using an open-source `pspp` tool. For `pdf` to `csv`, `tabula-py` can be used. For `xlsx` to `csv` -- `gnumeric`. Alternatively, online convertors can be used. 

## Key Scripts in this Repository

1. **`dataset_preprocess.py`**: Processes and prepares the data from peer-reviewed studies.

2. **`single_silo_learning.py`**: Trains the local models for the upper matrix shown in the paper (Figure 4).

3. **`multi_silo_learning.py`**: Trains the federated models for the lower matrix shown in the paper (Figure 4).

4. **`subsampling.py`**: Trains models using incomplete data, as described in Figure 5 of the paper.

---

## Requirements

The recommended way to set up the environment is by importing the Anaconda environment from the provided environment.yml file:

```bash
conda env create -f environment.yml
```

---
## Quickstart
To run the scripts, execute the following commands:

1. Preprocess the datasets (extract and transform features, handle missing values, construct silos):

```
python dataset_preprocess.py
```

**This works only if the data files are downloaded and converted properly. Otherwise just import `data.pkl`**

2. Train the local (single-silo) models using Monte Carlo simulations (example with 10 simulations):

```
python single_silo_learning.py --monte_carlo 10
```

3. Train the centralized, federated, and LSO (multi-silo) models using Monte Carlo simulations (example with 10 simulations):

```
python multi_silo_learning.py --monte_carlo 10
```

---

## Other files

By default, all results are saved in the `results/` folder. The plots presented in the paper are generated using the TikZ library, and the results are primarily stored in `.csv` files for easy import.

The source SVG file for the world map used in Figure 2 can be found as `World_Map.svg`

Additionally, the t-SNE plots shown in the appendix can be generated by running:

```
python tsne.py
```