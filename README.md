# Flight-Delay-Prediction
This project builds a machine learning pipeline to predict whether a U.S. domestic flight will be delayed by more than 15 minutes. Using the U.S. Department of Transportation’s On-Time Performance dataset, I explored flight-level data, engineered new features, addressed class imbalance, and trained models to uncover patterns in flight delays.

* Processed 100k+ flight records with Pandas for cleaning and feature engineering
* Engineered features: departure time bins, top airports, carrier encoding
* Handled class imbalance using SMOTE oversampling
* Trained and compared Random Forest and XGBoost classifiers
* Achieved ~81% accuracy, with improved recall for delayed flights
* Analyzed feature importances to identify top delay factors (carrier, departure time, origin/destination airport)

* Dataset downloaded from https://www.transtats.bts.gov/tables.asp?QO_VQ=EFD&QO_anzr=Nv4yv0r
