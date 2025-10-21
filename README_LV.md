Issued command:

dvc stage add -n train --force -d src/train.py -d data/processed/train.csv -d data/processed/test.csv -p train.n_estimators,train.max_depth,train.random_state -o models/model.joblib -M metrics/metrics.json "python src/train.py data/processed/train.csv data/processed/test.csv models/model.joblib metrics/metrics.json"

Followed 
dvc repro ...
dvc push