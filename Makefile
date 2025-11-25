PYTHON ?= python
RAW_DATA ?= /mnt/data/hour.csv
PROCESSED ?= data/processed_hour.csv
RESULTS ?= results
MODELS ?= models
NOTEBOOKS := notebooks/EDA.ipynb notebooks/baseline_eval.ipynb notebooks/modeling.ipynb

.PHONY: prepare train train-rf train-gbm tune tune-rf tune-gbm rolling-eval ensemble ablation notebooks clean

prepare:
	@echo ">> Preparing data from $(RAW_DATA)"
	$(PYTHON) src/prepare.py --input $(RAW_DATA) --output $(PROCESSED) --resample-frequency H --lags 1 3 6 24 --rolling-windows 3 24

train: train-rf train-gbm

train-rf:
	@echo ">> Training RandomForest pipeline"
	$(PYTHON) src/train_models.py --input $(PROCESSED) --output_dir $(RESULTS) --model_dir $(MODELS) --model rf --test-size 0.2 --save-predictions

train-gbm:
	@echo ">> Training GradientBoosting pipeline"
	$(PYTHON) src/train_models.py --input $(PROCESSED) --output_dir $(RESULTS) --model_dir $(MODELS) --model gbm --test-size 0.2 --save-predictions

tune: tune-rf tune-gbm

tune-rf:
	@echo ">> RandomizedSearchCV for RandomForest (expanded budget)"
	$(PYTHON) src/tune_models.py --input $(PROCESSED) --model rf --output-dir $(RESULTS) --model-dir $(MODELS)

tune-gbm:
	@echo ">> RandomizedSearchCV for GradientBoosting (expanded budget)"
	$(PYTHON) src/tune_models.py --input $(PROCESSED) --model gbm --output-dir $(RESULTS) --model-dir $(MODELS)

rolling-eval:
	@echo ">> Sliding-window evaluation for RandomForest"
	$(PYTHON) src/rolling_eval.py --input $(PROCESSED) --model rf --output $(RESULTS)/rolling_eval_rf.json --strategy sliding
	@echo ">> Sliding-window evaluation for GradientBoosting"
	$(PYTHON) src/rolling_eval.py --input $(PROCESSED) --model gbm --output $(RESULTS)/rolling_eval_gbm.json --strategy sliding

ensemble:
	@echo ">> Optimizing ensemble weights from saved predictions"
	$(PYTHON) src/ensemble_opt.py \
		--predictions $(RESULTS)/rf_test_predictions.csv \
		--predictions $(RESULTS)/gbm_test_predictions.csv \
		--output $(RESULTS)/ensemble_metrics.json

ablation:
	@echo ">> Running feature ablation study"
	$(PYTHON) src/run_ablation.py --input $(PROCESSED) --output $(RESULTS)/feature_ablation.json --test-size 0.2

notebooks:
	@for nb in $(NOTEBOOKS); do \
		echo ">> Executing $$nb"; \
		jupyter nbconvert --to notebook --execute $$nb --output $$(basename $$nb .ipynb).executed.ipynb; \
	done

clean:
	@echo ">> Removing executed notebook artifacts"
	@del /Q notebooks\\*.executed.ipynb 2> NUL || true
	@rm -f notebooks/*.executed.ipynb || true

