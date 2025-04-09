#!/bin/bash

# Execute pipeline steps
python src/data/collect_data.py
python src/models/train_models.py
python src/models/evaluate.py

echo "Pipeline completed successfully!"