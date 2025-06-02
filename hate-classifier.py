import pickle
import pandas as pd
from experiment import load_data
from transformer_classifier import cross_validate, calculate_scores

cache_dir = 'cache'
num_folds = 2
task = 'Metahate'

texts, targets = load_data('data', task=task)
pred_e5_multi, truth_e5_multi, logits_e5_multi, results_e5_multi = cross_validate(
    list(map(lambda x: "query: " + x, texts)), 
    targets, 
    model_path="intfloat/multilingual-e5-base", 
    num_folds=num_folds, 
    cache_dir=cache_dir, 
    batch_size=64,
    num_epochs=10,
    tuned_layers_count=-1,
    output_dir=f'{task.lower()}-e5-base'
)

def calculate_summary(actual, predictions, model_name):
    summary = calculate_scores(actual, predictions)
    summary['model'] = model_name
    return summary

with open(f'{task.lower()}-results.pkl', 'wb') as f:
    pickle.dump((
        (pred_e5_multi, truth_e5_multi, logits_e5_multi, results_e5_multi)
    ), f)
print(pd.DataFrame([
    calculate_summary(truth_e5_multi, pred_e5_multi, 'E5-multilingual')
]))
