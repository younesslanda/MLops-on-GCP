"""Helper components."""
from googleapiclient import discovery
from googleapiclient import errors
    
#import joblib
import pickle
import json
import pandas as pd
import subprocess
import sys

from sklearn.metrics import accuracy_score, recall_score

from typing import NamedTuple


def retrieve_best_run(project_id: str, job_id: str) -> NamedTuple('Outputs', [('metric_value', float), ('alpha', float),
                            ('max_iter', int)]):
    """Retrieves the parameters of the best Hypertune run."""
    
    ml = discovery.build('ml', 'v1')

    job_name = 'projects/{}/jobs/{}'.format(project_id, job_id)
    request = ml.projects().jobs().get(name=job_name)

    try:
        response = request.execute()
    except errors.HttpError as err:
        print(err)
    except:
        print('Unexpected error')

    print(response)

    best_trial = response['trainingOutput']['trials'][0]

    metric_value = best_trial['finalMetric']['objectiveValue']
    alpha = float(best_trial['hyperparameters']['alpha'])
    max_iter = int(best_trial['hyperparameters']['max_iter'])

    return (metric_value, alpha, max_iter)


def evaluate_model(dataset_path: str, model_path: str, metric_name: str) -> NamedTuple('Outputs', [('metric_name', str), ('metric_value', float),
                            ('mlpipeline_metrics', 'Metrics')]):
    """Evaluates a trained sklearn model."""
    
    df_test = pd.read_csv(dataset_path)

    X_test = df_test.drop('Cover_Type', axis=1)
    y_test = df_test['Cover_Type']

    # Copy the model from GCS
    model_filename = 'model.pkl'
    gcs_model_filepath = '{}/{}'.format(model_path, model_filename)
    print(gcs_model_filepath)
    subprocess.check_call(['gsutil', 'cp', gcs_model_filepath, model_filename],
                        stderr=sys.stdout)

    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)

    y_hat = model.predict(X_test)

    if metric_name == 'accuracy':
        metric_value = accuracy_score(y_test, y_hat)
    elif metric_name == 'recall':
        metric_value = recall_score(y_test, y_hat)
    else:
        metric_name = 'N/A'
        metric_value = 0

    # Export the metric
    metrics = {
      'metrics': [{
          'name': metric_name,
          'numberValue': float(metric_value)
      }]
    }

    return (metric_name, metric_value, json.dumps(metrics))