from src.ETL import Extract
from src.ETL import Transform

from src.model_functions import build_model
from src.modeling import Modeling

from keras._tf_keras.keras.callbacks import EarlyStopping
import time

import tensorflow as tf

# Force Mac to use CPU
tf.config.set_visible_devices([], 'GPU')

model_ = "gru"

batch_size = 1000
max_epochs = 250

forecast_horizon = 4

for dataset_ in ["2yr", "10yr"]:

    params = {
    model_: {
        'model': None, # will eventually get a keras model instance after running modeling.cross_val\
        'dataset': dataset_, "forecast_horizon": forecast_horizon,
        'function': build_model, 'color': 'green',
        'l1_reg': None, 'H': None, 'label': model_.upper(),
        'cv_results': {"means_": None, 'stds_': None, 'params_': None}
        }
    }

    start_time = time.time()

    transform = Transform(dataset_=dataset_)

    transform.split_and_reformat(
        train_split=0.7,
        val_split=0.1,
        forecast_horizon=forecast_horizon
    )

    es = EarlyStopping(
        monitor='loss',
        mode='min',
        verbose=1,
        patience=100,
        min_delta=1e-7,
        restore_best_weights=True
        )
    
    modeling = Modeling(
        batch_size=batch_size,
        max_epochs=max_epochs,
        n_steps=15,
        forecast_horizon=forecast_horizon,
        stateful=False
    )

    modeling.cross_validate_bayesian(
        params=params, 
        es=es, 
        dataset=dataset_,
        forecast_horizon=forecast_horizon
    )

    modeling.train(
        es=es, 
        dataset_=dataset_,
        model_ = model_,
        bayes=True
    )

    end_time = time.time()
    print(f"For dataset {dataset_}: Time elapsed on batch size {batch_size} and max epoch of {max_epochs} is {end_time - start_time} seconds or {(end_time - start_time)/60} mins.")
