```python
import pandas as pd
import torch
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.utils.class_weight import compute_class_weight
import os

```

    /home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
if torch.cuda.is_available():
    print("GPU 사용 가능! -", torch.cuda.get_device_name(0))
else:
    print("GPU를 사용할 수 없습니다. CPU로 학습합니다.")
```

    GPU 사용 가능! - NVIDIA GeForce RTX 3060



```python
# 데이터 불러오기
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
```


```python
# [.] => .
train_df['URL'] = train_df['URL'].apply(lambda x: x.replace('[.]', '.'))
test_df['URL'] = test_df['URL'].apply(lambda x: x.replace('[.]', '.'))
```


```python
predictor = TabularPredictor(label='label', problem_type='binary')
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20250223_015248"



```python
# 라벨별 가중치 계산
weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['label']), y=train_df['label'].values)
weights /= weights.sum()
weights = list(weights)
```


```python
# 사용 가능한 모든 모델
# models = ['GBM', 'CAT', 'XGB', 'RF', 'XT', 'KNN', 'NN_TORCH', 'FASTAI', 'LR', 'AG_TEXT_NN']
models = ['GBM', 'CAT']

predictors = {}

# for model in models:
predictor.fit(
    train_data=train_df,
    presets='best_quality',
    time_limit=None,
    # ag_args_fit={'num_gpus': 1},
    # hyperparameters={
    #     'XGB': {
    #         'tree_method = 'hist',
    #         device = 'cuda'
    #     }
    # },
    verbosity=3
)

results = predictor.fit_summary()

# preds = predictor.predict_proba(test_df)

# # 모델 저장
# save_path = f'./model/{model}'
# os.makedirs(save_path, exist_ok=True)
# predictor.save(save_path)

# # 예측 결과 저장
# submission_df = pd.read_csv('./data/sample_submission.csv')
# submission_df['probability'] = preds[1]
# submission_df.to_csv(f'./prediction/{model}.csv', index=False)

# predictors[model] = predictor
```

    Verbosity: 3 (Detailed Logging)
    =================== System Info ===================
    AutoGluon Version:  1.2
    Python Version:     3.10.13
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #17~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC Mon Jan 20 22:48:29 UTC 2
    CPU Count:          32
    GPU Count:          1
    Memory Avail:       65.67 GB / 78.38 GB (83.8%)
    Disk Space Avail:   1565.80 GB / 1831.67 GB (85.5%)
    ===================================================
    Presets specified: ['best_quality']
    ============ fit kwarg info ============
    User Specified kwargs:
    {'auto_stack': True, 'num_bag_sets': 1, 'verbosity': 3}
    Full kwargs:
    {'_feature_generator_kwargs': None,
     '_save_bag_folds': None,
     'ag_args': None,
     'ag_args_ensemble': None,
     'ag_args_fit': None,
     'auto_stack': True,
     'calibrate': 'auto',
     'delay_bag_sets': False,
     'ds_args': {'clean_up_fits': True,
                 'detection_time_frac': 0.25,
                 'enable_callbacks': False,
                 'enable_ray_logging': True,
                 'holdout_data': None,
                 'holdout_frac': 0.1111111111111111,
                 'memory_safe_fits': True,
                 'n_folds': 2,
                 'n_repeats': 1,
                 'validation_procedure': 'holdout'},
     'excluded_model_types': None,
     'feature_generator': 'auto',
     'feature_prune_kwargs': None,
     'holdout_frac': None,
     'hyperparameter_tune_kwargs': None,
     'included_model_types': None,
     'keep_only_best': False,
     'learning_curves': False,
     'name_suffix': None,
     'num_bag_folds': None,
     'num_bag_sets': 1,
     'num_stack_levels': None,
     'pseudo_data': None,
     'raise_on_no_models_fitted': True,
     'refit_full': False,
     'save_bag_folds': None,
     'save_space': False,
     'set_best_to_refit_full': False,
     'test_data': None,
     'unlabeled_data': None,
     'use_bag_holdout': False,
     'verbosity': 3}
    ========================================
    Setting dynamic_stacking from 'auto' to True. Reason: Enable dynamic_stacking when use_bag_holdout is disabled. (use_bag_holdout=False)
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=1
    DyStack is enabled (dynamic_stacking=True). AutoGluon will try to determine whether the input data is affected by stacked overfitting and enable or disable stacking as a consequence.
    	This is used to identify the optimal `num_stack_levels` value. Copies of AutoGluon will be fit on subsets of the data. Then holdout validation data is used to detect stacked overfitting.
    	Running DyStack for up to 900s of the 3600s of remaining time (25%).
    2025-02-23 10:52:49,295	INFO util.py:154 -- Missing packages: ['ipywidgets']. Run `pip install -U ipywidgets`, then restart the notebook server for rich notebook output.
    	Running DyStack sub-fit in a ray process to avoid memory leakage. Enabling ray logging (enable_ray_logging=True). Specify `ds_args={'enable_ray_logging': False}` if you experience logging issues.
    2025-02-23 10:52:50,424	INFO worker.py:1810 -- Started a local Ray instance. View the dashboard at [1m[32m127.0.0.1:8265 [39m[22m
    		Context path: "/home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho"
    [36m(_dystack pid=167306)[0m Running DyStack sub-fit ...
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/learner.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/predictor.pkl
    [36m(_dystack pid=167306)[0m Beginning AutoGluon training ... Time limit = 898s
    [36m(_dystack pid=167306)[0m AutoGluon will save models to "/home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho"
    [36m(_dystack pid=167306)[0m Train Data Rows:    6217827
    [36m(_dystack pid=167306)[0m Train Data Columns: 2
    [36m(_dystack pid=167306)[0m Label Column:       label
    [36m(_dystack pid=167306)[0m Problem Type:       binary
    [36m(_dystack pid=167306)[0m Preprocessing data ...
    [36m(_dystack pid=167306)[0m Selected class <--> label mapping:  class 1 = 1, class 0 = 0
    [36m(_dystack pid=167306)[0m Using Feature Generators to preprocess the data ...
    [36m(_dystack pid=167306)[0m Fitting AutoMLPipelineFeatureGenerator...
    [36m(_dystack pid=167306)[0m 	Available Memory:                    65618.96 MB
    [36m(_dystack pid=167306)[0m 	Train Data (Original)  Memory Usage: 897.34 MB (1.4% of available memory)
    [36m(_dystack pid=167306)[0m 	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    [36m(_dystack pid=167306)[0m 	Stage 1 Generators:
    [36m(_dystack pid=167306)[0m 		Fitting AsTypeFeatureGenerator...
    [36m(_dystack pid=167306)[0m 			Original Features (exact raw dtype, raw dtype):
    [36m(_dystack pid=167306)[0m 				('object', 'object') : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('object', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('object', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			1.4s = Fit runtime
    [36m(_dystack pid=167306)[0m 			2 features in original data used to generate 2 features in processed data.
    [36m(_dystack pid=167306)[0m 	Stage 2 Generators:
    [36m(_dystack pid=167306)[0m 		Fitting FillNaFeatureGenerator...
    [36m(_dystack pid=167306)[0m 			Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('object', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('object', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			1.6s = Fit runtime
    [36m(_dystack pid=167306)[0m 			2 features in original data used to generate 2 features in processed data.
    [36m(_dystack pid=167306)[0m 	Stage 3 Generators:
    [36m(_dystack pid=167306)[0m 		Skipping IdentityFeatureGenerator: No input feature with required dtypes.
    [36m(_dystack pid=167306)[0m 		Fitting CategoryFeatureGenerator...
    [36m(_dystack pid=167306)[0m 			Fitting CategoryMemoryMinimizeFeatureGenerator...
    [36m(_dystack pid=167306)[0m 				Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 					('category', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 				Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 					('category', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 				0.4s = Fit runtime
    [36m(_dystack pid=167306)[0m 				2 features in original data used to generate 2 features in processed data.
    [36m(_dystack pid=167306)[0m 			Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('object', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('category', []) : 2 | ['ID', 'URL']
    [36m(_dystack pid=167306)[0m 			26.6s = Fit runtime
    [36m(_dystack pid=167306)[0m 			2 features in original data used to generate 2 features in processed data.
    [36m(_dystack pid=167306)[0m 		Skipping DatetimeFeatureGenerator: No input feature with required dtypes.
    [36m(_dystack pid=167306)[0m 		Skipping TextSpecialFeatureGenerator: No input feature with required dtypes.
    [36m(_dystack pid=167306)[0m 		Skipping TextNgramFeatureGenerator: No input feature with required dtypes.
    [36m(_dystack pid=167306)[0m 		Skipping IdentityFeatureGenerator: No input feature with required dtypes.
    [36m(_dystack pid=167306)[0m 		Skipping IsNanFeatureGenerator: No input feature with required dtypes.
    [36m(_dystack pid=167306)[0m 	Stage 4 Generators:
    [36m(_dystack pid=167306)[0m 		Fitting DropUniqueFeatureGenerator...
    [36m(_dystack pid=167306)[0m 			Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('category', []) : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 			Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('category', []) : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 			0.4s = Fit runtime
    [36m(_dystack pid=167306)[0m 			1 features in original data used to generate 1 features in processed data.
    [36m(_dystack pid=167306)[0m 	Stage 5 Generators:
    [36m(_dystack pid=167306)[0m 		Fitting DropDuplicatesFeatureGenerator...
    [36m(_dystack pid=167306)[0m 			Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('category', []) : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 			Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 				('category', []) : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 			0.4s = Fit runtime
    [36m(_dystack pid=167306)[0m 			1 features in original data used to generate 1 features in processed data.
    [36m(_dystack pid=167306)[0m 	Unused Original Features (Count: 1): ['ID']
    [36m(_dystack pid=167306)[0m 		These features were not used to generate any of the output features. Add a feature generator compatible with these features to utilize them.
    [36m(_dystack pid=167306)[0m 		Features can also be unused if they carry very little information, such as being categorical but having almost entirely unique values or being duplicates of other features.
    [36m(_dystack pid=167306)[0m 		These features do not need to be present at inference time.
    [36m(_dystack pid=167306)[0m 		('object', []) : 1 | ['ID']
    [36m(_dystack pid=167306)[0m 	Types of features in original data (exact raw dtype, raw dtype):
    [36m(_dystack pid=167306)[0m 		('object', 'object') : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 	Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 		('object', []) : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 	Types of features in processed data (exact raw dtype, raw dtype):
    [36m(_dystack pid=167306)[0m 		('category', 'category') : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 	Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=167306)[0m 		('category', []) : 1 | ['URL']
    [36m(_dystack pid=167306)[0m 	37.7s = Fit runtime
    [36m(_dystack pid=167306)[0m 	1 features in original data used to generate 1 features in processed data.
    [36m(_dystack pid=167306)[0m 	Train Data (Processed) Memory Usage: 11.86 MB (0.0% of available memory)
    [36m(_dystack pid=167306)[0m Data preprocessing and feature engineering runtime = 38.97s ...
    [36m(_dystack pid=167306)[0m AutoGluon will gauge predictive performance using evaluation metric: 'accuracy'
    [36m(_dystack pid=167306)[0m 	To change this, specify the eval_metric parameter of Predictor()
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/learner.pkl
    [36m(_dystack pid=167306)[0m User-specified model hyperparameters to be fit:
    [36m(_dystack pid=167306)[0m {
    [36m(_dystack pid=167306)[0m 	'NN_TORCH': [{}, {'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2}}, {'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7}}, {'activation': 'elu', 'dropout_prob': 0.24622382571353768, 'hidden_size': 159, 'learning_rate': 0.008507536855608535, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.8201539594953562e-06, 'ag_args': {'name_suffix': '_r30', 'priority': -17}}, {'activation': 'relu', 'dropout_prob': 0.09976801642258049, 'hidden_size': 135, 'learning_rate': 0.001631450730978947, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 3.867683394425807e-05, 'ag_args': {'name_suffix': '_r86', 'priority': -19}}, {'activation': 'relu', 'dropout_prob': 0.3905837860053583, 'hidden_size': 106, 'learning_rate': 0.0018297905295930797, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 9.178069874232892e-08, 'ag_args': {'name_suffix': '_r14', 'priority': -26}}, {'activation': 'relu', 'dropout_prob': 0.05488816803887784, 'hidden_size': 32, 'learning_rate': 0.0075612897834015985, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.652353009917866e-08, 'ag_args': {'name_suffix': '_r41', 'priority': -35}}, {'activation': 'elu', 'dropout_prob': 0.01030258381183309, 'hidden_size': 111, 'learning_rate': 0.01845979186513771, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 0.00020238017476912164, 'ag_args': {'name_suffix': '_r158', 'priority': -38}}, {'activation': 'elu', 'dropout_prob': 0.18109219857068798, 'hidden_size': 250, 'learning_rate': 0.00634181748507711, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 5.3861175580695396e-08, 'ag_args': {'name_suffix': '_r197', 'priority': -41}}, {'activation': 'elu', 'dropout_prob': 0.1703783780377607, 'hidden_size': 212, 'learning_rate': 0.0004107199833213839, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.105439140660822e-07, 'ag_args': {'name_suffix': '_r143', 'priority': -49}}, {'activation': 'elu', 'dropout_prob': 0.013288954106470907, 'hidden_size': 81, 'learning_rate': 0.005340914647396154, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 8.762168370775353e-05, 'ag_args': {'name_suffix': '_r31', 'priority': -52}}, {'activation': 'relu', 'dropout_prob': 0.36669080773207274, 'hidden_size': 95, 'learning_rate': 0.015280159186761077, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.3082489374636015e-08, 'ag_args': {'name_suffix': '_r87', 'priority': -59}}, {'activation': 'relu', 'dropout_prob': 0.3027114570947557, 'hidden_size': 196, 'learning_rate': 0.006482759295309238, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 1.2806509958776e-12, 'ag_args': {'name_suffix': '_r71', 'priority': -60}}, {'activation': 'relu', 'dropout_prob': 0.12166942295569863, 'hidden_size': 151, 'learning_rate': 0.0018866871631794007, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 9.190843763153802e-05, 'ag_args': {'name_suffix': '_r185', 'priority': -65}}, {'activation': 'relu', 'dropout_prob': 0.006531401073483156, 'hidden_size': 192, 'learning_rate': 0.012418052210914356, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 3.0406866089493607e-05, 'ag_args': {'name_suffix': '_r76', 'priority': -77}}, {'activation': 'relu', 'dropout_prob': 0.33926015213879396, 'hidden_size': 247, 'learning_rate': 0.0029983839090226075, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 0.00038926240517691234, 'ag_args': {'name_suffix': '_r121', 'priority': -79}}, {'activation': 'elu', 'dropout_prob': 0.06134755114373829, 'hidden_size': 144, 'learning_rate': 0.005834535148903801, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 2.0826540090463355e-09, 'ag_args': {'name_suffix': '_r135', 'priority': -84}}, {'activation': 'elu', 'dropout_prob': 0.3457125770744979, 'hidden_size': 37, 'learning_rate': 0.006435774191713849, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 2.4012185204155345e-08, 'ag_args': {'name_suffix': '_r36', 'priority': -87}}, {'activation': 'relu', 'dropout_prob': 0.2211285919550286, 'hidden_size': 196, 'learning_rate': 0.011307978270179143, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 1.8441764217351068e-06, 'ag_args': {'name_suffix': '_r19', 'priority': -92}}, {'activation': 'relu', 'dropout_prob': 0.23713784729000734, 'hidden_size': 200, 'learning_rate': 0.00311256170909018, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 4.573016756474468e-08, 'ag_args': {'name_suffix': '_r1', 'priority': -96}}, {'activation': 'relu', 'dropout_prob': 0.33567564890346097, 'hidden_size': 245, 'learning_rate': 0.006746560197328548, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.6470047305392933e-10, 'ag_args': {'name_suffix': '_r89', 'priority': -97}}],
    [36m(_dystack pid=167306)[0m 	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}, {'extra_trees': False, 'feature_fraction': 0.7023601671276614, 'learning_rate': 0.012144796373999013, 'min_data_in_leaf': 14, 'num_leaves': 53, 'ag_args': {'name_suffix': '_r131', 'priority': -3}}, {'extra_trees': True, 'feature_fraction': 0.5636931414546802, 'learning_rate': 0.01518660230385841, 'min_data_in_leaf': 48, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r96', 'priority': -6}}, {'extra_trees': True, 'feature_fraction': 0.8282601210460099, 'learning_rate': 0.033929021353492905, 'min_data_in_leaf': 6, 'num_leaves': 127, 'ag_args': {'name_suffix': '_r188', 'priority': -14}}, {'extra_trees': False, 'feature_fraction': 0.6245777099925497, 'learning_rate': 0.04711573688184715, 'min_data_in_leaf': 56, 'num_leaves': 89, 'ag_args': {'name_suffix': '_r130', 'priority': -18}}, {'extra_trees': False, 'feature_fraction': 0.5898927512279213, 'learning_rate': 0.010464516487486093, 'min_data_in_leaf': 11, 'num_leaves': 252, 'ag_args': {'name_suffix': '_r161', 'priority': -27}}, {'extra_trees': True, 'feature_fraction': 0.5143401489640409, 'learning_rate': 0.00529479887023554, 'min_data_in_leaf': 6, 'num_leaves': 133, 'ag_args': {'name_suffix': '_r196', 'priority': -31}}, {'extra_trees': False, 'feature_fraction': 0.7421180622507277, 'learning_rate': 0.018603888565740096, 'min_data_in_leaf': 6, 'num_leaves': 22, 'ag_args': {'name_suffix': '_r15', 'priority': -37}}, {'extra_trees': False, 'feature_fraction': 0.9408897917880529, 'learning_rate': 0.01343464462043561, 'min_data_in_leaf': 21, 'num_leaves': 178, 'ag_args': {'name_suffix': '_r143', 'priority': -44}}, {'extra_trees': True, 'feature_fraction': 0.4341088458599442, 'learning_rate': 0.04034449862560467, 'min_data_in_leaf': 33, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r94', 'priority': -48}}, {'extra_trees': True, 'feature_fraction': 0.9773131270704629, 'learning_rate': 0.010534290864227067, 'min_data_in_leaf': 21, 'num_leaves': 111, 'ag_args': {'name_suffix': '_r30', 'priority': -56}}, {'extra_trees': False, 'feature_fraction': 0.8254432681390782, 'learning_rate': 0.031251656439648626, 'min_data_in_leaf': 50, 'num_leaves': 210, 'ag_args': {'name_suffix': '_r135', 'priority': -69}}, {'extra_trees': False, 'feature_fraction': 0.5730390983988963, 'learning_rate': 0.010305352949119608, 'min_data_in_leaf': 10, 'num_leaves': 215, 'ag_args': {'name_suffix': '_r121', 'priority': -74}}, {'extra_trees': True, 'feature_fraction': 0.4601361323873807, 'learning_rate': 0.07856777698860955, 'min_data_in_leaf': 12, 'num_leaves': 198, 'ag_args': {'name_suffix': '_r42', 'priority': -95}}],
    [36m(_dystack pid=167306)[0m 	'CAT': [{}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5}}, {'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.559174625782161, 'learning_rate': 0.04939557741379516, 'max_ctr_complexity': 3, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r137', 'priority': -10}}, {'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3274013177541373, 'learning_rate': 0.017301189655111057, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r13', 'priority': -12}}, {'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7018061518087038, 'learning_rate': 0.07092851311746352, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r50', 'priority': -20}}, {'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.0457098345001241, 'learning_rate': 0.050294288910022224, 'max_ctr_complexity': 5, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r69', 'priority': -24}}, {'depth': 6, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.3584121369544215, 'learning_rate': 0.03743901034980473, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r70', 'priority': -29}}, {'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.522712492188319, 'learning_rate': 0.08481607830570326, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r167', 'priority': -33}}, {'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6376578537958237, 'learning_rate': 0.032899230324940465, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r86', 'priority': -39}}, {'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.353268454214423, 'learning_rate': 0.06028218319511302, 'max_ctr_complexity': 1, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r49', 'priority': -42}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.640921865280573, 'learning_rate': 0.036232951900213306, 'max_ctr_complexity': 3, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r128', 'priority': -50}}, {'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.894432181094842, 'learning_rate': 0.055078095725390575, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r5', 'priority': -58}}, {'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6761016245166451, 'learning_rate': 0.06566144806528762, 'max_ctr_complexity': 2, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r143', 'priority': -61}}, {'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3217885487525205, 'learning_rate': 0.05291587380674719, 'max_ctr_complexity': 5, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r60', 'priority': -67}}, {'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.5734131496361856, 'learning_rate': 0.08472519974533015, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r6', 'priority': -72}}, {'depth': 7, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 4.43335055453705, 'learning_rate': 0.055406199833457785, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r180', 'priority': -76}}, {'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.835797074498082, 'learning_rate': 0.03534026385152556, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r12', 'priority': -83}}, {'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.7454481983750014, 'learning_rate': 0.09328642499990342, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r163', 'priority': -89}}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.637071465711953, 'learning_rate': 0.04387418552563314, 'max_ctr_complexity': 4, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r198', 'priority': -90}}],
    [36m(_dystack pid=167306)[0m 	'XGB': [{}, {'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'name_suffix': '_r33', 'priority': -8}}, {'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'name_suffix': '_r89', 'priority': -16}}, {'colsample_bytree': 0.9090166528779192, 'enable_categorical': True, 'learning_rate': 0.09290221350439203, 'max_depth': 7, 'min_child_weight': 0.8041986915994078, 'ag_args': {'name_suffix': '_r194', 'priority': -22}}, {'colsample_bytree': 0.516652313273348, 'enable_categorical': True, 'learning_rate': 0.007158072983547058, 'max_depth': 9, 'min_child_weight': 0.8567068904025429, 'ag_args': {'name_suffix': '_r98', 'priority': -36}}, {'colsample_bytree': 0.7452294043087835, 'enable_categorical': False, 'learning_rate': 0.038404229910104046, 'max_depth': 7, 'min_child_weight': 0.5564183327139662, 'ag_args': {'name_suffix': '_r49', 'priority': -57}}, {'colsample_bytree': 0.7506621909633511, 'enable_categorical': False, 'learning_rate': 0.009974712407899168, 'max_depth': 4, 'min_child_weight': 0.9238550485581797, 'ag_args': {'name_suffix': '_r31', 'priority': -64}}, {'colsample_bytree': 0.6326947454697227, 'enable_categorical': False, 'learning_rate': 0.07792091886639502, 'max_depth': 6, 'min_child_weight': 1.0759464955561793, 'ag_args': {'name_suffix': '_r22', 'priority': -70}}, {'colsample_bytree': 0.975937238416368, 'enable_categorical': False, 'learning_rate': 0.06634196266155237, 'max_depth': 5, 'min_child_weight': 1.4088437184127383, 'ag_args': {'name_suffix': '_r95', 'priority': -93}}, {'colsample_bytree': 0.546186944730449, 'enable_categorical': False, 'learning_rate': 0.029357102578825213, 'max_depth': 10, 'min_child_weight': 1.1532008198571337, 'ag_args': {'name_suffix': '_r34', 'priority': -94}}],
    [36m(_dystack pid=167306)[0m 	'FASTAI': [{}, {'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4}}, {'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11}}, {'bs': 128, 'emb_drop': 0.44339037504795686, 'epochs': 31, 'layers': [400, 200, 100], 'lr': 0.008615195908919904, 'ps': 0.19220253419114286, 'ag_args': {'name_suffix': '_r145', 'priority': -15}}, {'bs': 128, 'emb_drop': 0.026897798530914306, 'epochs': 31, 'layers': [800, 400], 'lr': 0.08045277634470181, 'ps': 0.4569532219038436, 'ag_args': {'name_suffix': '_r11', 'priority': -21}}, {'bs': 256, 'emb_drop': 0.1508701680951814, 'epochs': 46, 'layers': [400, 200], 'lr': 0.08794353125787312, 'ps': 0.19110623090573325, 'ag_args': {'name_suffix': '_r103', 'priority': -25}}, {'bs': 1024, 'emb_drop': 0.6239200452002372, 'epochs': 39, 'layers': [200, 100, 50], 'lr': 0.07170321592506483, 'ps': 0.670815151683455, 'ag_args': {'name_suffix': '_r143', 'priority': -28}}, {'bs': 2048, 'emb_drop': 0.5055288166864152, 'epochs': 44, 'layers': [400], 'lr': 0.0047762208542912405, 'ps': 0.06572612802222005, 'ag_args': {'name_suffix': '_r156', 'priority': -30}}, {'bs': 128, 'emb_drop': 0.6656668277387758, 'epochs': 32, 'layers': [400, 200, 100], 'lr': 0.019326244622675428, 'ps': 0.04084945128641206, 'ag_args': {'name_suffix': '_r95', 'priority': -34}}, {'bs': 512, 'emb_drop': 0.1567472816422661, 'epochs': 41, 'layers': [400, 200, 100], 'lr': 0.06831450078222204, 'ps': 0.4930900813464729, 'ag_args': {'name_suffix': '_r37', 'priority': -40}}, {'bs': 2048, 'emb_drop': 0.006251885504130949, 'epochs': 47, 'layers': [800, 400], 'lr': 0.01329622020483052, 'ps': 0.2677080696008348, 'ag_args': {'name_suffix': '_r134', 'priority': -46}}, {'bs': 2048, 'emb_drop': 0.6343202884164582, 'epochs': 21, 'layers': [400, 200], 'lr': 0.08479209380262258, 'ps': 0.48362560779595565, 'ag_args': {'name_suffix': '_r111', 'priority': -51}}, {'bs': 1024, 'emb_drop': 0.22771721361129746, 'epochs': 38, 'layers': [400], 'lr': 0.0005383511954451698, 'ps': 0.3734259772256502, 'ag_args': {'name_suffix': '_r65', 'priority': -54}}, {'bs': 1024, 'emb_drop': 0.4329361816589235, 'epochs': 50, 'layers': [400], 'lr': 0.09501311551121323, 'ps': 0.2863378667611431, 'ag_args': {'name_suffix': '_r88', 'priority': -55}}, {'bs': 128, 'emb_drop': 0.3171659718142149, 'epochs': 20, 'layers': [400, 200, 100], 'lr': 0.03087210106068273, 'ps': 0.5909644730871169, 'ag_args': {'name_suffix': '_r160', 'priority': -66}}, {'bs': 128, 'emb_drop': 0.3209601865656554, 'epochs': 21, 'layers': [200, 100, 50], 'lr': 0.019935403046870463, 'ps': 0.19846319260751663, 'ag_args': {'name_suffix': '_r69', 'priority': -71}}, {'bs': 128, 'emb_drop': 0.08669109226243704, 'epochs': 45, 'layers': [800, 400], 'lr': 0.0041554361714983635, 'ps': 0.2669780074016213, 'ag_args': {'name_suffix': '_r138', 'priority': -73}}, {'bs': 512, 'emb_drop': 0.05604276533830355, 'epochs': 32, 'layers': [400], 'lr': 0.027320709383189166, 'ps': 0.022591301744255762, 'ag_args': {'name_suffix': '_r172', 'priority': -75}}, {'bs': 1024, 'emb_drop': 0.31956392388385874, 'epochs': 25, 'layers': [200, 100], 'lr': 0.08552736732040143, 'ps': 0.0934076022219228, 'ag_args': {'name_suffix': '_r127', 'priority': -80}}, {'bs': 256, 'emb_drop': 0.5117456464220826, 'epochs': 21, 'layers': [400, 200, 100], 'lr': 0.007212882302137526, 'ps': 0.2747013981281539, 'ag_args': {'name_suffix': '_r194', 'priority': -82}}, {'bs': 256, 'emb_drop': 0.06099050979107849, 'epochs': 39, 'layers': [200], 'lr': 0.04119582873110387, 'ps': 0.5447097256648953, 'ag_args': {'name_suffix': '_r4', 'priority': -85}}, {'bs': 2048, 'emb_drop': 0.6960805527533755, 'epochs': 38, 'layers': [800, 400], 'lr': 0.0007278526871749883, 'ps': 0.20495582200836318, 'ag_args': {'name_suffix': '_r100', 'priority': -88}}, {'bs': 1024, 'emb_drop': 0.5074958658302495, 'epochs': 42, 'layers': [200, 100, 50], 'lr': 0.026342427824862867, 'ps': 0.34814978753283593, 'ag_args': {'name_suffix': '_r187', 'priority': -91}}],
    [36m(_dystack pid=167306)[0m 	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}, {'max_features': 0.75, 'max_leaf_nodes': 37308, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r195', 'priority': -13}}, {'max_features': 0.75, 'max_leaf_nodes': 28310, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r39', 'priority': -32}}, {'max_features': 1.0, 'max_leaf_nodes': 38572, 'min_samples_leaf': 5, 'ag_args': {'name_suffix': '_r127', 'priority': -45}}, {'max_features': 0.75, 'max_leaf_nodes': 18242, 'min_samples_leaf': 40, 'ag_args': {'name_suffix': '_r34', 'priority': -47}}, {'max_features': 'log2', 'max_leaf_nodes': 42644, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r166', 'priority': -63}}, {'max_features': 0.75, 'max_leaf_nodes': 36230, 'min_samples_leaf': 3, 'ag_args': {'name_suffix': '_r15', 'priority': -68}}, {'max_features': 1.0, 'max_leaf_nodes': 48136, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r16', 'priority': -81}}],
    [36m(_dystack pid=167306)[0m 	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}, {'max_features': 0.75, 'max_leaf_nodes': 18392, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r42', 'priority': -9}}, {'max_features': 1.0, 'max_leaf_nodes': 12845, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r172', 'priority': -23}}, {'max_features': 'sqrt', 'max_leaf_nodes': 28532, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r49', 'priority': -43}}, {'max_features': 1.0, 'max_leaf_nodes': 19935, 'min_samples_leaf': 20, 'ag_args': {'name_suffix': '_r4', 'priority': -53}}, {'max_features': 0.75, 'max_leaf_nodes': 29813, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r178', 'priority': -62}}, {'max_features': 1.0, 'max_leaf_nodes': 40459, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r197', 'priority': -78}}, {'max_features': 'sqrt', 'max_leaf_nodes': 29702, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r126', 'priority': -86}}],
    [36m(_dystack pid=167306)[0m 	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
    [36m(_dystack pid=167306)[0m }
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/utils/data/X.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/utils/data/y.pkl
    [36m(_dystack pid=167306)[0m AutoGluon will fit 2 stack levels (L1 to L2) ...
    [36m(_dystack pid=167306)[0m Model configs that will be trained (in order):
    [36m(_dystack pid=167306)[0m 	KNeighborsUnif_BAG_L1: 	{'weights': 'uniform', 'ag_args': {'valid_stacker': False, 'problem_types': ['binary', 'multiclass', 'regression'], 'name_suffix': 'Unif', 'model_type': <class 'autogluon.tabular.models.knn.knn_model.KNNModel'>, 'priority': 100}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	KNeighborsDist_BAG_L1: 	{'weights': 'distance', 'ag_args': {'valid_stacker': False, 'problem_types': ['binary', 'multiclass', 'regression'], 'name_suffix': 'Dist', 'model_type': <class 'autogluon.tabular.models.knn.knn_model.KNNModel'>, 'priority': 100}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBMXT_BAG_L1: 	{'extra_trees': True, 'ag_args': {'name_suffix': 'XT', 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    [36m(_dystack pid=167306)[0m 	LightGBM_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    [36m(_dystack pid=167306)[0m 	RandomForestGini_BAG_L1: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	RandomForestEntr_BAG_L1: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>, 'priority': 70}}
    [36m(_dystack pid=167306)[0m 	ExtraTreesGini_BAG_L1: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	ExtraTreesEntr_BAG_L1: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>, 'priority': 50}}
    [36m(_dystack pid=167306)[0m 	XGBoost_BAG_L1: 	{'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>, 'priority': 40}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>, 'priority': 25}}
    [36m(_dystack pid=167306)[0m 	LightGBMLarge_BAG_L1: 	{'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r177_BAG_L1: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r79_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r131_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.7023601671276614, 'learning_rate': 0.012144796373999013, 'min_data_in_leaf': 14, 'num_leaves': 53, 'ag_args': {'name_suffix': '_r131', 'priority': -3, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r191_BAG_L1: 	{'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r9_BAG_L1: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r96_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.5636931414546802, 'learning_rate': 0.01518660230385841, 'min_data_in_leaf': 48, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r96', 'priority': -6, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r22_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r33_BAG_L1: 	{'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r33', 'priority': -8, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r42_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 18392, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r42', 'priority': -9, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r137_BAG_L1: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.559174625782161, 'learning_rate': 0.04939557741379516, 'max_ctr_complexity': 3, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r137', 'priority': -10, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r102_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r13_BAG_L1: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3274013177541373, 'learning_rate': 0.017301189655111057, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r13', 'priority': -12, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r195_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 37308, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r195', 'priority': -13, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r188_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.8282601210460099, 'learning_rate': 0.033929021353492905, 'min_data_in_leaf': 6, 'num_leaves': 127, 'ag_args': {'name_suffix': '_r188', 'priority': -14, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r145_BAG_L1: 	{'bs': 128, 'emb_drop': 0.44339037504795686, 'epochs': 31, 'layers': [400, 200, 100], 'lr': 0.008615195908919904, 'ps': 0.19220253419114286, 'ag_args': {'name_suffix': '_r145', 'priority': -15, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r89_BAG_L1: 	{'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r89', 'priority': -16, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r30_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.24622382571353768, 'hidden_size': 159, 'learning_rate': 0.008507536855608535, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.8201539594953562e-06, 'ag_args': {'name_suffix': '_r30', 'priority': -17, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r130_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.6245777099925497, 'learning_rate': 0.04711573688184715, 'min_data_in_leaf': 56, 'num_leaves': 89, 'ag_args': {'name_suffix': '_r130', 'priority': -18, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r86_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.09976801642258049, 'hidden_size': 135, 'learning_rate': 0.001631450730978947, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 3.867683394425807e-05, 'ag_args': {'name_suffix': '_r86', 'priority': -19, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r50_BAG_L1: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7018061518087038, 'learning_rate': 0.07092851311746352, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r50', 'priority': -20, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r11_BAG_L1: 	{'bs': 128, 'emb_drop': 0.026897798530914306, 'epochs': 31, 'layers': [800, 400], 'lr': 0.08045277634470181, 'ps': 0.4569532219038436, 'ag_args': {'name_suffix': '_r11', 'priority': -21, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r194_BAG_L1: 	{'colsample_bytree': 0.9090166528779192, 'enable_categorical': True, 'learning_rate': 0.09290221350439203, 'max_depth': 7, 'min_child_weight': 0.8041986915994078, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r194', 'priority': -22, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r172_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 12845, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r172', 'priority': -23, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r69_BAG_L1: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.0457098345001241, 'learning_rate': 0.050294288910022224, 'max_ctr_complexity': 5, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r69', 'priority': -24, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r103_BAG_L1: 	{'bs': 256, 'emb_drop': 0.1508701680951814, 'epochs': 46, 'layers': [400, 200], 'lr': 0.08794353125787312, 'ps': 0.19110623090573325, 'ag_args': {'name_suffix': '_r103', 'priority': -25, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r14_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.3905837860053583, 'hidden_size': 106, 'learning_rate': 0.0018297905295930797, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 9.178069874232892e-08, 'ag_args': {'name_suffix': '_r14', 'priority': -26, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r161_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.5898927512279213, 'learning_rate': 0.010464516487486093, 'min_data_in_leaf': 11, 'num_leaves': 252, 'ag_args': {'name_suffix': '_r161', 'priority': -27, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r143_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.6239200452002372, 'epochs': 39, 'layers': [200, 100, 50], 'lr': 0.07170321592506483, 'ps': 0.670815151683455, 'ag_args': {'name_suffix': '_r143', 'priority': -28, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r70_BAG_L1: 	{'depth': 6, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.3584121369544215, 'learning_rate': 0.03743901034980473, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r70', 'priority': -29, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r156_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.5055288166864152, 'epochs': 44, 'layers': [400], 'lr': 0.0047762208542912405, 'ps': 0.06572612802222005, 'ag_args': {'name_suffix': '_r156', 'priority': -30, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r196_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.5143401489640409, 'learning_rate': 0.00529479887023554, 'min_data_in_leaf': 6, 'num_leaves': 133, 'ag_args': {'name_suffix': '_r196', 'priority': -31, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r39_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 28310, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r39', 'priority': -32, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r167_BAG_L1: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.522712492188319, 'learning_rate': 0.08481607830570326, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r167', 'priority': -33, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r95_BAG_L1: 	{'bs': 128, 'emb_drop': 0.6656668277387758, 'epochs': 32, 'layers': [400, 200, 100], 'lr': 0.019326244622675428, 'ps': 0.04084945128641206, 'ag_args': {'name_suffix': '_r95', 'priority': -34, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r41_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.05488816803887784, 'hidden_size': 32, 'learning_rate': 0.0075612897834015985, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.652353009917866e-08, 'ag_args': {'name_suffix': '_r41', 'priority': -35, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r98_BAG_L1: 	{'colsample_bytree': 0.516652313273348, 'enable_categorical': True, 'learning_rate': 0.007158072983547058, 'max_depth': 9, 'min_child_weight': 0.8567068904025429, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r98', 'priority': -36, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r15_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.7421180622507277, 'learning_rate': 0.018603888565740096, 'min_data_in_leaf': 6, 'num_leaves': 22, 'ag_args': {'name_suffix': '_r15', 'priority': -37, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r158_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.01030258381183309, 'hidden_size': 111, 'learning_rate': 0.01845979186513771, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 0.00020238017476912164, 'ag_args': {'name_suffix': '_r158', 'priority': -38, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r86_BAG_L1: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6376578537958237, 'learning_rate': 0.032899230324940465, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r86', 'priority': -39, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r37_BAG_L1: 	{'bs': 512, 'emb_drop': 0.1567472816422661, 'epochs': 41, 'layers': [400, 200, 100], 'lr': 0.06831450078222204, 'ps': 0.4930900813464729, 'ag_args': {'name_suffix': '_r37', 'priority': -40, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r197_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.18109219857068798, 'hidden_size': 250, 'learning_rate': 0.00634181748507711, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 5.3861175580695396e-08, 'ag_args': {'name_suffix': '_r197', 'priority': -41, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r49_BAG_L1: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.353268454214423, 'learning_rate': 0.06028218319511302, 'max_ctr_complexity': 1, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r49', 'priority': -42, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r49_BAG_L1: 	{'max_features': 'sqrt', 'max_leaf_nodes': 28532, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r49', 'priority': -43, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r143_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.9408897917880529, 'learning_rate': 0.01343464462043561, 'min_data_in_leaf': 21, 'num_leaves': 178, 'ag_args': {'name_suffix': '_r143', 'priority': -44, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r127_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 38572, 'min_samples_leaf': 5, 'ag_args': {'name_suffix': '_r127', 'priority': -45, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r134_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.006251885504130949, 'epochs': 47, 'layers': [800, 400], 'lr': 0.01329622020483052, 'ps': 0.2677080696008348, 'ag_args': {'name_suffix': '_r134', 'priority': -46, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r34_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 18242, 'min_samples_leaf': 40, 'ag_args': {'name_suffix': '_r34', 'priority': -47, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r94_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.4341088458599442, 'learning_rate': 0.04034449862560467, 'min_data_in_leaf': 33, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r94', 'priority': -48, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r143_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.1703783780377607, 'hidden_size': 212, 'learning_rate': 0.0004107199833213839, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.105439140660822e-07, 'ag_args': {'name_suffix': '_r143', 'priority': -49, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r128_BAG_L1: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.640921865280573, 'learning_rate': 0.036232951900213306, 'max_ctr_complexity': 3, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r128', 'priority': -50, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r111_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.6343202884164582, 'epochs': 21, 'layers': [400, 200], 'lr': 0.08479209380262258, 'ps': 0.48362560779595565, 'ag_args': {'name_suffix': '_r111', 'priority': -51, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r31_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.013288954106470907, 'hidden_size': 81, 'learning_rate': 0.005340914647396154, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 8.762168370775353e-05, 'ag_args': {'name_suffix': '_r31', 'priority': -52, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r4_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 19935, 'min_samples_leaf': 20, 'ag_args': {'name_suffix': '_r4', 'priority': -53, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r65_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.22771721361129746, 'epochs': 38, 'layers': [400], 'lr': 0.0005383511954451698, 'ps': 0.3734259772256502, 'ag_args': {'name_suffix': '_r65', 'priority': -54, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r88_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.4329361816589235, 'epochs': 50, 'layers': [400], 'lr': 0.09501311551121323, 'ps': 0.2863378667611431, 'ag_args': {'name_suffix': '_r88', 'priority': -55, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r30_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.9773131270704629, 'learning_rate': 0.010534290864227067, 'min_data_in_leaf': 21, 'num_leaves': 111, 'ag_args': {'name_suffix': '_r30', 'priority': -56, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r49_BAG_L1: 	{'colsample_bytree': 0.7452294043087835, 'enable_categorical': False, 'learning_rate': 0.038404229910104046, 'max_depth': 7, 'min_child_weight': 0.5564183327139662, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r49', 'priority': -57, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r5_BAG_L1: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.894432181094842, 'learning_rate': 0.055078095725390575, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r5', 'priority': -58, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r87_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.36669080773207274, 'hidden_size': 95, 'learning_rate': 0.015280159186761077, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.3082489374636015e-08, 'ag_args': {'name_suffix': '_r87', 'priority': -59, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r71_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.3027114570947557, 'hidden_size': 196, 'learning_rate': 0.006482759295309238, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 1.2806509958776e-12, 'ag_args': {'name_suffix': '_r71', 'priority': -60, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r143_BAG_L1: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6761016245166451, 'learning_rate': 0.06566144806528762, 'max_ctr_complexity': 2, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r143', 'priority': -61, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r178_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 29813, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r178', 'priority': -62, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r166_BAG_L1: 	{'max_features': 'log2', 'max_leaf_nodes': 42644, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r166', 'priority': -63, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r31_BAG_L1: 	{'colsample_bytree': 0.7506621909633511, 'enable_categorical': False, 'learning_rate': 0.009974712407899168, 'max_depth': 4, 'min_child_weight': 0.9238550485581797, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r31', 'priority': -64, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r185_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.12166942295569863, 'hidden_size': 151, 'learning_rate': 0.0018866871631794007, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 9.190843763153802e-05, 'ag_args': {'name_suffix': '_r185', 'priority': -65, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r160_BAG_L1: 	{'bs': 128, 'emb_drop': 0.3171659718142149, 'epochs': 20, 'layers': [400, 200, 100], 'lr': 0.03087210106068273, 'ps': 0.5909644730871169, 'ag_args': {'name_suffix': '_r160', 'priority': -66, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r60_BAG_L1: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3217885487525205, 'learning_rate': 0.05291587380674719, 'max_ctr_complexity': 5, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r60', 'priority': -67, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r15_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 36230, 'min_samples_leaf': 3, 'ag_args': {'name_suffix': '_r15', 'priority': -68, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r135_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.8254432681390782, 'learning_rate': 0.031251656439648626, 'min_data_in_leaf': 50, 'num_leaves': 210, 'ag_args': {'name_suffix': '_r135', 'priority': -69, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r22_BAG_L1: 	{'colsample_bytree': 0.6326947454697227, 'enable_categorical': False, 'learning_rate': 0.07792091886639502, 'max_depth': 6, 'min_child_weight': 1.0759464955561793, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r22', 'priority': -70, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r69_BAG_L1: 	{'bs': 128, 'emb_drop': 0.3209601865656554, 'epochs': 21, 'layers': [200, 100, 50], 'lr': 0.019935403046870463, 'ps': 0.19846319260751663, 'ag_args': {'name_suffix': '_r69', 'priority': -71, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r6_BAG_L1: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.5734131496361856, 'learning_rate': 0.08472519974533015, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r6', 'priority': -72, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r138_BAG_L1: 	{'bs': 128, 'emb_drop': 0.08669109226243704, 'epochs': 45, 'layers': [800, 400], 'lr': 0.0041554361714983635, 'ps': 0.2669780074016213, 'ag_args': {'name_suffix': '_r138', 'priority': -73, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r121_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.5730390983988963, 'learning_rate': 0.010305352949119608, 'min_data_in_leaf': 10, 'num_leaves': 215, 'ag_args': {'name_suffix': '_r121', 'priority': -74, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r172_BAG_L1: 	{'bs': 512, 'emb_drop': 0.05604276533830355, 'epochs': 32, 'layers': [400], 'lr': 0.027320709383189166, 'ps': 0.022591301744255762, 'ag_args': {'name_suffix': '_r172', 'priority': -75, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r180_BAG_L1: 	{'depth': 7, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 4.43335055453705, 'learning_rate': 0.055406199833457785, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r180', 'priority': -76, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r76_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.006531401073483156, 'hidden_size': 192, 'learning_rate': 0.012418052210914356, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 3.0406866089493607e-05, 'ag_args': {'name_suffix': '_r76', 'priority': -77, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r197_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 40459, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r197', 'priority': -78, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r121_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.33926015213879396, 'hidden_size': 247, 'learning_rate': 0.0029983839090226075, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 0.00038926240517691234, 'ag_args': {'name_suffix': '_r121', 'priority': -79, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r127_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.31956392388385874, 'epochs': 25, 'layers': [200, 100], 'lr': 0.08552736732040143, 'ps': 0.0934076022219228, 'ag_args': {'name_suffix': '_r127', 'priority': -80, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r16_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 48136, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r16', 'priority': -81, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r194_BAG_L1: 	{'bs': 256, 'emb_drop': 0.5117456464220826, 'epochs': 21, 'layers': [400, 200, 100], 'lr': 0.007212882302137526, 'ps': 0.2747013981281539, 'ag_args': {'name_suffix': '_r194', 'priority': -82, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r12_BAG_L1: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.835797074498082, 'learning_rate': 0.03534026385152556, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r12', 'priority': -83, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r135_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.06134755114373829, 'hidden_size': 144, 'learning_rate': 0.005834535148903801, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 2.0826540090463355e-09, 'ag_args': {'name_suffix': '_r135', 'priority': -84, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r4_BAG_L1: 	{'bs': 256, 'emb_drop': 0.06099050979107849, 'epochs': 39, 'layers': [200], 'lr': 0.04119582873110387, 'ps': 0.5447097256648953, 'ag_args': {'name_suffix': '_r4', 'priority': -85, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r126_BAG_L1: 	{'max_features': 'sqrt', 'max_leaf_nodes': 29702, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r126', 'priority': -86, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r36_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.3457125770744979, 'hidden_size': 37, 'learning_rate': 0.006435774191713849, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 2.4012185204155345e-08, 'ag_args': {'name_suffix': '_r36', 'priority': -87, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r100_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.6960805527533755, 'epochs': 38, 'layers': [800, 400], 'lr': 0.0007278526871749883, 'ps': 0.20495582200836318, 'ag_args': {'name_suffix': '_r100', 'priority': -88, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r163_BAG_L1: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.7454481983750014, 'learning_rate': 0.09328642499990342, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r163', 'priority': -89, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r198_BAG_L1: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.637071465711953, 'learning_rate': 0.04387418552563314, 'max_ctr_complexity': 4, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r198', 'priority': -90, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r187_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.5074958658302495, 'epochs': 42, 'layers': [200, 100, 50], 'lr': 0.026342427824862867, 'ps': 0.34814978753283593, 'ag_args': {'name_suffix': '_r187', 'priority': -91, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r19_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.2211285919550286, 'hidden_size': 196, 'learning_rate': 0.011307978270179143, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 1.8441764217351068e-06, 'ag_args': {'name_suffix': '_r19', 'priority': -92, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r95_BAG_L1: 	{'colsample_bytree': 0.975937238416368, 'enable_categorical': False, 'learning_rate': 0.06634196266155237, 'max_depth': 5, 'min_child_weight': 1.4088437184127383, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r95', 'priority': -93, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r34_BAG_L1: 	{'colsample_bytree': 0.546186944730449, 'enable_categorical': False, 'learning_rate': 0.029357102578825213, 'max_depth': 10, 'min_child_weight': 1.1532008198571337, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r34', 'priority': -94, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r42_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.4601361323873807, 'learning_rate': 0.07856777698860955, 'min_data_in_leaf': 12, 'num_leaves': 198, 'ag_args': {'name_suffix': '_r42', 'priority': -95, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r1_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.23713784729000734, 'hidden_size': 200, 'learning_rate': 0.00311256170909018, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 4.573016756474468e-08, 'ag_args': {'name_suffix': '_r1', 'priority': -96, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r89_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.33567564890346097, 'hidden_size': 245, 'learning_rate': 0.006746560197328548, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.6470047305392933e-10, 'ag_args': {'name_suffix': '_r89', 'priority': -97, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m Fitting 110 L1 models, fit_strategy="sequential" ...
    [36m(_dystack pid=167306)[0m Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 572.78s of the 859.37s of remaining time.
    [36m(_dystack pid=167306)[0m 	No valid features to train KNeighborsUnif_BAG_L1... Skipping this model.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 572.75s of the 859.34s of remaining time.
    [36m(_dystack pid=167306)[0m 	No valid features to train KNeighborsDist_BAG_L1... Skipping this model.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 572.72s of the 859.32s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting LightGBMXT_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.68%)
    [36m(_ray_fit pid=170789)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05, 'extra_trees': True}
    [36m(_dystack pid=167306)[0m 	0.7763	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	2.96s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.21s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	3646504.3	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: LightGBM_BAG_L1 ... Training model for up to 567.75s of the 854.35s of remaining time.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl[32m [repeated 11x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
    [36m(_dystack pid=167306)[0m 	Fitting LightGBM_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.70%)
    [36m(_ray_fit pid=171586)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05}
    [36m(_ray_fit pid=170785)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05, 'extra_trees': True}[32m [repeated 7x across cluster][0m
    [36m(_dystack pid=167306)[0m 	0.7763	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	3.0s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.25s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	3088899.3	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: RandomForestGini_BAG_L1 ... Training model for up to 562.81s of the 849.41s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting RandomForestGini_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/utils/model_template.pkl[32m [repeated 13x across cluster][0m
    [36m(_dystack pid=167306)[0m 	454.45s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m 	`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.
    [36m(_ray_fit pid=171585)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05}[32m [repeated 7x across cluster][0m
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m 	0.7785	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	66.03s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	52.05s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	119453.5	 = Inference  throughput (rows/s | 6217827 batch size)
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Fitting model: RandomForestEntr_BAG_L1 ... Training model for up to 443.92s of the 730.52s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting RandomForestEntr_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	301.99s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m 	`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m 	0.7785	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	64.56s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	51.61s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	120467.3	 = Inference  throughput (rows/s | 6217827 batch size)
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Fitting model: CatBoost_BAG_L1 ... Training model for up to 326.98s of the 613.58s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting CatBoost_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.71%)
    [36m(_ray_fit pid=177652)[0m 	Catboost model hyperparameters: {'iterations': 10000, 'learning_rate': 0.05, 'random_seed': 0, 'allow_writing_files': False, 'eval_metric': 'Accuracy', 'thread_count': 4}


    [36m(_ray_fit pid=177651)[0m 0:	learn: 0.7773892	test: 0.7781655	best: 0.7781655 (0)	total: 3.5s	remaining: 9h 42m 56s
    [36m(_ray_fit pid=177651)[0m 
    [36m(_ray_fit pid=177651)[0m bestTest = 0.7781654804
    [36m(_ray_fit pid=177651)[0m bestIteration = 0
    [36m(_ray_fit pid=177651)[0m 
    [36m(_ray_fit pid=177651)[0m Shrink model to first 1 iterations.
    [36m(_ray_fit pid=177650)[0m 0:	learn: 0.7773745	test: 0.7782723	best: 0.7782723 (0)	total: 4.63s	remaining: 12h 51m 50s[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=177649)[0m 
    [36m(_ray_fit pid=177649)[0m 


    [36m(_ray_fit pid=177649)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/S1F4/model.pkl
    [36m(_ray_fit pid=177650)[0m 	Catboost model hyperparameters: {'iterations': 10000, 'learning_rate': 0.05, 'random_seed': 0, 'allow_writing_files': False, 'eval_metric': 'Accuracy', 'thread_count': 4}[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=177651)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/S1F6/model.pkl


    [36m(_ray_fit pid=177647)[0m 
    [36m(_ray_fit pid=177647)[0m 
    [36m(_ray_fit pid=177648)[0m 
    [36m(_ray_fit pid=177648)[0m 
    [36m(_ray_fit pid=177645)[0m 
    [36m(_ray_fit pid=177645)[0m 
    [36m(_ray_fit pid=177650)[0m 
    [36m(_ray_fit pid=177650)[0m 
    [36m(_ray_fit pid=177652)[0m 
    [36m(_ray_fit pid=177652)[0m 
    [36m(_ray_fit pid=177646)[0m 
    [36m(_ray_fit pid=177646)[0m 


    [36m(_dystack pid=167306)[0m 	0.7783	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	76.24s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.74s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	1053958.1	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: ExtraTreesGini_BAG_L1 ... Training model for up to 248.82s of the 535.42s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting ExtraTreesGini_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/ExtraTreesGini_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	444.5s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m 	Not enough time to generate out-of-fold predictions for model. Estimated time required was 444.5s compared to 240.17s of available time.
    [36m(_dystack pid=167306)[0m 	Time limit exceeded... Skipping ExtraTreesGini_BAG_L1.
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/ExtraTreesGini_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Fitting model: ExtraTreesEntr_BAG_L1 ... Training model for up to 165.38s of the 451.98s of remaining time.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl[32m [repeated 11x across cluster][0m
    [36m(_dystack pid=167306)[0m 	Fitting ExtraTreesEntr_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/ExtraTreesEntr_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	560.05s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m 	Not enough time to generate out-of-fold predictions for model. Estimated time required was 560.05s compared to 131.7s of available time.
    [36m(_dystack pid=167306)[0m 	Time limit exceeded... Skipping ExtraTreesEntr_BAG_L1.
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/ExtraTreesEntr_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 81.94s of the 368.54s of remaining time.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl[32m [repeated 2x across cluster][0m
    [36m(_dystack pid=167306)[0m 	Fitting NeuralNetFastAI_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/NeuralNetFastAI_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=1.15%)
    [36m(_ray_fit pid=184097)[0m Fitting Neural Network with parameters {'layers': None, 'emb_drop': 0.1, 'ps': 0.1, 'bs': 'auto', 'lr': 0.01, 'epochs': 'auto', 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}...
    [36m(_ray_fit pid=184093)[0m Using 0/1 categorical features
    [36m(_ray_fit pid=184093)[0m Using 0 cont features
    [36m(_ray_fit pid=184093)[0m Automated batch size selection: 512
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/NeuralNetFastAI_BAG_L1/utils/model_template.pkl
    [36m(_ray_fit pid=184096)[0m /home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/init.py:511: UserWarning: Initializing zero-element tensors is a no-op
    [36m(_ray_fit pid=184096)[0m   warnings.warn("Initializing zero-element tensors is a no-op")
    [36m(_ray_fit pid=184096)[0m TabularModel(
    [36m(_ray_fit pid=184096)[0m   (embeds): ModuleList()
    [36m(_ray_fit pid=184096)[0m   (emb_drop): Dropout(p=0.1, inplace=False)
    [36m(_ray_fit pid=184096)[0m   (bn_cont): BatchNorm1d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    [36m(_ray_fit pid=184096)[0m   (layers): Sequential(
    [36m(_ray_fit pid=184096)[0m     (0): LinBnDrop(
    [36m(_ray_fit pid=184096)[0m       (0): Linear(in_features=0, out_features=200, bias=False)
    [36m(_ray_fit pid=184096)[0m       (1): ReLU(inplace=True)
    [36m(_ray_fit pid=184096)[0m       (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    [36m(_ray_fit pid=184096)[0m       (3): Dropout(p=0.1, inplace=False)
    [36m(_ray_fit pid=184096)[0m     )
    [36m(_ray_fit pid=184096)[0m     (1): LinBnDrop(
    [36m(_ray_fit pid=184096)[0m       (0): Linear(in_features=200, out_features=100, bias=False)
    [36m(_ray_fit pid=184096)[0m       (1): ReLU(inplace=True)
    [36m(_ray_fit pid=184096)[0m       (2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    [36m(_ray_fit pid=184096)[0m       (3): Dropout(p=0.1, inplace=False)
    [36m(_ray_fit pid=184096)[0m     )
    [36m(_ray_fit pid=184096)[0m     (2): LinBnDrop(
    [36m(_ray_fit pid=184096)[0m       (0): Linear(in_features=100, out_features=2, bias=True)
    [36m(_ray_fit pid=184096)[0m     )
    [36m(_ray_fit pid=184096)[0m   )
    [36m(_ray_fit pid=184096)[0m )
    [36m(_ray_fit pid=184095)[0m Fitting Neural Network with parameters {'layers': None, 'emb_drop': 0.1, 'ps': 0.1, 'bs': 'auto', 'lr': 0.01, 'epochs': 'auto', 'early.stopping.min_delta': 0.0001, 'early.stopping.patience': 20, 'smoothing': 0.0}...[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184095)[0m Using 0/1 categorical features[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184095)[0m Using 0 cont features[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184098)[0m Automated batch size selection: 512[32m [repeated 7x across cluster][0m
    [36m(_dystack pid=167306)[0m 	Warning: Exception caused NeuralNetFastAI_BAG_L1 to fail during training... Skipping this model.
    [36m(_dystack pid=167306)[0m 		[36mray::_ray_fit()[39m (pid=184093, ip=172.30.1.59)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 413, in _ray_fit
    [36m(_dystack pid=167306)[0m     fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **resources, **kwargs_fold)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/abstract/abstract_model.py", line 925, in fit
    [36m(_dystack pid=167306)[0m     out = self._fit(**kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 361, in _fit
    [36m(_dystack pid=167306)[0m     epochs = self._get_epochs_number(samples_num=len(X) + len_val, epochs=params["epochs"], batch_size=batch_size, time_left=time_left)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 410, in _get_epochs_number
    [36m(_dystack pid=167306)[0m     est_batch_time = self._measure_batch_times(min_batches_count)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 431, in _measure_batch_times
    [36m(_dystack pid=167306)[0m     self.model.fit(1, lr=0, cbs=[batch_time_tracker_callback])
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 266, in fit
    [36m(_dystack pid=167306)[0m     self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 255, in _do_fit
    [36m(_dystack pid=167306)[0m     self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 249, in _do_epoch
    [36m(_dystack pid=167306)[0m     self._do_epoch_train()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 241, in _do_epoch_train
    [36m(_dystack pid=167306)[0m     self._with_events(self.all_batches, 'train', CancelTrainException)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 207, in all_batches
    [36m(_dystack pid=167306)[0m     for o in enumerate(self.dl): self.one_batch(*o)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 237, in one_batch
    [36m(_dystack pid=167306)[0m     self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 218, in _do_one_batch
    [36m(_dystack pid=167306)[0m     self.pred = self.model(*self.xb)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    [36m(_dystack pid=167306)[0m     return self._call_impl(*args, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    [36m(_dystack pid=167306)[0m     return forward_call(*args, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/tabular/model.py", line 75, in forward
    [36m(_dystack pid=167306)[0m     return self.layers(x)
    [36m(_dystack pid=167306)[0m UnboundLocalError: local variable 'x' referenced before assignment
    [36m(_dystack pid=167306)[0m Detailed Traceback:
    [36m(_dystack pid=167306)[0m Traceback (most recent call last):
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/trainer/abstract_trainer.py", line 2106, in _train_and_save
    [36m(_dystack pid=167306)[0m     model = self._train_single(**model_fit_kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1993, in _train_single
    [36m(_dystack pid=167306)[0m     model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, total_resources=total_resources, **model_fit_kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/abstract/abstract_model.py", line 925, in fit
    [36m(_dystack pid=167306)[0m     out = self._fit(**kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 270, in _fit
    [36m(_dystack pid=167306)[0m     return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 298, in _fit
    [36m(_dystack pid=167306)[0m     self._fit_folds(
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 724, in _fit_folds
    [36m(_dystack pid=167306)[0m     fold_fitting_strategy.after_all_folds_scheduled()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 690, in after_all_folds_scheduled
    [36m(_dystack pid=167306)[0m     self._run_parallel(X, y, X_pseudo, y_pseudo, model_base_ref, time_limit_fold, head_node_id)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 631, in _run_parallel
    [36m(_dystack pid=167306)[0m     self._process_fold_results(finished, unfinished, fold_ctx)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 587, in _process_fold_results
    [36m(_dystack pid=167306)[0m     raise processed_exception
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 550, in _process_fold_results
    [36m(_dystack pid=167306)[0m     fold_model, pred_proba, time_start_fit, time_end_fit, predict_time, predict_1_time, predict_n_size, fit_num_cpus, fit_num_gpus = self.ray.get(finished)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    [36m(_dystack pid=167306)[0m     return fn(*args, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    [36m(_dystack pid=167306)[0m     return func(*args, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/worker.py", line 2753, in get
    [36m(_dystack pid=167306)[0m     values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/worker.py", line 904, in get_objects
    [36m(_dystack pid=167306)[0m     raise value.as_instanceof_cause()
    [36m(_dystack pid=167306)[0m ray.exceptions.RayTaskError(UnboundLocalError): [36mray::_ray_fit()[39m (pid=184093, ip=172.30.1.59)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 413, in _ray_fit
    [36m(_dystack pid=167306)[0m     fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **resources, **kwargs_fold)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/abstract/abstract_model.py", line 925, in fit
    [36m(_dystack pid=167306)[0m     out = self._fit(**kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 361, in _fit
    [36m(_dystack pid=167306)[0m     epochs = self._get_epochs_number(samples_num=len(X) + len_val, epochs=params["epochs"], batch_size=batch_size, time_left=time_left)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 410, in _get_epochs_number
    [36m(_dystack pid=167306)[0m     est_batch_time = self._measure_batch_times(min_batches_count)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 431, in _measure_batch_times
    [36m(_dystack pid=167306)[0m     self.model.fit(1, lr=0, cbs=[batch_time_tracker_callback])
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 266, in fit
    [36m(_dystack pid=167306)[0m     self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 255, in _do_fit
    [36m(_dystack pid=167306)[0m     self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 249, in _do_epoch
    [36m(_dystack pid=167306)[0m     self._do_epoch_train()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 241, in _do_epoch_train
    [36m(_dystack pid=167306)[0m     self._with_events(self.all_batches, 'train', CancelTrainException)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 207, in all_batches
    [36m(_dystack pid=167306)[0m     for o in enumerate(self.dl): self.one_batch(*o)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 237, in one_batch
    [36m(_dystack pid=167306)[0m     self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
    [36m(_dystack pid=167306)[0m     try: self(f'before_{event_type}');  f()
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 218, in _do_one_batch
    [36m(_dystack pid=167306)[0m     self.pred = self.model(*self.xb)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
    [36m(_dystack pid=167306)[0m     return self._call_impl(*args, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    [36m(_dystack pid=167306)[0m     return forward_call(*args, **kwargs)
    [36m(_dystack pid=167306)[0m   File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/tabular/model.py", line 75, in forward
    [36m(_dystack pid=167306)[0m     return self.layers(x)
    [36m(_dystack pid=167306)[0m UnboundLocalError: local variable 'x' referenced before assignment
    [36m(_ray_fit pid=184099)[0m /home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/init.py:511: UserWarning: Initializing zero-element tensors is a no-op[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m   warnings.warn("Initializing zero-element tensors is a no-op")[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m TabularModel([32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m   (embeds): ModuleList()[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m   (emb_drop): Dropout(p=0.1, inplace=False)[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m   (bn_cont): BatchNorm1d(0, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m   (layers): Sequential([32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=184099)[0m     (2): LinBnDrop([32m [repeated 21x across cluster][0m
    [36m(_ray_fit pid=184099)[0m       (0): Linear(in_features=200, out_features=100, bias=False)[32m [repeated 14x across cluster][0m
    [36m(_ray_fit pid=184099)[0m       (1): ReLU(inplace=True)[32m [repeated 14x across cluster][0m
    [36m(_ray_fit pid=184099)[0m       (2): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)[32m [repeated 14x across cluster][0m
    [36m(_ray_fit pid=184099)[0m       (3): Dropout(p=0.1, inplace=False)[32m [repeated 14x across cluster][0m
    [36m(_ray_fit pid=184099)[0m )[32m [repeated 35x across cluster][0m
    [36m(_ray_fit pid=184099)[0m       (0): Linear(in_features=100, out_features=2, bias=True)[32m [repeated 7x across cluster][0m
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/NeuralNetFastAI_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Fitting model: XGBoost_BAG_L1 ... Training model for up to 47.56s of the 334.16s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting XGBoost_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.91%)


    [36m(_ray_fit pid=185774)[0m [0]	validation_0-error:0.22371
    [36m(_ray_fit pid=177646)[0m bestTest = 0.7783870648[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=177646)[0m bestIteration = 0[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=177646)[0m Shrink model to first 1 iterations.[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=185776)[0m [0]	validation_0-error:0.22371
    [36m(_ray_fit pid=185771)[0m [0]	validation_0-error:0.22372
    [36m(_ray_fit pid=185772)[0m [0]	validation_0-error:0.22372
    [36m(_ray_fit pid=185770)[0m [0]	validation_0-error:0.22371
    [36m(_ray_fit pid=185775)[0m [0]	validation_0-error:0.22371
    [36m(_ray_fit pid=185769)[0m [0]	validation_0-error:0.22372
    [36m(_ray_fit pid=185773)[0m [0]	validation_0-error:0.22371
    [36m(_ray_fit pid=185775)[0m [20]	validation_0-error:0.22374
    [36m(_ray_fit pid=185776)[0m [20]	validation_0-error:0.22374
    [36m(_ray_fit pid=185770)[0m [20]	validation_0-error:0.22377


    [36m(_ray_fit pid=185775)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/S1F7/model.pkl
    [36m(_ray_fit pid=185776)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/S1F8/model.pkl


    [36m(_ray_fit pid=185774)[0m [24]	validation_0-error:0.22368
    [36m(_ray_fit pid=185771)[0m [24]	validation_0-error:0.22371
    [36m(_ray_fit pid=185772)[0m [24]	validation_0-error:0.22370
    [36m(_ray_fit pid=185769)[0m [23]	validation_0-error:0.22366
    [36m(_ray_fit pid=185773)[0m [24]	validation_0-error:0.22370


    [36m(_dystack pid=167306)[0m 	0.7763	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	24.9s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	3.12s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	249309.2	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 20.45s of the 307.05s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting NeuralNetTorch_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/NeuralNetTorch_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.68%)
    [36m(_ray_fit pid=187116)[0m Tabular Neural Network treats features as the following types:
    [36m(_ray_fit pid=187116)[0m {
    [36m(_ray_fit pid=187116)[0m     "continuous": [],
    [36m(_ray_fit pid=187116)[0m     "skewed": [],
    [36m(_ray_fit pid=187116)[0m     "onehot": [],
    [36m(_ray_fit pid=187116)[0m     "embed": [
    [36m(_ray_fit pid=187116)[0m         "URL"
    [36m(_ray_fit pid=187116)[0m     ],
    [36m(_ray_fit pid=187116)[0m     "language": [],
    [36m(_ray_fit pid=187116)[0m     "bool": []
    [36m(_ray_fit pid=187116)[0m }
    [36m(_ray_fit pid=187116)[0m 
    [36m(_ray_fit pid=187116)[0m 
    [36m(_ray_fit pid=187114)[0m 
    [36m(_ray_fit pid=187114)[0m 
    [36m(_ray_fit pid=187119)[0m 
    [36m(_ray_fit pid=187119)[0m 
    [36m(_ray_fit pid=187120)[0m 
    [36m(_ray_fit pid=187120)[0m 
    [36m(_ray_fit pid=187115)[0m 
    [36m(_ray_fit pid=187115)[0m 
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/NeuralNetTorch_BAG_L1/utils/model_template.pkl[32m [repeated 10x across cluster][0m
    [36m(_ray_fit pid=187117)[0m 
    [36m(_ray_fit pid=187117)[0m 
    [36m(_ray_fit pid=187113)[0m 
    [36m(_ray_fit pid=187113)[0m 
    [36m(_ray_fit pid=187118)[0m 
    [36m(_ray_fit pid=187118)[0m 
    [36m(_ray_fit pid=187114)[0m Training data for TabularNeuralNetTorchModel has: 5440598 examples, 1 features (0 vector, 1 embedding)
    [36m(_ray_fit pid=187114)[0m Training on CPU
    [36m(_ray_fit pid=187118)[0m Tabular Neural Network treats features as the following types:[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m {[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     "continuous": [],[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     "skewed": [],[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     "onehot": [],[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     "embed": [[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m         "URL"[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     ],[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     "language": [],[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m     "bool": [][32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187118)[0m }[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187114)[0m Neural network architecture:
    [36m(_ray_fit pid=187114)[0m EmbedNet(
    [36m(_ray_fit pid=187114)[0m   (embed_blocks): ModuleList(
    [36m(_ray_fit pid=187114)[0m     (0): Embedding(102, 21)
    [36m(_ray_fit pid=187114)[0m   )
    [36m(_ray_fit pid=187114)[0m   (main_block): Sequential(
    [36m(_ray_fit pid=187114)[0m     (0): Linear(in_features=21, out_features=128, bias=True)
    [36m(_ray_fit pid=187114)[0m     (1): ReLU()
    [36m(_ray_fit pid=187114)[0m     (2): Dropout(p=0.1, inplace=False)
    [36m(_ray_fit pid=187114)[0m     (3): Linear(in_features=128, out_features=128, bias=True)
    [36m(_ray_fit pid=187114)[0m     (4): ReLU()
    [36m(_ray_fit pid=187114)[0m     (5): Dropout(p=0.1, inplace=False)
    [36m(_ray_fit pid=187114)[0m     (6): Linear(in_features=128, out_features=128, bias=True)
    [36m(_ray_fit pid=187114)[0m     (7): ReLU()
    [36m(_ray_fit pid=187114)[0m     (8): Dropout(p=0.1, inplace=False)
    [36m(_ray_fit pid=187114)[0m     (9): Linear(in_features=128, out_features=128, bias=True)
    [36m(_ray_fit pid=187114)[0m     (10): ReLU()
    [36m(_ray_fit pid=187114)[0m     (11): Linear(in_features=128, out_features=2, bias=True)
    [36m(_ray_fit pid=187114)[0m   )
    [36m(_ray_fit pid=187114)[0m   (softmax): Softmax(dim=1)
    [36m(_ray_fit pid=187114)[0m )
    [36m(_ray_fit pid=187114)[0m Training tabular neural network for up to 1000 epochs...
    [36m(_ray_fit pid=187116)[0m     (0): Embedding(102, 21)
    [36m(_ray_fit pid=187115)[0m     (0): Embedding(102, 21)
    [36m(_ray_fit pid=187117)[0m     (0): Embedding(102, 21)
    [36m(_dystack pid=167306)[0m 	Time limit exceeded... Skipping NeuralNetTorch_BAG_L1.
    [36m(_ray_fit pid=187114)[0m 	Not enough time to train first epoch. (Time Required: 47.71s, Time Left: 7.21s)
    [36m(_ray_fit pid=187113)[0m     (0): Embedding(102, 21)
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/NeuralNetTorch_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Fitting model: LightGBMLarge_BAG_L1 ... Training model for up to 8.96s of the 295.55s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting LightGBMLarge_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.74%)
    [36m(_ray_fit pid=188346)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3}
    [36m(_ray_fit pid=187120)[0m Training data for TabularNeuralNetTorchModel has: 5440599 examples, 1 features (0 vector, 1 embedding)[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187120)[0m Training on CPU[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=187113)[0m Neural network architecture:[32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=187113)[0m EmbedNet([32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=187113)[0m   (embed_blocks): ModuleList([32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=187113)[0m )[32m [repeated 12x across cluster][0m
    [36m(_ray_fit pid=187113)[0m   (main_block): Sequential([32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=187113)[0m     (11): Linear(in_features=128, out_features=2, bias=True)[32m [repeated 20x across cluster][0m
    [36m(_ray_fit pid=187113)[0m     (10): ReLU()[32m [repeated 16x across cluster][0m
    [36m(_ray_fit pid=187113)[0m     (8): Dropout(p=0.1, inplace=False)[32m [repeated 12x across cluster][0m
    [36m(_ray_fit pid=187113)[0m   (softmax): Softmax(dim=1)[32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=187113)[0m Training tabular neural network for up to 1000 epochs...[32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=188350)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/S1F8/model.pkl[32m [repeated 8x across cluster][0m
    [36m(_dystack pid=167306)[0m 	0.7763	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	3.42s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.26s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	3004606.3	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: CatBoost_r177_BAG_L1 ... Training model for up to 3.52s of the 290.12s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting CatBoost_r177_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_r177_BAG_L1/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.73%)
    [36m(_ray_fit pid=189224)[0m 	Catboost model hyperparameters: {'iterations': 10000, 'learning_rate': 0.06864209415792857, 'random_seed': 0, 'allow_writing_files': False, 'eval_metric': 'Accuracy', 'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'thread_count': 4}
    [36m(_ray_fit pid=188348)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3}[32m [repeated 7x across cluster][0m


    [36m(_ray_fit pid=189224)[0m 0:	learn: 0.7773785	test: 0.7782738	best: 0.7782738 (0)	total: 3.77s	remaining: 10h 28m
    [36m(_ray_fit pid=189224)[0m 
    [36m(_ray_fit pid=189224)[0m bestTest = 0.7782738421
    [36m(_ray_fit pid=189224)[0m bestIteration = 0
    [36m(_ray_fit pid=189224)[0m 
    [36m(_ray_fit pid=189224)[0m Shrink model to first 1 iterations.
    [36m(_ray_fit pid=189221)[0m 
    [36m(_ray_fit pid=189221)[0m 


    [36m(_ray_fit pid=189224)[0m 	Ran out of time, early stopping on iteration 1.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_r177_BAG_L1/utils/model_template.pkl[32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=189222)[0m 	Catboost model hyperparameters: {'iterations': 10000, 'learning_rate': 0.06864209415792857, 'random_seed': 0, 'allow_writing_files': False, 'eval_metric': 'Accuracy', 'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'thread_count': 4}[32m [repeated 7x across cluster][0m


    [36m(_ray_fit pid=189218)[0m 
    [36m(_ray_fit pid=189218)[0m 
    [36m(_ray_fit pid=189220)[0m 
    [36m(_ray_fit pid=189220)[0m 
    [36m(_ray_fit pid=189219)[0m 
    [36m(_ray_fit pid=189219)[0m 
    [36m(_ray_fit pid=189222)[0m 
    [36m(_ray_fit pid=189222)[0m 
    [36m(_ray_fit pid=189217)[0m 
    [36m(_ray_fit pid=189217)[0m 
    [36m(_ray_fit pid=189223)[0m 
    [36m(_ray_fit pid=189223)[0m 


    [36m(_dystack pid=167306)[0m 	0.7783	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	7.37s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.5s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	1558094.8	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r79_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r131_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r191_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r9_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r96_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r22_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r33_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r42_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r137_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r102_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r13_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r195_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r188_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r145_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r89_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r30_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r130_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r86_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r50_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r11_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r194_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r172_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r69_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r103_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r14_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r161_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r143_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r70_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r156_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r196_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r39_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r167_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r95_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r41_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r98_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r15_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r158_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r86_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r37_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r197_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r49_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r49_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r143_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r127_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r134_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r34_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r94_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r143_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r128_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r111_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r31_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r4_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r65_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r88_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r30_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r49_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r5_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r87_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r71_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r143_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r178_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r166_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r31_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r185_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r160_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r60_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r15_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r135_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r22_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r69_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r6_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r138_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r121_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r172_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r180_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r76_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r197_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r121_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r127_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r16_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r194_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r12_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r135_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r4_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r126_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r36_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r100_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r163_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r198_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r187_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r19_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r95_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r34_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r42_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r1_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r89_BAG_L1 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Model configs that will be trained (in order):
    [36m(_dystack pid=167306)[0m 	WeightedEnsemble_L2: 	{'ag_args': {'valid_base': False, 'name_bag_suffix': '', 'model_type': <class 'autogluon.core.models.greedy_ensemble.greedy_weighted_ensemble_model.GreedyWeightedEnsembleModel'>, 'priority': 0}, 'ag_args_ensemble': {'save_bag_folds': True}}
    [36m(_dystack pid=167306)[0m Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.00s of the 280.43s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting WeightedEnsemble_L2 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Subsampling to 1000000 samples to speedup ensemble selection...
    [36m(_dystack pid=167306)[0m Ensemble size: 2
    [36m(_dystack pid=167306)[0m Ensemble weights: 
    [36m(_dystack pid=167306)[0m [0.  0.  0.5 0.  0.  0.  0.  0.5]
    [36m(_ray_fit pid=189217)[0m 	Ran out of time, early stopping on iteration 1.[32m [repeated 7x across cluster][0m
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L2/utils/model_template.pkl[32m [repeated 108x across cluster][0m
    [36m(_dystack pid=167306)[0m 	5.5s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m 	Ensemble Weights: {'RandomForestGini_BAG_L1': 0.5, 'CatBoost_r177_BAG_L1': 0.5}
    [36m(_dystack pid=167306)[0m 	0.7785	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	6.02s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.25s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	110460.6	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Model configs that will be trained (in order):
    [36m(_dystack pid=167306)[0m 	LightGBMXT_BAG_L2: 	{'extra_trees': True, 'ag_args': {'name_suffix': 'XT', 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    [36m(_dystack pid=167306)[0m 	LightGBM_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    [36m(_dystack pid=167306)[0m 	RandomForestGini_BAG_L2: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	RandomForestEntr_BAG_L2: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>, 'priority': 70}}
    [36m(_dystack pid=167306)[0m 	ExtraTreesGini_BAG_L2: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	ExtraTreesEntr_BAG_L2: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>, 'priority': 50}}
    [36m(_dystack pid=167306)[0m 	XGBoost_BAG_L2: 	{'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>, 'priority': 40}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>, 'priority': 25}}
    [36m(_dystack pid=167306)[0m 	LightGBMLarge_BAG_L2: 	{'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r177_BAG_L2: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r79_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r131_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.7023601671276614, 'learning_rate': 0.012144796373999013, 'min_data_in_leaf': 14, 'num_leaves': 53, 'ag_args': {'name_suffix': '_r131', 'priority': -3, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r191_BAG_L2: 	{'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r9_BAG_L2: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r96_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.5636931414546802, 'learning_rate': 0.01518660230385841, 'min_data_in_leaf': 48, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r96', 'priority': -6, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r22_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r33_BAG_L2: 	{'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r33', 'priority': -8, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r42_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 18392, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r42', 'priority': -9, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r137_BAG_L2: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.559174625782161, 'learning_rate': 0.04939557741379516, 'max_ctr_complexity': 3, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r137', 'priority': -10, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r102_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r13_BAG_L2: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3274013177541373, 'learning_rate': 0.017301189655111057, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r13', 'priority': -12, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r195_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 37308, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r195', 'priority': -13, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r188_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.8282601210460099, 'learning_rate': 0.033929021353492905, 'min_data_in_leaf': 6, 'num_leaves': 127, 'ag_args': {'name_suffix': '_r188', 'priority': -14, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r145_BAG_L2: 	{'bs': 128, 'emb_drop': 0.44339037504795686, 'epochs': 31, 'layers': [400, 200, 100], 'lr': 0.008615195908919904, 'ps': 0.19220253419114286, 'ag_args': {'name_suffix': '_r145', 'priority': -15, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r89_BAG_L2: 	{'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r89', 'priority': -16, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r30_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.24622382571353768, 'hidden_size': 159, 'learning_rate': 0.008507536855608535, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.8201539594953562e-06, 'ag_args': {'name_suffix': '_r30', 'priority': -17, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r130_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.6245777099925497, 'learning_rate': 0.04711573688184715, 'min_data_in_leaf': 56, 'num_leaves': 89, 'ag_args': {'name_suffix': '_r130', 'priority': -18, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r86_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.09976801642258049, 'hidden_size': 135, 'learning_rate': 0.001631450730978947, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 3.867683394425807e-05, 'ag_args': {'name_suffix': '_r86', 'priority': -19, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r50_BAG_L2: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7018061518087038, 'learning_rate': 0.07092851311746352, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r50', 'priority': -20, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r11_BAG_L2: 	{'bs': 128, 'emb_drop': 0.026897798530914306, 'epochs': 31, 'layers': [800, 400], 'lr': 0.08045277634470181, 'ps': 0.4569532219038436, 'ag_args': {'name_suffix': '_r11', 'priority': -21, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r194_BAG_L2: 	{'colsample_bytree': 0.9090166528779192, 'enable_categorical': True, 'learning_rate': 0.09290221350439203, 'max_depth': 7, 'min_child_weight': 0.8041986915994078, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r194', 'priority': -22, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r172_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 12845, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r172', 'priority': -23, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r69_BAG_L2: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.0457098345001241, 'learning_rate': 0.050294288910022224, 'max_ctr_complexity': 5, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r69', 'priority': -24, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r103_BAG_L2: 	{'bs': 256, 'emb_drop': 0.1508701680951814, 'epochs': 46, 'layers': [400, 200], 'lr': 0.08794353125787312, 'ps': 0.19110623090573325, 'ag_args': {'name_suffix': '_r103', 'priority': -25, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r14_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.3905837860053583, 'hidden_size': 106, 'learning_rate': 0.0018297905295930797, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 9.178069874232892e-08, 'ag_args': {'name_suffix': '_r14', 'priority': -26, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r161_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.5898927512279213, 'learning_rate': 0.010464516487486093, 'min_data_in_leaf': 11, 'num_leaves': 252, 'ag_args': {'name_suffix': '_r161', 'priority': -27, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r143_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.6239200452002372, 'epochs': 39, 'layers': [200, 100, 50], 'lr': 0.07170321592506483, 'ps': 0.670815151683455, 'ag_args': {'name_suffix': '_r143', 'priority': -28, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r70_BAG_L2: 	{'depth': 6, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.3584121369544215, 'learning_rate': 0.03743901034980473, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r70', 'priority': -29, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r156_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.5055288166864152, 'epochs': 44, 'layers': [400], 'lr': 0.0047762208542912405, 'ps': 0.06572612802222005, 'ag_args': {'name_suffix': '_r156', 'priority': -30, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r196_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.5143401489640409, 'learning_rate': 0.00529479887023554, 'min_data_in_leaf': 6, 'num_leaves': 133, 'ag_args': {'name_suffix': '_r196', 'priority': -31, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r39_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 28310, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r39', 'priority': -32, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r167_BAG_L2: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.522712492188319, 'learning_rate': 0.08481607830570326, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r167', 'priority': -33, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r95_BAG_L2: 	{'bs': 128, 'emb_drop': 0.6656668277387758, 'epochs': 32, 'layers': [400, 200, 100], 'lr': 0.019326244622675428, 'ps': 0.04084945128641206, 'ag_args': {'name_suffix': '_r95', 'priority': -34, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r41_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.05488816803887784, 'hidden_size': 32, 'learning_rate': 0.0075612897834015985, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.652353009917866e-08, 'ag_args': {'name_suffix': '_r41', 'priority': -35, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r98_BAG_L2: 	{'colsample_bytree': 0.516652313273348, 'enable_categorical': True, 'learning_rate': 0.007158072983547058, 'max_depth': 9, 'min_child_weight': 0.8567068904025429, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r98', 'priority': -36, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r15_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.7421180622507277, 'learning_rate': 0.018603888565740096, 'min_data_in_leaf': 6, 'num_leaves': 22, 'ag_args': {'name_suffix': '_r15', 'priority': -37, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r158_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.01030258381183309, 'hidden_size': 111, 'learning_rate': 0.01845979186513771, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 0.00020238017476912164, 'ag_args': {'name_suffix': '_r158', 'priority': -38, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r86_BAG_L2: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6376578537958237, 'learning_rate': 0.032899230324940465, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r86', 'priority': -39, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r37_BAG_L2: 	{'bs': 512, 'emb_drop': 0.1567472816422661, 'epochs': 41, 'layers': [400, 200, 100], 'lr': 0.06831450078222204, 'ps': 0.4930900813464729, 'ag_args': {'name_suffix': '_r37', 'priority': -40, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r197_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.18109219857068798, 'hidden_size': 250, 'learning_rate': 0.00634181748507711, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 5.3861175580695396e-08, 'ag_args': {'name_suffix': '_r197', 'priority': -41, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r49_BAG_L2: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.353268454214423, 'learning_rate': 0.06028218319511302, 'max_ctr_complexity': 1, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r49', 'priority': -42, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r49_BAG_L2: 	{'max_features': 'sqrt', 'max_leaf_nodes': 28532, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r49', 'priority': -43, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r143_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.9408897917880529, 'learning_rate': 0.01343464462043561, 'min_data_in_leaf': 21, 'num_leaves': 178, 'ag_args': {'name_suffix': '_r143', 'priority': -44, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r127_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 38572, 'min_samples_leaf': 5, 'ag_args': {'name_suffix': '_r127', 'priority': -45, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r134_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.006251885504130949, 'epochs': 47, 'layers': [800, 400], 'lr': 0.01329622020483052, 'ps': 0.2677080696008348, 'ag_args': {'name_suffix': '_r134', 'priority': -46, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r34_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 18242, 'min_samples_leaf': 40, 'ag_args': {'name_suffix': '_r34', 'priority': -47, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r94_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.4341088458599442, 'learning_rate': 0.04034449862560467, 'min_data_in_leaf': 33, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r94', 'priority': -48, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r143_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.1703783780377607, 'hidden_size': 212, 'learning_rate': 0.0004107199833213839, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.105439140660822e-07, 'ag_args': {'name_suffix': '_r143', 'priority': -49, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r128_BAG_L2: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.640921865280573, 'learning_rate': 0.036232951900213306, 'max_ctr_complexity': 3, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r128', 'priority': -50, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r111_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.6343202884164582, 'epochs': 21, 'layers': [400, 200], 'lr': 0.08479209380262258, 'ps': 0.48362560779595565, 'ag_args': {'name_suffix': '_r111', 'priority': -51, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r31_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.013288954106470907, 'hidden_size': 81, 'learning_rate': 0.005340914647396154, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 8.762168370775353e-05, 'ag_args': {'name_suffix': '_r31', 'priority': -52, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r4_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 19935, 'min_samples_leaf': 20, 'ag_args': {'name_suffix': '_r4', 'priority': -53, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r65_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.22771721361129746, 'epochs': 38, 'layers': [400], 'lr': 0.0005383511954451698, 'ps': 0.3734259772256502, 'ag_args': {'name_suffix': '_r65', 'priority': -54, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r88_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.4329361816589235, 'epochs': 50, 'layers': [400], 'lr': 0.09501311551121323, 'ps': 0.2863378667611431, 'ag_args': {'name_suffix': '_r88', 'priority': -55, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r30_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.9773131270704629, 'learning_rate': 0.010534290864227067, 'min_data_in_leaf': 21, 'num_leaves': 111, 'ag_args': {'name_suffix': '_r30', 'priority': -56, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r49_BAG_L2: 	{'colsample_bytree': 0.7452294043087835, 'enable_categorical': False, 'learning_rate': 0.038404229910104046, 'max_depth': 7, 'min_child_weight': 0.5564183327139662, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r49', 'priority': -57, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r5_BAG_L2: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.894432181094842, 'learning_rate': 0.055078095725390575, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r5', 'priority': -58, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r87_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.36669080773207274, 'hidden_size': 95, 'learning_rate': 0.015280159186761077, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.3082489374636015e-08, 'ag_args': {'name_suffix': '_r87', 'priority': -59, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r71_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.3027114570947557, 'hidden_size': 196, 'learning_rate': 0.006482759295309238, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 1.2806509958776e-12, 'ag_args': {'name_suffix': '_r71', 'priority': -60, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r143_BAG_L2: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6761016245166451, 'learning_rate': 0.06566144806528762, 'max_ctr_complexity': 2, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r143', 'priority': -61, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r178_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 29813, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r178', 'priority': -62, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r166_BAG_L2: 	{'max_features': 'log2', 'max_leaf_nodes': 42644, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r166', 'priority': -63, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r31_BAG_L2: 	{'colsample_bytree': 0.7506621909633511, 'enable_categorical': False, 'learning_rate': 0.009974712407899168, 'max_depth': 4, 'min_child_weight': 0.9238550485581797, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r31', 'priority': -64, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r185_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.12166942295569863, 'hidden_size': 151, 'learning_rate': 0.0018866871631794007, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 9.190843763153802e-05, 'ag_args': {'name_suffix': '_r185', 'priority': -65, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r160_BAG_L2: 	{'bs': 128, 'emb_drop': 0.3171659718142149, 'epochs': 20, 'layers': [400, 200, 100], 'lr': 0.03087210106068273, 'ps': 0.5909644730871169, 'ag_args': {'name_suffix': '_r160', 'priority': -66, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r60_BAG_L2: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3217885487525205, 'learning_rate': 0.05291587380674719, 'max_ctr_complexity': 5, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r60', 'priority': -67, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r15_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 36230, 'min_samples_leaf': 3, 'ag_args': {'name_suffix': '_r15', 'priority': -68, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r135_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.8254432681390782, 'learning_rate': 0.031251656439648626, 'min_data_in_leaf': 50, 'num_leaves': 210, 'ag_args': {'name_suffix': '_r135', 'priority': -69, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r22_BAG_L2: 	{'colsample_bytree': 0.6326947454697227, 'enable_categorical': False, 'learning_rate': 0.07792091886639502, 'max_depth': 6, 'min_child_weight': 1.0759464955561793, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r22', 'priority': -70, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r69_BAG_L2: 	{'bs': 128, 'emb_drop': 0.3209601865656554, 'epochs': 21, 'layers': [200, 100, 50], 'lr': 0.019935403046870463, 'ps': 0.19846319260751663, 'ag_args': {'name_suffix': '_r69', 'priority': -71, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r6_BAG_L2: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.5734131496361856, 'learning_rate': 0.08472519974533015, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r6', 'priority': -72, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r138_BAG_L2: 	{'bs': 128, 'emb_drop': 0.08669109226243704, 'epochs': 45, 'layers': [800, 400], 'lr': 0.0041554361714983635, 'ps': 0.2669780074016213, 'ag_args': {'name_suffix': '_r138', 'priority': -73, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r121_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.5730390983988963, 'learning_rate': 0.010305352949119608, 'min_data_in_leaf': 10, 'num_leaves': 215, 'ag_args': {'name_suffix': '_r121', 'priority': -74, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r172_BAG_L2: 	{'bs': 512, 'emb_drop': 0.05604276533830355, 'epochs': 32, 'layers': [400], 'lr': 0.027320709383189166, 'ps': 0.022591301744255762, 'ag_args': {'name_suffix': '_r172', 'priority': -75, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r180_BAG_L2: 	{'depth': 7, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 4.43335055453705, 'learning_rate': 0.055406199833457785, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r180', 'priority': -76, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r76_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.006531401073483156, 'hidden_size': 192, 'learning_rate': 0.012418052210914356, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 3.0406866089493607e-05, 'ag_args': {'name_suffix': '_r76', 'priority': -77, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r197_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 40459, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r197', 'priority': -78, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r121_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.33926015213879396, 'hidden_size': 247, 'learning_rate': 0.0029983839090226075, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 0.00038926240517691234, 'ag_args': {'name_suffix': '_r121', 'priority': -79, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r127_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.31956392388385874, 'epochs': 25, 'layers': [200, 100], 'lr': 0.08552736732040143, 'ps': 0.0934076022219228, 'ag_args': {'name_suffix': '_r127', 'priority': -80, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	RandomForest_r16_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 48136, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r16', 'priority': -81, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r194_BAG_L2: 	{'bs': 256, 'emb_drop': 0.5117456464220826, 'epochs': 21, 'layers': [400, 200, 100], 'lr': 0.007212882302137526, 'ps': 0.2747013981281539, 'ag_args': {'name_suffix': '_r194', 'priority': -82, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r12_BAG_L2: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.835797074498082, 'learning_rate': 0.03534026385152556, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r12', 'priority': -83, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r135_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.06134755114373829, 'hidden_size': 144, 'learning_rate': 0.005834535148903801, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 2.0826540090463355e-09, 'ag_args': {'name_suffix': '_r135', 'priority': -84, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r4_BAG_L2: 	{'bs': 256, 'emb_drop': 0.06099050979107849, 'epochs': 39, 'layers': [200], 'lr': 0.04119582873110387, 'ps': 0.5447097256648953, 'ag_args': {'name_suffix': '_r4', 'priority': -85, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	ExtraTrees_r126_BAG_L2: 	{'max_features': 'sqrt', 'max_leaf_nodes': 29702, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r126', 'priority': -86, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r36_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.3457125770744979, 'hidden_size': 37, 'learning_rate': 0.006435774191713849, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 2.4012185204155345e-08, 'ag_args': {'name_suffix': '_r36', 'priority': -87, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r100_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.6960805527533755, 'epochs': 38, 'layers': [800, 400], 'lr': 0.0007278526871749883, 'ps': 0.20495582200836318, 'ag_args': {'name_suffix': '_r100', 'priority': -88, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r163_BAG_L2: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.7454481983750014, 'learning_rate': 0.09328642499990342, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r163', 'priority': -89, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	CatBoost_r198_BAG_L2: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.637071465711953, 'learning_rate': 0.04387418552563314, 'max_ctr_complexity': 4, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r198', 'priority': -90, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetFastAI_r187_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.5074958658302495, 'epochs': 42, 'layers': [200, 100, 50], 'lr': 0.026342427824862867, 'ps': 0.34814978753283593, 'ag_args': {'name_suffix': '_r187', 'priority': -91, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r19_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.2211285919550286, 'hidden_size': 196, 'learning_rate': 0.011307978270179143, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 1.8441764217351068e-06, 'ag_args': {'name_suffix': '_r19', 'priority': -92, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r95_BAG_L2: 	{'colsample_bytree': 0.975937238416368, 'enable_categorical': False, 'learning_rate': 0.06634196266155237, 'max_depth': 5, 'min_child_weight': 1.4088437184127383, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r95', 'priority': -93, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	XGBoost_r34_BAG_L2: 	{'colsample_bytree': 0.546186944730449, 'enable_categorical': False, 'learning_rate': 0.029357102578825213, 'max_depth': 10, 'min_child_weight': 1.1532008198571337, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r34', 'priority': -94, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    [36m(_dystack pid=167306)[0m 	LightGBM_r42_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.4601361323873807, 'learning_rate': 0.07856777698860955, 'min_data_in_leaf': 12, 'num_leaves': 198, 'ag_args': {'name_suffix': '_r42', 'priority': -95, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r1_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.23713784729000734, 'hidden_size': 200, 'learning_rate': 0.00311256170909018, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 4.573016756474468e-08, 'ag_args': {'name_suffix': '_r1', 'priority': -96, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m 	NeuralNetTorch_r89_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.33567564890346097, 'hidden_size': 245, 'learning_rate': 0.006746560197328548, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.6470047305392933e-10, 'ag_args': {'name_suffix': '_r89', 'priority': -97, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    [36m(_dystack pid=167306)[0m Fitting 108 L2 models, fit_strategy="sequential" ...
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 273.85s of the 273.59s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting LightGBMXT_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=2.81%)
    [36m(_ray_fit pid=191094)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05, 'extra_trees': True}
    [36m(_ray_fit pid=191093)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L2/S1F6/model.pkl[32m [repeated 5x across cluster][0m
    [36m(_ray_fit pid=191098)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05, 'extra_trees': True}[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=191095)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L2/S1F5/model.pkl[32m [repeated 2x across cluster][0m


    [36m(_ray_fit pid=191099)[0m [50]	valid_set's binary_error: 0.221464
    [36m(_ray_fit pid=189223)[0m 0:	learn: 0.7773745	test: 0.7782723	best: 0.7782723 (0)	total: 4.76s	remaining: 13h 14m 12s[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=189223)[0m bestTest = 0.7782722702[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=189223)[0m bestIteration = 0[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=189223)[0m Shrink model to first 1 iterations.[32m [repeated 7x across cluster][0m


    [36m(_dystack pid=167306)[0m 	0.7785	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	27.96s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	1.41s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	39969.9	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: LightGBM_BAG_L2 ... Training model for up to 243.55s of the 243.29s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting LightGBM_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=2.74%)
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L2/utils/model_template.pkl[32m [repeated 9x across cluster][0m
    [36m(_ray_fit pid=192538)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05}
    [36m(_ray_fit pid=192544)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L2/S1F5/model.pkl
    [36m(_ray_fit pid=192543)[0m 	Fitting 10000 rounds... Hyperparameters: {'learning_rate': 0.05}[32m [repeated 7x across cluster][0m
    [36m(_ray_fit pid=192538)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L2/S1F1/model.pkl
    [36m(_dystack pid=167306)[0m 	0.7785	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	25.53s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	1.43s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	39920.7	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Fitting model: RandomForestGini_BAG_L2 ... Training model for up to 215.60s of the 215.34s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting RandomForestGini_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Warning: Reducing model 'n_estimators' from 300 -> 74 due to low time. Expected time usage reduced from 862.7s -> 215.1s...
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L2/utils/model_template.pkl[32m [repeated 10x across cluster][0m
    [36m(_dystack pid=167306)[0m 	178.07s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m 	Not enough time to generate out-of-fold predictions for model. Estimated time required was 178.07s compared to 82.08s of available time.
    [36m(_dystack pid=167306)[0m 	Time limit exceeded... Skipping RandomForestGini_BAG_L2.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Fitting model: RandomForestEntr_BAG_L2 ... Training model for up to 17.35s of the 17.09s of remaining time.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m 	Fitting RandomForestEntr_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m 	Warning: Model is expected to require 871.9s to train, which exceeds the maximum time limit of 16.8s, skipping model...
    [36m(_dystack pid=167306)[0m 	Time limit exceeded... Skipping RandomForestEntr_BAG_L2.
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L2/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTreesGini_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTreesEntr_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBMLarge_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r177_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r79_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r131_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r191_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r9_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r96_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r22_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r33_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r42_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r137_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r102_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r13_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r195_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r188_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r145_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r89_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r30_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r130_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r86_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r50_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r11_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r194_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r172_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r69_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r103_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r14_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r161_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r143_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r70_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r156_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r196_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r39_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r167_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r95_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r41_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r98_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r15_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r158_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r86_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r37_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r197_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r49_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r49_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r143_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r127_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r134_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r34_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r94_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r143_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r128_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r111_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r31_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r4_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r65_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r88_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r30_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r49_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r5_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r87_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r71_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r143_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r178_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r166_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r31_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r185_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r160_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r60_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r15_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r135_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r22_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r69_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r6_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r138_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r121_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r172_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r180_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r76_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r197_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r121_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r127_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping RandomForest_r16_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r194_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r12_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r135_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r4_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping ExtraTrees_r126_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r36_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r100_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r163_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping CatBoost_r198_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetFastAI_r187_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r19_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r95_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping XGBoost_r34_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping LightGBM_r42_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r1_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Skipping NeuralNetTorch_r89_BAG_L2 due to lack of time remaining.
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L2/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L2/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Model configs that will be trained (in order):
    [36m(_dystack pid=167306)[0m 	WeightedEnsemble_L3: 	{'ag_args': {'valid_base': False, 'name_bag_suffix': '', 'model_type': <class 'autogluon.core.models.greedy_ensemble.greedy_weighted_ensemble_model.GreedyWeightedEnsembleModel'>, 'priority': 0}, 'ag_args_ensemble': {'save_bag_folds': True}}
    [36m(_dystack pid=167306)[0m Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.00s of the 4.41s of remaining time.
    [36m(_dystack pid=167306)[0m 	Fitting WeightedEnsemble_L3 with 'num_gpus': 0, 'num_cpus': 32
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L3/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L3/utils/model_template.pkl
    [36m(_dystack pid=167306)[0m Subsampling to 1000000 samples to speedup ensemble selection...
    [36m(_dystack pid=167306)[0m Ensemble size: 2
    [36m(_dystack pid=167306)[0m Ensemble weights: 
    [36m(_dystack pid=167306)[0m [0.  0.  0.5 0.  0.5 0.  0.  0.  0.  0. ]
    [36m(_dystack pid=167306)[0m 	5.95s	= Estimated out-of-fold prediction time...
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L3/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L3/model.pkl
    [36m(_dystack pid=167306)[0m 	Ensemble Weights: {'RandomForestGini_BAG_L1': 0.5, 'CatBoost_BAG_L1': 0.5}
    [36m(_dystack pid=167306)[0m 	0.7785	 = Validation score   (accuracy)
    [36m(_dystack pid=167306)[0m 	7.44s	 = Training   runtime
    [36m(_dystack pid=167306)[0m 	0.24s	 = Validation runtime
    [36m(_dystack pid=167306)[0m 	106842.4	 = Inference  throughput (rows/s | 777229 batch size)
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m AutoGluon training complete, total runtime = 902.04s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 110460.6 rows/s (777229 batch size)
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Enabling decision threshold calibration (calibrate_decision_threshold='auto', metric is valid, problem_type is 'binary')
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/utils/data/y.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L2/utils/oof.pkl
    [36m(_dystack pid=167306)[0m Subsampling y to 1000000 samples to speedup threshold calibration...
    [36m(_dystack pid=167306)[0m Calibrating decision threshold to optimize metric accuracy | Checking 51 thresholds...
    [36m(_dystack pid=167306)[0m 	threshold: 0.500	| val: 0.7781	| NEW BEST
    [36m(_dystack pid=167306)[0m 	threshold: 0.480	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.520	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.460	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.540	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.440	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.560	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.420	| val: 0.7780
    [36m(_dystack pid=167306)[0m 	threshold: 0.580	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.400	| val: 0.7779
    [36m(_dystack pid=167306)[0m 	threshold: 0.600	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.380	| val: 0.7779
    [36m(_dystack pid=167306)[0m 	threshold: 0.620	| val: 0.7780
    [36m(_dystack pid=167306)[0m 	threshold: 0.360	| val: 0.7778
    [36m(_dystack pid=167306)[0m 	threshold: 0.640	| val: 0.7779
    [36m(_dystack pid=167306)[0m 	threshold: 0.340	| val: 0.2258
    [36m(_dystack pid=167306)[0m 	threshold: 0.660	| val: 0.7778
    [36m(_dystack pid=167306)[0m 	threshold: 0.320	| val: 0.2258
    [36m(_dystack pid=167306)[0m 	threshold: 0.680	| val: 0.7778
    [36m(_dystack pid=167306)[0m 	threshold: 0.300	| val: 0.2258
    [36m(_dystack pid=167306)[0m 	threshold: 0.700	| val: 0.7778
    [36m(_dystack pid=167306)[0m 	threshold: 0.280	| val: 0.2258
    [36m(_dystack pid=167306)[0m 	threshold: 0.720	| val: 0.7778
    [36m(_dystack pid=167306)[0m 	threshold: 0.260	| val: 0.2257
    [36m(_dystack pid=167306)[0m 	threshold: 0.740	| val: 0.7777
    [36m(_dystack pid=167306)[0m 	threshold: 0.240	| val: 0.2255
    [36m(_dystack pid=167306)[0m 	threshold: 0.760	| val: 0.7775
    [36m(_dystack pid=167306)[0m 	threshold: 0.220	| val: 0.2247
    [36m(_dystack pid=167306)[0m 	threshold: 0.780	| val: 0.7774
    [36m(_dystack pid=167306)[0m 	threshold: 0.200	| val: 0.2245
    [36m(_dystack pid=167306)[0m 	threshold: 0.800	| val: 0.7772
    [36m(_dystack pid=167306)[0m 	threshold: 0.180	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.820	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.160	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.840	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.140	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.860	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.120	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.880	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.100	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.900	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.080	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.920	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.060	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.940	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.040	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.960	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.020	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 0.980	| val: 0.7758
    [36m(_dystack pid=167306)[0m 	threshold: 0.000	| val: 0.2242
    [36m(_dystack pid=167306)[0m 	threshold: 1.000	| val: 0.7758
    [36m(_dystack pid=167306)[0m Calibrating decision threshold via fine-grained search | Checking 38 thresholds...
    [36m(_dystack pid=167306)[0m 	threshold: 0.501	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.502	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.503	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.504	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.505	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.506	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.507	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.508	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.509	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.510	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.511	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.512	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.513	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.514	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.515	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.516	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.517	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.518	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.519	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.499	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.498	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.497	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.496	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.495	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.494	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.493	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.492	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.491	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.490	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.489	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.488	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.487	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.486	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.485	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.484	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.483	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.482	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	threshold: 0.481	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	Base Threshold: 0.500	| val: 0.7781
    [36m(_dystack pid=167306)[0m 	Best Threshold: 0.500	| val: 0.7781
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/trainer.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/learner.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/predictor.pkl
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/version.txt with contents "1.2"
    [36m(_dystack pid=167306)[0m Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/metadata.json
    [36m(_dystack pid=167306)[0m TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho")
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestGini_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/RandomForestEntr_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/XGBoost_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMLarge_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/CatBoost_r177_BAG_L1/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L2/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBMXT_BAG_L2/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/LightGBM_BAG_L2/model.pkl
    [36m(_dystack pid=167306)[0m Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/ds_sub_fit/sub_fit_ho/models/WeightedEnsemble_L3/model.pkl
    [36m(_dystack pid=167306)[0m Deleting DyStack predictor artifacts (clean_up_fits=True) ...
    Leaderboard on holdout data (DyStack):
                          model  score_holdout  score_val eval_metric  pred_time_test  pred_time_val    fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      CatBoost_r177_BAG_L1       0.776398   0.778263    accuracy        0.267328       0.498833    7.369178                 0.267328                0.498833           7.369178            1       True          8
    1           CatBoost_BAG_L1       0.776398   0.778263    accuracy        0.460165       0.737438   76.239794                 0.460165                0.737438          76.239794            1       True          5
    2   RandomForestEntr_BAG_L1       0.776398   0.778520    accuracy        0.827243      51.614229   64.558109                 0.827243               51.614229          64.558109            1       True          4
    3   RandomForestGini_BAG_L1       0.776398   0.778520    accuracy        0.891119      52.052295   66.029967                 0.891119               52.052295          66.029967            1       True          3
    4       WeightedEnsemble_L2       0.776398   0.778521    accuracy        1.162330      52.798184   79.421196                 0.003883                0.247056           6.022051            2       True          9
    5       WeightedEnsemble_L3       0.776398   0.778521    accuracy        1.357468      53.034174  149.707164                 0.006184                0.244441           7.437403            3       True         12
    6           LightGBM_BAG_L2       0.776398   0.778474    accuracy        6.293372     110.177507  274.009581                 0.297113                1.433739          25.532861            2       True         11
    7         LightGBMXT_BAG_L2       0.776398   0.778470    accuracy        6.405225     110.153574  276.434564                 0.408966                1.409805          27.957844            2       True         10
    8           LightGBM_BAG_L1       0.776285   0.776285    accuracy        0.054016       0.251620    3.001216                 0.054016                0.251620           3.001216            1       True          2
    9      LightGBMLarge_BAG_L1       0.776285   0.776285    accuracy        0.058442       0.258679    3.419815                 0.058442                0.258679           3.419815            1       True          7
    10        LightGBMXT_BAG_L1       0.776285   0.776285    accuracy        0.094787       0.213144    2.962061                 0.094787                0.213144           2.962061            1       True          1
    11           XGBoost_BAG_L1       0.776285   0.776301    accuracy        3.343159       3.117530   24.896581                 3.343159                3.117530          24.896581            1       True          6
    	1	 = Optimal   num_stack_levels (Stacked Overfitting Occurred: False)
    	921s	 = DyStack   runtime |	2679s	 = Remaining runtime
    Starting main fit with num_stack_levels=1.
    	For future fit calls on this dataset, you can skip DyStack to save time: `predictor.fit(..., dynamic_stacking=False, num_stack_levels=1)`
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/learner.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/predictor.pkl
    Beginning AutoGluon training ... Time limit = 2679s
    AutoGluon will save models to "/home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248"
    Train Data Rows:    6995056
    Train Data Columns: 2
    Label Column:       label
    Problem Type:       binary
    Preprocessing data ...
    Selected class <--> label mapping:  class 1 = 1, class 0 = 0
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    66170.57 MB
    	Train Data (Original)  Memory Usage: 1009.98 MB (1.5% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Original Features (exact raw dtype, raw dtype):
    				('object', 'object') : 2 | ['ID', 'URL']
    			Types of features in original data (raw dtype, special dtypes):
    				('object', []) : 2 | ['ID', 'URL']
    			Types of features in processed data (raw dtype, special dtypes):
    				('object', []) : 2 | ['ID', 'URL']
    			1.1s = Fit runtime
    			2 features in original data used to generate 2 features in processed data.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    			Types of features in original data (raw dtype, special dtypes):
    				('object', []) : 2 | ['ID', 'URL']
    			Types of features in processed data (raw dtype, special dtypes):
    				('object', []) : 2 | ['ID', 'URL']
    			1.2s = Fit runtime
    			2 features in original data used to generate 2 features in processed data.
    	Stage 3 Generators:
    		Skipping IdentityFeatureGenerator: No input feature with required dtypes.
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    				Types of features in original data (raw dtype, special dtypes):
    					('category', []) : 2 | ['ID', 'URL']
    				Types of features in processed data (raw dtype, special dtypes):
    					('category', []) : 2 | ['ID', 'URL']
    				0.4s = Fit runtime
    				2 features in original data used to generate 2 features in processed data.
    			Types of features in original data (raw dtype, special dtypes):
    				('object', []) : 2 | ['ID', 'URL']
    			Types of features in processed data (raw dtype, special dtypes):
    				('category', []) : 2 | ['ID', 'URL']
    			22.8s = Fit runtime
    			2 features in original data used to generate 2 features in processed data.
    		Skipping DatetimeFeatureGenerator: No input feature with required dtypes.
    		Skipping TextSpecialFeatureGenerator: No input feature with required dtypes.
    		Skipping TextNgramFeatureGenerator: No input feature with required dtypes.
    		Skipping IdentityFeatureGenerator: No input feature with required dtypes.
    		Skipping IsNanFeatureGenerator: No input feature with required dtypes.
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    			Types of features in original data (raw dtype, special dtypes):
    				('category', []) : 1 | ['URL']
    			Types of features in processed data (raw dtype, special dtypes):
    				('category', []) : 1 | ['URL']
    			0.4s = Fit runtime
    			1 features in original data used to generate 1 features in processed data.
    	Stage 5 Generators:
    		Fitting DropDuplicatesFeatureGenerator...
    			Types of features in original data (raw dtype, special dtypes):
    				('category', []) : 1 | ['URL']
    			Types of features in processed data (raw dtype, special dtypes):
    				('category', []) : 1 | ['URL']
    			0.4s = Fit runtime
    			1 features in original data used to generate 1 features in processed data.
    	Unused Original Features (Count: 1): ['ID']
    		These features were not used to generate any of the output features. Add a feature generator compatible with these features to utilize them.
    		Features can also be unused if they carry very little information, such as being categorical but having almost entirely unique values or being duplicates of other features.
    		These features do not need to be present at inference time.
    		('object', []) : 1 | ['ID']
    	Types of features in original data (exact raw dtype, raw dtype):
    		('object', 'object') : 1 | ['URL']
    	Types of features in original data (raw dtype, special dtypes):
    		('object', []) : 1 | ['URL']
    	Types of features in processed data (exact raw dtype, raw dtype):
    		('category', 'category') : 1 | ['URL']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', []) : 1 | ['URL']
    	33.0s = Fit runtime
    	1 features in original data used to generate 1 features in processed data.
    	Train Data (Processed) Memory Usage: 13.34 MB (0.0% of available memory)
    Data preprocessing and feature engineering runtime = 34.07s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'accuracy'
    	To change this, specify the eval_metric parameter of Predictor()
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/learner.pkl
    User-specified model hyperparameters to be fit:
    {
    	'NN_TORCH': [{}, {'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2}}, {'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7}}, {'activation': 'elu', 'dropout_prob': 0.24622382571353768, 'hidden_size': 159, 'learning_rate': 0.008507536855608535, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.8201539594953562e-06, 'ag_args': {'name_suffix': '_r30', 'priority': -17}}, {'activation': 'relu', 'dropout_prob': 0.09976801642258049, 'hidden_size': 135, 'learning_rate': 0.001631450730978947, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 3.867683394425807e-05, 'ag_args': {'name_suffix': '_r86', 'priority': -19}}, {'activation': 'relu', 'dropout_prob': 0.3905837860053583, 'hidden_size': 106, 'learning_rate': 0.0018297905295930797, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 9.178069874232892e-08, 'ag_args': {'name_suffix': '_r14', 'priority': -26}}, {'activation': 'relu', 'dropout_prob': 0.05488816803887784, 'hidden_size': 32, 'learning_rate': 0.0075612897834015985, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.652353009917866e-08, 'ag_args': {'name_suffix': '_r41', 'priority': -35}}, {'activation': 'elu', 'dropout_prob': 0.01030258381183309, 'hidden_size': 111, 'learning_rate': 0.01845979186513771, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 0.00020238017476912164, 'ag_args': {'name_suffix': '_r158', 'priority': -38}}, {'activation': 'elu', 'dropout_prob': 0.18109219857068798, 'hidden_size': 250, 'learning_rate': 0.00634181748507711, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 5.3861175580695396e-08, 'ag_args': {'name_suffix': '_r197', 'priority': -41}}, {'activation': 'elu', 'dropout_prob': 0.1703783780377607, 'hidden_size': 212, 'learning_rate': 0.0004107199833213839, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.105439140660822e-07, 'ag_args': {'name_suffix': '_r143', 'priority': -49}}, {'activation': 'elu', 'dropout_prob': 0.013288954106470907, 'hidden_size': 81, 'learning_rate': 0.005340914647396154, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 8.762168370775353e-05, 'ag_args': {'name_suffix': '_r31', 'priority': -52}}, {'activation': 'relu', 'dropout_prob': 0.36669080773207274, 'hidden_size': 95, 'learning_rate': 0.015280159186761077, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.3082489374636015e-08, 'ag_args': {'name_suffix': '_r87', 'priority': -59}}, {'activation': 'relu', 'dropout_prob': 0.3027114570947557, 'hidden_size': 196, 'learning_rate': 0.006482759295309238, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 1.2806509958776e-12, 'ag_args': {'name_suffix': '_r71', 'priority': -60}}, {'activation': 'relu', 'dropout_prob': 0.12166942295569863, 'hidden_size': 151, 'learning_rate': 0.0018866871631794007, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 9.190843763153802e-05, 'ag_args': {'name_suffix': '_r185', 'priority': -65}}, {'activation': 'relu', 'dropout_prob': 0.006531401073483156, 'hidden_size': 192, 'learning_rate': 0.012418052210914356, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 3.0406866089493607e-05, 'ag_args': {'name_suffix': '_r76', 'priority': -77}}, {'activation': 'relu', 'dropout_prob': 0.33926015213879396, 'hidden_size': 247, 'learning_rate': 0.0029983839090226075, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 0.00038926240517691234, 'ag_args': {'name_suffix': '_r121', 'priority': -79}}, {'activation': 'elu', 'dropout_prob': 0.06134755114373829, 'hidden_size': 144, 'learning_rate': 0.005834535148903801, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 2.0826540090463355e-09, 'ag_args': {'name_suffix': '_r135', 'priority': -84}}, {'activation': 'elu', 'dropout_prob': 0.3457125770744979, 'hidden_size': 37, 'learning_rate': 0.006435774191713849, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 2.4012185204155345e-08, 'ag_args': {'name_suffix': '_r36', 'priority': -87}}, {'activation': 'relu', 'dropout_prob': 0.2211285919550286, 'hidden_size': 196, 'learning_rate': 0.011307978270179143, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 1.8441764217351068e-06, 'ag_args': {'name_suffix': '_r19', 'priority': -92}}, {'activation': 'relu', 'dropout_prob': 0.23713784729000734, 'hidden_size': 200, 'learning_rate': 0.00311256170909018, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 4.573016756474468e-08, 'ag_args': {'name_suffix': '_r1', 'priority': -96}}, {'activation': 'relu', 'dropout_prob': 0.33567564890346097, 'hidden_size': 245, 'learning_rate': 0.006746560197328548, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.6470047305392933e-10, 'ag_args': {'name_suffix': '_r89', 'priority': -97}}],
    	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}, {'extra_trees': False, 'feature_fraction': 0.7023601671276614, 'learning_rate': 0.012144796373999013, 'min_data_in_leaf': 14, 'num_leaves': 53, 'ag_args': {'name_suffix': '_r131', 'priority': -3}}, {'extra_trees': True, 'feature_fraction': 0.5636931414546802, 'learning_rate': 0.01518660230385841, 'min_data_in_leaf': 48, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r96', 'priority': -6}}, {'extra_trees': True, 'feature_fraction': 0.8282601210460099, 'learning_rate': 0.033929021353492905, 'min_data_in_leaf': 6, 'num_leaves': 127, 'ag_args': {'name_suffix': '_r188', 'priority': -14}}, {'extra_trees': False, 'feature_fraction': 0.6245777099925497, 'learning_rate': 0.04711573688184715, 'min_data_in_leaf': 56, 'num_leaves': 89, 'ag_args': {'name_suffix': '_r130', 'priority': -18}}, {'extra_trees': False, 'feature_fraction': 0.5898927512279213, 'learning_rate': 0.010464516487486093, 'min_data_in_leaf': 11, 'num_leaves': 252, 'ag_args': {'name_suffix': '_r161', 'priority': -27}}, {'extra_trees': True, 'feature_fraction': 0.5143401489640409, 'learning_rate': 0.00529479887023554, 'min_data_in_leaf': 6, 'num_leaves': 133, 'ag_args': {'name_suffix': '_r196', 'priority': -31}}, {'extra_trees': False, 'feature_fraction': 0.7421180622507277, 'learning_rate': 0.018603888565740096, 'min_data_in_leaf': 6, 'num_leaves': 22, 'ag_args': {'name_suffix': '_r15', 'priority': -37}}, {'extra_trees': False, 'feature_fraction': 0.9408897917880529, 'learning_rate': 0.01343464462043561, 'min_data_in_leaf': 21, 'num_leaves': 178, 'ag_args': {'name_suffix': '_r143', 'priority': -44}}, {'extra_trees': True, 'feature_fraction': 0.4341088458599442, 'learning_rate': 0.04034449862560467, 'min_data_in_leaf': 33, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r94', 'priority': -48}}, {'extra_trees': True, 'feature_fraction': 0.9773131270704629, 'learning_rate': 0.010534290864227067, 'min_data_in_leaf': 21, 'num_leaves': 111, 'ag_args': {'name_suffix': '_r30', 'priority': -56}}, {'extra_trees': False, 'feature_fraction': 0.8254432681390782, 'learning_rate': 0.031251656439648626, 'min_data_in_leaf': 50, 'num_leaves': 210, 'ag_args': {'name_suffix': '_r135', 'priority': -69}}, {'extra_trees': False, 'feature_fraction': 0.5730390983988963, 'learning_rate': 0.010305352949119608, 'min_data_in_leaf': 10, 'num_leaves': 215, 'ag_args': {'name_suffix': '_r121', 'priority': -74}}, {'extra_trees': True, 'feature_fraction': 0.4601361323873807, 'learning_rate': 0.07856777698860955, 'min_data_in_leaf': 12, 'num_leaves': 198, 'ag_args': {'name_suffix': '_r42', 'priority': -95}}],
    	'CAT': [{}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5}}, {'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.559174625782161, 'learning_rate': 0.04939557741379516, 'max_ctr_complexity': 3, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r137', 'priority': -10}}, {'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3274013177541373, 'learning_rate': 0.017301189655111057, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r13', 'priority': -12}}, {'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7018061518087038, 'learning_rate': 0.07092851311746352, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r50', 'priority': -20}}, {'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.0457098345001241, 'learning_rate': 0.050294288910022224, 'max_ctr_complexity': 5, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r69', 'priority': -24}}, {'depth': 6, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.3584121369544215, 'learning_rate': 0.03743901034980473, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r70', 'priority': -29}}, {'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.522712492188319, 'learning_rate': 0.08481607830570326, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r167', 'priority': -33}}, {'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6376578537958237, 'learning_rate': 0.032899230324940465, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r86', 'priority': -39}}, {'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.353268454214423, 'learning_rate': 0.06028218319511302, 'max_ctr_complexity': 1, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r49', 'priority': -42}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.640921865280573, 'learning_rate': 0.036232951900213306, 'max_ctr_complexity': 3, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r128', 'priority': -50}}, {'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.894432181094842, 'learning_rate': 0.055078095725390575, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r5', 'priority': -58}}, {'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6761016245166451, 'learning_rate': 0.06566144806528762, 'max_ctr_complexity': 2, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r143', 'priority': -61}}, {'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3217885487525205, 'learning_rate': 0.05291587380674719, 'max_ctr_complexity': 5, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r60', 'priority': -67}}, {'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.5734131496361856, 'learning_rate': 0.08472519974533015, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r6', 'priority': -72}}, {'depth': 7, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 4.43335055453705, 'learning_rate': 0.055406199833457785, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r180', 'priority': -76}}, {'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.835797074498082, 'learning_rate': 0.03534026385152556, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r12', 'priority': -83}}, {'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.7454481983750014, 'learning_rate': 0.09328642499990342, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r163', 'priority': -89}}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.637071465711953, 'learning_rate': 0.04387418552563314, 'max_ctr_complexity': 4, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r198', 'priority': -90}}],
    	'XGB': [{}, {'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'name_suffix': '_r33', 'priority': -8}}, {'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'name_suffix': '_r89', 'priority': -16}}, {'colsample_bytree': 0.9090166528779192, 'enable_categorical': True, 'learning_rate': 0.09290221350439203, 'max_depth': 7, 'min_child_weight': 0.8041986915994078, 'ag_args': {'name_suffix': '_r194', 'priority': -22}}, {'colsample_bytree': 0.516652313273348, 'enable_categorical': True, 'learning_rate': 0.007158072983547058, 'max_depth': 9, 'min_child_weight': 0.8567068904025429, 'ag_args': {'name_suffix': '_r98', 'priority': -36}}, {'colsample_bytree': 0.7452294043087835, 'enable_categorical': False, 'learning_rate': 0.038404229910104046, 'max_depth': 7, 'min_child_weight': 0.5564183327139662, 'ag_args': {'name_suffix': '_r49', 'priority': -57}}, {'colsample_bytree': 0.7506621909633511, 'enable_categorical': False, 'learning_rate': 0.009974712407899168, 'max_depth': 4, 'min_child_weight': 0.9238550485581797, 'ag_args': {'name_suffix': '_r31', 'priority': -64}}, {'colsample_bytree': 0.6326947454697227, 'enable_categorical': False, 'learning_rate': 0.07792091886639502, 'max_depth': 6, 'min_child_weight': 1.0759464955561793, 'ag_args': {'name_suffix': '_r22', 'priority': -70}}, {'colsample_bytree': 0.975937238416368, 'enable_categorical': False, 'learning_rate': 0.06634196266155237, 'max_depth': 5, 'min_child_weight': 1.4088437184127383, 'ag_args': {'name_suffix': '_r95', 'priority': -93}}, {'colsample_bytree': 0.546186944730449, 'enable_categorical': False, 'learning_rate': 0.029357102578825213, 'max_depth': 10, 'min_child_weight': 1.1532008198571337, 'ag_args': {'name_suffix': '_r34', 'priority': -94}}],
    	'FASTAI': [{}, {'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4}}, {'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11}}, {'bs': 128, 'emb_drop': 0.44339037504795686, 'epochs': 31, 'layers': [400, 200, 100], 'lr': 0.008615195908919904, 'ps': 0.19220253419114286, 'ag_args': {'name_suffix': '_r145', 'priority': -15}}, {'bs': 128, 'emb_drop': 0.026897798530914306, 'epochs': 31, 'layers': [800, 400], 'lr': 0.08045277634470181, 'ps': 0.4569532219038436, 'ag_args': {'name_suffix': '_r11', 'priority': -21}}, {'bs': 256, 'emb_drop': 0.1508701680951814, 'epochs': 46, 'layers': [400, 200], 'lr': 0.08794353125787312, 'ps': 0.19110623090573325, 'ag_args': {'name_suffix': '_r103', 'priority': -25}}, {'bs': 1024, 'emb_drop': 0.6239200452002372, 'epochs': 39, 'layers': [200, 100, 50], 'lr': 0.07170321592506483, 'ps': 0.670815151683455, 'ag_args': {'name_suffix': '_r143', 'priority': -28}}, {'bs': 2048, 'emb_drop': 0.5055288166864152, 'epochs': 44, 'layers': [400], 'lr': 0.0047762208542912405, 'ps': 0.06572612802222005, 'ag_args': {'name_suffix': '_r156', 'priority': -30}}, {'bs': 128, 'emb_drop': 0.6656668277387758, 'epochs': 32, 'layers': [400, 200, 100], 'lr': 0.019326244622675428, 'ps': 0.04084945128641206, 'ag_args': {'name_suffix': '_r95', 'priority': -34}}, {'bs': 512, 'emb_drop': 0.1567472816422661, 'epochs': 41, 'layers': [400, 200, 100], 'lr': 0.06831450078222204, 'ps': 0.4930900813464729, 'ag_args': {'name_suffix': '_r37', 'priority': -40}}, {'bs': 2048, 'emb_drop': 0.006251885504130949, 'epochs': 47, 'layers': [800, 400], 'lr': 0.01329622020483052, 'ps': 0.2677080696008348, 'ag_args': {'name_suffix': '_r134', 'priority': -46}}, {'bs': 2048, 'emb_drop': 0.6343202884164582, 'epochs': 21, 'layers': [400, 200], 'lr': 0.08479209380262258, 'ps': 0.48362560779595565, 'ag_args': {'name_suffix': '_r111', 'priority': -51}}, {'bs': 1024, 'emb_drop': 0.22771721361129746, 'epochs': 38, 'layers': [400], 'lr': 0.0005383511954451698, 'ps': 0.3734259772256502, 'ag_args': {'name_suffix': '_r65', 'priority': -54}}, {'bs': 1024, 'emb_drop': 0.4329361816589235, 'epochs': 50, 'layers': [400], 'lr': 0.09501311551121323, 'ps': 0.2863378667611431, 'ag_args': {'name_suffix': '_r88', 'priority': -55}}, {'bs': 128, 'emb_drop': 0.3171659718142149, 'epochs': 20, 'layers': [400, 200, 100], 'lr': 0.03087210106068273, 'ps': 0.5909644730871169, 'ag_args': {'name_suffix': '_r160', 'priority': -66}}, {'bs': 128, 'emb_drop': 0.3209601865656554, 'epochs': 21, 'layers': [200, 100, 50], 'lr': 0.019935403046870463, 'ps': 0.19846319260751663, 'ag_args': {'name_suffix': '_r69', 'priority': -71}}, {'bs': 128, 'emb_drop': 0.08669109226243704, 'epochs': 45, 'layers': [800, 400], 'lr': 0.0041554361714983635, 'ps': 0.2669780074016213, 'ag_args': {'name_suffix': '_r138', 'priority': -73}}, {'bs': 512, 'emb_drop': 0.05604276533830355, 'epochs': 32, 'layers': [400], 'lr': 0.027320709383189166, 'ps': 0.022591301744255762, 'ag_args': {'name_suffix': '_r172', 'priority': -75}}, {'bs': 1024, 'emb_drop': 0.31956392388385874, 'epochs': 25, 'layers': [200, 100], 'lr': 0.08552736732040143, 'ps': 0.0934076022219228, 'ag_args': {'name_suffix': '_r127', 'priority': -80}}, {'bs': 256, 'emb_drop': 0.5117456464220826, 'epochs': 21, 'layers': [400, 200, 100], 'lr': 0.007212882302137526, 'ps': 0.2747013981281539, 'ag_args': {'name_suffix': '_r194', 'priority': -82}}, {'bs': 256, 'emb_drop': 0.06099050979107849, 'epochs': 39, 'layers': [200], 'lr': 0.04119582873110387, 'ps': 0.5447097256648953, 'ag_args': {'name_suffix': '_r4', 'priority': -85}}, {'bs': 2048, 'emb_drop': 0.6960805527533755, 'epochs': 38, 'layers': [800, 400], 'lr': 0.0007278526871749883, 'ps': 0.20495582200836318, 'ag_args': {'name_suffix': '_r100', 'priority': -88}}, {'bs': 1024, 'emb_drop': 0.5074958658302495, 'epochs': 42, 'layers': [200, 100, 50], 'lr': 0.026342427824862867, 'ps': 0.34814978753283593, 'ag_args': {'name_suffix': '_r187', 'priority': -91}}],
    	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}, {'max_features': 0.75, 'max_leaf_nodes': 37308, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r195', 'priority': -13}}, {'max_features': 0.75, 'max_leaf_nodes': 28310, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r39', 'priority': -32}}, {'max_features': 1.0, 'max_leaf_nodes': 38572, 'min_samples_leaf': 5, 'ag_args': {'name_suffix': '_r127', 'priority': -45}}, {'max_features': 0.75, 'max_leaf_nodes': 18242, 'min_samples_leaf': 40, 'ag_args': {'name_suffix': '_r34', 'priority': -47}}, {'max_features': 'log2', 'max_leaf_nodes': 42644, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r166', 'priority': -63}}, {'max_features': 0.75, 'max_leaf_nodes': 36230, 'min_samples_leaf': 3, 'ag_args': {'name_suffix': '_r15', 'priority': -68}}, {'max_features': 1.0, 'max_leaf_nodes': 48136, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r16', 'priority': -81}}],
    	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}, {'max_features': 0.75, 'max_leaf_nodes': 18392, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r42', 'priority': -9}}, {'max_features': 1.0, 'max_leaf_nodes': 12845, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r172', 'priority': -23}}, {'max_features': 'sqrt', 'max_leaf_nodes': 28532, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r49', 'priority': -43}}, {'max_features': 1.0, 'max_leaf_nodes': 19935, 'min_samples_leaf': 20, 'ag_args': {'name_suffix': '_r4', 'priority': -53}}, {'max_features': 0.75, 'max_leaf_nodes': 29813, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r178', 'priority': -62}}, {'max_features': 1.0, 'max_leaf_nodes': 40459, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r197', 'priority': -78}}, {'max_features': 'sqrt', 'max_leaf_nodes': 29702, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r126', 'priority': -86}}],
    	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
    }
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/utils/data/X.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/utils/data/y.pkl
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Model configs that will be trained (in order):
    	KNeighborsUnif_BAG_L1: 	{'weights': 'uniform', 'ag_args': {'valid_stacker': False, 'problem_types': ['binary', 'multiclass', 'regression'], 'name_suffix': 'Unif', 'model_type': <class 'autogluon.tabular.models.knn.knn_model.KNNModel'>, 'priority': 100}, 'ag_args_ensemble': {'use_child_oof': True}}
    	KNeighborsDist_BAG_L1: 	{'weights': 'distance', 'ag_args': {'valid_stacker': False, 'problem_types': ['binary', 'multiclass', 'regression'], 'name_suffix': 'Dist', 'model_type': <class 'autogluon.tabular.models.knn.knn_model.KNNModel'>, 'priority': 100}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBMXT_BAG_L1: 	{'extra_trees': True, 'ag_args': {'name_suffix': 'XT', 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    	LightGBM_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    	RandomForestGini_BAG_L1: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    	RandomForestEntr_BAG_L1: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>, 'priority': 70}}
    	ExtraTreesGini_BAG_L1: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    	ExtraTreesEntr_BAG_L1: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>, 'priority': 50}}
    	XGBoost_BAG_L1: 	{'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>, 'priority': 40}}
    	NeuralNetTorch_BAG_L1: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>, 'priority': 25}}
    	LightGBMLarge_BAG_L1: 	{'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	CatBoost_r177_BAG_L1: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r79_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	LightGBM_r131_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.7023601671276614, 'learning_rate': 0.012144796373999013, 'min_data_in_leaf': 14, 'num_leaves': 53, 'ag_args': {'name_suffix': '_r131', 'priority': -3, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r191_BAG_L1: 	{'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r9_BAG_L1: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	LightGBM_r96_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.5636931414546802, 'learning_rate': 0.01518660230385841, 'min_data_in_leaf': 48, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r96', 'priority': -6, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r22_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	XGBoost_r33_BAG_L1: 	{'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r33', 'priority': -8, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	ExtraTrees_r42_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 18392, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r42', 'priority': -9, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_r137_BAG_L1: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.559174625782161, 'learning_rate': 0.04939557741379516, 'max_ctr_complexity': 3, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r137', 'priority': -10, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r102_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r13_BAG_L1: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3274013177541373, 'learning_rate': 0.017301189655111057, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r13', 'priority': -12, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	RandomForest_r195_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 37308, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r195', 'priority': -13, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r188_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.8282601210460099, 'learning_rate': 0.033929021353492905, 'min_data_in_leaf': 6, 'num_leaves': 127, 'ag_args': {'name_suffix': '_r188', 'priority': -14, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r145_BAG_L1: 	{'bs': 128, 'emb_drop': 0.44339037504795686, 'epochs': 31, 'layers': [400, 200, 100], 'lr': 0.008615195908919904, 'ps': 0.19220253419114286, 'ag_args': {'name_suffix': '_r145', 'priority': -15, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	XGBoost_r89_BAG_L1: 	{'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r89', 'priority': -16, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	NeuralNetTorch_r30_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.24622382571353768, 'hidden_size': 159, 'learning_rate': 0.008507536855608535, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.8201539594953562e-06, 'ag_args': {'name_suffix': '_r30', 'priority': -17, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	LightGBM_r130_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.6245777099925497, 'learning_rate': 0.04711573688184715, 'min_data_in_leaf': 56, 'num_leaves': 89, 'ag_args': {'name_suffix': '_r130', 'priority': -18, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r86_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.09976801642258049, 'hidden_size': 135, 'learning_rate': 0.001631450730978947, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 3.867683394425807e-05, 'ag_args': {'name_suffix': '_r86', 'priority': -19, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r50_BAG_L1: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7018061518087038, 'learning_rate': 0.07092851311746352, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r50', 'priority': -20, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r11_BAG_L1: 	{'bs': 128, 'emb_drop': 0.026897798530914306, 'epochs': 31, 'layers': [800, 400], 'lr': 0.08045277634470181, 'ps': 0.4569532219038436, 'ag_args': {'name_suffix': '_r11', 'priority': -21, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	XGBoost_r194_BAG_L1: 	{'colsample_bytree': 0.9090166528779192, 'enable_categorical': True, 'learning_rate': 0.09290221350439203, 'max_depth': 7, 'min_child_weight': 0.8041986915994078, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r194', 'priority': -22, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	ExtraTrees_r172_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 12845, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r172', 'priority': -23, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_r69_BAG_L1: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.0457098345001241, 'learning_rate': 0.050294288910022224, 'max_ctr_complexity': 5, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r69', 'priority': -24, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r103_BAG_L1: 	{'bs': 256, 'emb_drop': 0.1508701680951814, 'epochs': 46, 'layers': [400, 200], 'lr': 0.08794353125787312, 'ps': 0.19110623090573325, 'ag_args': {'name_suffix': '_r103', 'priority': -25, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r14_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.3905837860053583, 'hidden_size': 106, 'learning_rate': 0.0018297905295930797, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 9.178069874232892e-08, 'ag_args': {'name_suffix': '_r14', 'priority': -26, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	LightGBM_r161_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.5898927512279213, 'learning_rate': 0.010464516487486093, 'min_data_in_leaf': 11, 'num_leaves': 252, 'ag_args': {'name_suffix': '_r161', 'priority': -27, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r143_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.6239200452002372, 'epochs': 39, 'layers': [200, 100, 50], 'lr': 0.07170321592506483, 'ps': 0.670815151683455, 'ag_args': {'name_suffix': '_r143', 'priority': -28, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r70_BAG_L1: 	{'depth': 6, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.3584121369544215, 'learning_rate': 0.03743901034980473, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r70', 'priority': -29, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r156_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.5055288166864152, 'epochs': 44, 'layers': [400], 'lr': 0.0047762208542912405, 'ps': 0.06572612802222005, 'ag_args': {'name_suffix': '_r156', 'priority': -30, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	LightGBM_r196_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.5143401489640409, 'learning_rate': 0.00529479887023554, 'min_data_in_leaf': 6, 'num_leaves': 133, 'ag_args': {'name_suffix': '_r196', 'priority': -31, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	RandomForest_r39_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 28310, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r39', 'priority': -32, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_r167_BAG_L1: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.522712492188319, 'learning_rate': 0.08481607830570326, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r167', 'priority': -33, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r95_BAG_L1: 	{'bs': 128, 'emb_drop': 0.6656668277387758, 'epochs': 32, 'layers': [400, 200, 100], 'lr': 0.019326244622675428, 'ps': 0.04084945128641206, 'ag_args': {'name_suffix': '_r95', 'priority': -34, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r41_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.05488816803887784, 'hidden_size': 32, 'learning_rate': 0.0075612897834015985, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.652353009917866e-08, 'ag_args': {'name_suffix': '_r41', 'priority': -35, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	XGBoost_r98_BAG_L1: 	{'colsample_bytree': 0.516652313273348, 'enable_categorical': True, 'learning_rate': 0.007158072983547058, 'max_depth': 9, 'min_child_weight': 0.8567068904025429, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r98', 'priority': -36, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	LightGBM_r15_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.7421180622507277, 'learning_rate': 0.018603888565740096, 'min_data_in_leaf': 6, 'num_leaves': 22, 'ag_args': {'name_suffix': '_r15', 'priority': -37, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r158_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.01030258381183309, 'hidden_size': 111, 'learning_rate': 0.01845979186513771, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 0.00020238017476912164, 'ag_args': {'name_suffix': '_r158', 'priority': -38, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r86_BAG_L1: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6376578537958237, 'learning_rate': 0.032899230324940465, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r86', 'priority': -39, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r37_BAG_L1: 	{'bs': 512, 'emb_drop': 0.1567472816422661, 'epochs': 41, 'layers': [400, 200, 100], 'lr': 0.06831450078222204, 'ps': 0.4930900813464729, 'ag_args': {'name_suffix': '_r37', 'priority': -40, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r197_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.18109219857068798, 'hidden_size': 250, 'learning_rate': 0.00634181748507711, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 5.3861175580695396e-08, 'ag_args': {'name_suffix': '_r197', 'priority': -41, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r49_BAG_L1: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.353268454214423, 'learning_rate': 0.06028218319511302, 'max_ctr_complexity': 1, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r49', 'priority': -42, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	ExtraTrees_r49_BAG_L1: 	{'max_features': 'sqrt', 'max_leaf_nodes': 28532, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r49', 'priority': -43, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r143_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.9408897917880529, 'learning_rate': 0.01343464462043561, 'min_data_in_leaf': 21, 'num_leaves': 178, 'ag_args': {'name_suffix': '_r143', 'priority': -44, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	RandomForest_r127_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 38572, 'min_samples_leaf': 5, 'ag_args': {'name_suffix': '_r127', 'priority': -45, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_r134_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.006251885504130949, 'epochs': 47, 'layers': [800, 400], 'lr': 0.01329622020483052, 'ps': 0.2677080696008348, 'ag_args': {'name_suffix': '_r134', 'priority': -46, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	RandomForest_r34_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 18242, 'min_samples_leaf': 40, 'ag_args': {'name_suffix': '_r34', 'priority': -47, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r94_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.4341088458599442, 'learning_rate': 0.04034449862560467, 'min_data_in_leaf': 33, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r94', 'priority': -48, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r143_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.1703783780377607, 'hidden_size': 212, 'learning_rate': 0.0004107199833213839, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.105439140660822e-07, 'ag_args': {'name_suffix': '_r143', 'priority': -49, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r128_BAG_L1: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.640921865280573, 'learning_rate': 0.036232951900213306, 'max_ctr_complexity': 3, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r128', 'priority': -50, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r111_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.6343202884164582, 'epochs': 21, 'layers': [400, 200], 'lr': 0.08479209380262258, 'ps': 0.48362560779595565, 'ag_args': {'name_suffix': '_r111', 'priority': -51, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r31_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.013288954106470907, 'hidden_size': 81, 'learning_rate': 0.005340914647396154, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 8.762168370775353e-05, 'ag_args': {'name_suffix': '_r31', 'priority': -52, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	ExtraTrees_r4_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 19935, 'min_samples_leaf': 20, 'ag_args': {'name_suffix': '_r4', 'priority': -53, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_r65_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.22771721361129746, 'epochs': 38, 'layers': [400], 'lr': 0.0005383511954451698, 'ps': 0.3734259772256502, 'ag_args': {'name_suffix': '_r65', 'priority': -54, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetFastAI_r88_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.4329361816589235, 'epochs': 50, 'layers': [400], 'lr': 0.09501311551121323, 'ps': 0.2863378667611431, 'ag_args': {'name_suffix': '_r88', 'priority': -55, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	LightGBM_r30_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.9773131270704629, 'learning_rate': 0.010534290864227067, 'min_data_in_leaf': 21, 'num_leaves': 111, 'ag_args': {'name_suffix': '_r30', 'priority': -56, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	XGBoost_r49_BAG_L1: 	{'colsample_bytree': 0.7452294043087835, 'enable_categorical': False, 'learning_rate': 0.038404229910104046, 'max_depth': 7, 'min_child_weight': 0.5564183327139662, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r49', 'priority': -57, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	CatBoost_r5_BAG_L1: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.894432181094842, 'learning_rate': 0.055078095725390575, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r5', 'priority': -58, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r87_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.36669080773207274, 'hidden_size': 95, 'learning_rate': 0.015280159186761077, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.3082489374636015e-08, 'ag_args': {'name_suffix': '_r87', 'priority': -59, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetTorch_r71_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.3027114570947557, 'hidden_size': 196, 'learning_rate': 0.006482759295309238, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 1.2806509958776e-12, 'ag_args': {'name_suffix': '_r71', 'priority': -60, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r143_BAG_L1: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6761016245166451, 'learning_rate': 0.06566144806528762, 'max_ctr_complexity': 2, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r143', 'priority': -61, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	ExtraTrees_r178_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 29813, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r178', 'priority': -62, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	RandomForest_r166_BAG_L1: 	{'max_features': 'log2', 'max_leaf_nodes': 42644, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r166', 'priority': -63, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	XGBoost_r31_BAG_L1: 	{'colsample_bytree': 0.7506621909633511, 'enable_categorical': False, 'learning_rate': 0.009974712407899168, 'max_depth': 4, 'min_child_weight': 0.9238550485581797, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r31', 'priority': -64, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	NeuralNetTorch_r185_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.12166942295569863, 'hidden_size': 151, 'learning_rate': 0.0018866871631794007, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 9.190843763153802e-05, 'ag_args': {'name_suffix': '_r185', 'priority': -65, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r160_BAG_L1: 	{'bs': 128, 'emb_drop': 0.3171659718142149, 'epochs': 20, 'layers': [400, 200, 100], 'lr': 0.03087210106068273, 'ps': 0.5909644730871169, 'ag_args': {'name_suffix': '_r160', 'priority': -66, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r60_BAG_L1: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3217885487525205, 'learning_rate': 0.05291587380674719, 'max_ctr_complexity': 5, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r60', 'priority': -67, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	RandomForest_r15_BAG_L1: 	{'max_features': 0.75, 'max_leaf_nodes': 36230, 'min_samples_leaf': 3, 'ag_args': {'name_suffix': '_r15', 'priority': -68, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r135_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.8254432681390782, 'learning_rate': 0.031251656439648626, 'min_data_in_leaf': 50, 'num_leaves': 210, 'ag_args': {'name_suffix': '_r135', 'priority': -69, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	XGBoost_r22_BAG_L1: 	{'colsample_bytree': 0.6326947454697227, 'enable_categorical': False, 'learning_rate': 0.07792091886639502, 'max_depth': 6, 'min_child_weight': 1.0759464955561793, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r22', 'priority': -70, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	NeuralNetFastAI_r69_BAG_L1: 	{'bs': 128, 'emb_drop': 0.3209601865656554, 'epochs': 21, 'layers': [200, 100, 50], 'lr': 0.019935403046870463, 'ps': 0.19846319260751663, 'ag_args': {'name_suffix': '_r69', 'priority': -71, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r6_BAG_L1: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.5734131496361856, 'learning_rate': 0.08472519974533015, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r6', 'priority': -72, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r138_BAG_L1: 	{'bs': 128, 'emb_drop': 0.08669109226243704, 'epochs': 45, 'layers': [800, 400], 'lr': 0.0041554361714983635, 'ps': 0.2669780074016213, 'ag_args': {'name_suffix': '_r138', 'priority': -73, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	LightGBM_r121_BAG_L1: 	{'extra_trees': False, 'feature_fraction': 0.5730390983988963, 'learning_rate': 0.010305352949119608, 'min_data_in_leaf': 10, 'num_leaves': 215, 'ag_args': {'name_suffix': '_r121', 'priority': -74, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r172_BAG_L1: 	{'bs': 512, 'emb_drop': 0.05604276533830355, 'epochs': 32, 'layers': [400], 'lr': 0.027320709383189166, 'ps': 0.022591301744255762, 'ag_args': {'name_suffix': '_r172', 'priority': -75, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r180_BAG_L1: 	{'depth': 7, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 4.43335055453705, 'learning_rate': 0.055406199833457785, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r180', 'priority': -76, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r76_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.006531401073483156, 'hidden_size': 192, 'learning_rate': 0.012418052210914356, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 3.0406866089493607e-05, 'ag_args': {'name_suffix': '_r76', 'priority': -77, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	ExtraTrees_r197_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 40459, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r197', 'priority': -78, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetTorch_r121_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.33926015213879396, 'hidden_size': 247, 'learning_rate': 0.0029983839090226075, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 0.00038926240517691234, 'ag_args': {'name_suffix': '_r121', 'priority': -79, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r127_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.31956392388385874, 'epochs': 25, 'layers': [200, 100], 'lr': 0.08552736732040143, 'ps': 0.0934076022219228, 'ag_args': {'name_suffix': '_r127', 'priority': -80, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	RandomForest_r16_BAG_L1: 	{'max_features': 1.0, 'max_leaf_nodes': 48136, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r16', 'priority': -81, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_r194_BAG_L1: 	{'bs': 256, 'emb_drop': 0.5117456464220826, 'epochs': 21, 'layers': [400, 200, 100], 'lr': 0.007212882302137526, 'ps': 0.2747013981281539, 'ag_args': {'name_suffix': '_r194', 'priority': -82, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r12_BAG_L1: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.835797074498082, 'learning_rate': 0.03534026385152556, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r12', 'priority': -83, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r135_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.06134755114373829, 'hidden_size': 144, 'learning_rate': 0.005834535148903801, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 2.0826540090463355e-09, 'ag_args': {'name_suffix': '_r135', 'priority': -84, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r4_BAG_L1: 	{'bs': 256, 'emb_drop': 0.06099050979107849, 'epochs': 39, 'layers': [200], 'lr': 0.04119582873110387, 'ps': 0.5447097256648953, 'ag_args': {'name_suffix': '_r4', 'priority': -85, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	ExtraTrees_r126_BAG_L1: 	{'max_features': 'sqrt', 'max_leaf_nodes': 29702, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r126', 'priority': -86, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetTorch_r36_BAG_L1: 	{'activation': 'elu', 'dropout_prob': 0.3457125770744979, 'hidden_size': 37, 'learning_rate': 0.006435774191713849, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 2.4012185204155345e-08, 'ag_args': {'name_suffix': '_r36', 'priority': -87, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r100_BAG_L1: 	{'bs': 2048, 'emb_drop': 0.6960805527533755, 'epochs': 38, 'layers': [800, 400], 'lr': 0.0007278526871749883, 'ps': 0.20495582200836318, 'ag_args': {'name_suffix': '_r100', 'priority': -88, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r163_BAG_L1: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.7454481983750014, 'learning_rate': 0.09328642499990342, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r163', 'priority': -89, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	CatBoost_r198_BAG_L1: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.637071465711953, 'learning_rate': 0.04387418552563314, 'max_ctr_complexity': 4, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r198', 'priority': -90, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r187_BAG_L1: 	{'bs': 1024, 'emb_drop': 0.5074958658302495, 'epochs': 42, 'layers': [200, 100, 50], 'lr': 0.026342427824862867, 'ps': 0.34814978753283593, 'ag_args': {'name_suffix': '_r187', 'priority': -91, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r19_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.2211285919550286, 'hidden_size': 196, 'learning_rate': 0.011307978270179143, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 1.8441764217351068e-06, 'ag_args': {'name_suffix': '_r19', 'priority': -92, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	XGBoost_r95_BAG_L1: 	{'colsample_bytree': 0.975937238416368, 'enable_categorical': False, 'learning_rate': 0.06634196266155237, 'max_depth': 5, 'min_child_weight': 1.4088437184127383, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r95', 'priority': -93, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	XGBoost_r34_BAG_L1: 	{'colsample_bytree': 0.546186944730449, 'enable_categorical': False, 'learning_rate': 0.029357102578825213, 'max_depth': 10, 'min_child_weight': 1.1532008198571337, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r34', 'priority': -94, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	LightGBM_r42_BAG_L1: 	{'extra_trees': True, 'feature_fraction': 0.4601361323873807, 'learning_rate': 0.07856777698860955, 'min_data_in_leaf': 12, 'num_leaves': 198, 'ag_args': {'name_suffix': '_r42', 'priority': -95, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r1_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.23713784729000734, 'hidden_size': 200, 'learning_rate': 0.00311256170909018, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 4.573016756474468e-08, 'ag_args': {'name_suffix': '_r1', 'priority': -96, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetTorch_r89_BAG_L1: 	{'activation': 'relu', 'dropout_prob': 0.33567564890346097, 'hidden_size': 245, 'learning_rate': 0.006746560197328548, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.6470047305392933e-10, 'ag_args': {'name_suffix': '_r89', 'priority': -97, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    Fitting 110 L1 models, fit_strategy="sequential" ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 1763.17s of the 2645.40s of remaining time.
    	No valid features to train KNeighborsUnif_BAG_L1... Skipping this model.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 1763.03s of the 2645.25s of remaining time.
    	No valid features to train KNeighborsDist_BAG_L1... Skipping this model.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 1762.90s of the 2645.12s of remaining time.
    	Fitting LightGBMXT_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.14%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/model.pkl
    	0.7763	 = Validation score   (accuracy)
    	3.28s	 = Training   runtime
    	0.24s	 = Validation runtime
    	3710912.1	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 1757.70s of the 2639.93s of remaining time.
    	Fitting LightGBM_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.15%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/model.pkl
    	0.7763	 = Validation score   (accuracy)
    	3.11s	 = Training   runtime
    	0.29s	 = Validation runtime
    	3060540.2	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: RandomForestGini_BAG_L1 ... Training model for up to 1752.54s of the 2634.76s of remaining time.
    	Fitting RandomForestGini_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/utils/model_template.pkl
    	495.55s	= Estimated out-of-fold prediction time...
    	`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/model.pkl
    	0.7788	 = Validation score   (accuracy)
    	73.9s	 = Training   runtime
    	56.24s	 = Validation runtime
    	124381.9	 = Inference  throughput (rows/s | 6995056 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: RandomForestEntr_BAG_L1 ... Training model for up to 1621.60s of the 2503.82s of remaining time.
    	Fitting RandomForestEntr_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/utils/model_template.pkl
    	516.64s	= Estimated out-of-fold prediction time...
    	`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/model.pkl
    	0.7788	 = Validation score   (accuracy)
    	73.77s	 = Training   runtime
    	56.23s	 = Validation runtime
    	124403.0	 = Inference  throughput (rows/s | 6995056 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 1490.82s of the 2373.05s of remaining time.
    	Fitting CatBoost_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.15%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/model.pkl
    	0.7785	 = Validation score   (accuracy)
    	86.3s	 = Training   runtime
    	0.98s	 = Validation runtime
    	896079.5	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: ExtraTreesGini_BAG_L1 ... Training model for up to 1402.64s of the 2284.87s of remaining time.
    	Fitting ExtraTreesGini_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/utils/model_template.pkl
    	497.64s	= Estimated out-of-fold prediction time...
    	`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/model.pkl
    	0.7788	 = Validation score   (accuracy)
    	93.8s	 = Training   runtime
    	62.62s	 = Validation runtime
    	111701.5	 = Inference  throughput (rows/s | 6995056 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: ExtraTreesEntr_BAG_L1 ... Training model for up to 1245.34s of the 2127.56s of remaining time.
    	Fitting ExtraTreesEntr_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/utils/model_template.pkl
    	496.64s	= Estimated out-of-fold prediction time...
    	`use_child_oof` was specified for this model. It will function similarly to a bagged model, but will only fit one child model.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/model.pkl
    	0.7788	 = Validation score   (accuracy)
    	95.16s	 = Training   runtime
    	63.02s	 = Validation runtime
    	110999.7	 = Inference  throughput (rows/s | 6995056 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 1086.30s of the 1968.53s of remaining time.
    	Fitting NeuralNetFastAI_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetFastAI_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetFastAI_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.24%)
    	Warning: Exception caused NeuralNetFastAI_BAG_L1 to fail during training... Skipping this model.
    		[36mray::_ray_fit()[39m (pid=217507, ip=172.30.1.59)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 413, in _ray_fit
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **resources, **kwargs_fold)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/abstract/abstract_model.py", line 925, in fit
        out = self._fit(**kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 361, in _fit
        epochs = self._get_epochs_number(samples_num=len(X) + len_val, epochs=params["epochs"], batch_size=batch_size, time_left=time_left)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 410, in _get_epochs_number
        est_batch_time = self._measure_batch_times(min_batches_count)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 431, in _measure_batch_times
        self.model.fit(1, lr=0, cbs=[batch_time_tracker_callback])
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 266, in fit
        self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 255, in _do_fit
        self._with_events(self._do_epoch, 'epoch', CancelEpochException)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 249, in _do_epoch
        self._do_epoch_train()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 241, in _do_epoch_train
        self._with_events(self.all_batches, 'train', CancelTrainException)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 207, in all_batches
        for o in enumerate(self.dl): self.one_batch(*o)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 237, in one_batch
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 218, in _do_one_batch
        self.pred = self.model(*self.xb)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/tabular/model.py", line 75, in forward
        return self.layers(x)
    UnboundLocalError: local variable 'x' referenced before assignment
    Detailed Traceback:
    Traceback (most recent call last):
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/trainer/abstract_trainer.py", line 2106, in _train_and_save
        model = self._train_single(**model_fit_kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/trainer/abstract_trainer.py", line 1993, in _train_single
        model = model.fit(X=X, y=y, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, total_resources=total_resources, **model_fit_kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/abstract/abstract_model.py", line 925, in fit
        out = self._fit(**kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 270, in _fit
        return super()._fit(X=X, y=y, time_limit=time_limit, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 298, in _fit
        self._fit_folds(
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 724, in _fit_folds
        fold_fitting_strategy.after_all_folds_scheduled()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 690, in after_all_folds_scheduled
        self._run_parallel(X, y, X_pseudo, y_pseudo, model_base_ref, time_limit_fold, head_node_id)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 631, in _run_parallel
        self._process_fold_results(finished, unfinished, fold_ctx)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 587, in _process_fold_results
        raise processed_exception
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 550, in _process_fold_results
        fold_model, pred_proba, time_start_fit, time_end_fit, predict_time, predict_1_time, predict_n_size, fit_num_cpus, fit_num_gpus = self.ray.get(finished)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
        return fn(*args, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
        return func(*args, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/worker.py", line 2753, in get
        values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/ray/_private/worker.py", line 904, in get_objects
        raise value.as_instanceof_cause()
    ray.exceptions.RayTaskError(UnboundLocalError): [36mray::_ray_fit()[39m (pid=217507, ip=172.30.1.59)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/ensemble/fold_fitting_strategy.py", line 413, in _ray_fit
        fold_model.fit(X=X_fold, y=y_fold, X_val=X_val_fold, y_val=y_val_fold, time_limit=time_limit_fold, **resources, **kwargs_fold)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/models/abstract/abstract_model.py", line 925, in fit
        out = self._fit(**kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 361, in _fit
        epochs = self._get_epochs_number(samples_num=len(X) + len_val, epochs=params["epochs"], batch_size=batch_size, time_left=time_left)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 410, in _get_epochs_number
        est_batch_time = self._measure_batch_times(min_batches_count)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/tabular/models/fastainn/tabular_nn_fastai.py", line 431, in _measure_batch_times
        self.model.fit(1, lr=0, cbs=[batch_time_tracker_callback])
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 266, in fit
        self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 255, in _do_fit
        self._with_events(self._do_epoch, 'epoch', CancelEpochException)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 249, in _do_epoch
        self._do_epoch_train()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 241, in _do_epoch_train
        self._with_events(self.all_batches, 'train', CancelTrainException)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 207, in all_batches
        for o in enumerate(self.dl): self.one_batch(*o)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 237, in one_batch
        self._with_events(self._do_one_batch, 'batch', CancelBatchException)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 201, in _with_events
        try: self(f'before_{event_type}');  f()
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/learner.py", line 218, in _do_one_batch
        self.pred = self.model(*self.xb)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
        return forward_call(*args, **kwargs)
      File "/home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/fastai/tabular/model.py", line 75, in forward
        return self.layers(x)
    UnboundLocalError: local variable 'x' referenced before assignment
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetFastAI_BAG_L1/utils/model_template.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: XGBoost_BAG_L1 ... Training model for up to 1047.46s of the 1929.69s of remaining time.
    	Fitting XGBoost_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.19%)
    2025-02-23 11:20:44,662	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:20:44,665	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:20:44,667	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:20:44,668	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:20:44,669	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:20:44,669	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:20:44,669	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/model.pkl
    	0.7763	 = Validation score   (accuracy)
    	33.82s	 = Training   runtime
    	7.05s	 = Validation runtime
    	124067.9	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: NeuralNetTorch_BAG_L1 ... Training model for up to 1011.23s of the 1893.46s of remaining time.
    	Fitting NeuralNetTorch_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.14%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/model.pkl
    	0.7765	 = Validation score   (accuracy)
    	809.8s	 = Training   runtime
    	33.14s	 = Validation runtime
    	26380.8	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: LightGBMLarge_BAG_L1 ... Training model for up to 195.20s of the 1077.42s of remaining time.
    	Fitting LightGBMLarge_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.15%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/model.pkl
    	0.7763	 = Validation score   (accuracy)
    	3.84s	 = Training   runtime
    	0.28s	 = Validation runtime
    	3104991.9	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: CatBoost_r177_BAG_L1 ... Training model for up to 189.47s of the 1071.70s of remaining time.
    	Fitting CatBoost_r177_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.15%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/model.pkl
    	0.7785	 = Validation score   (accuracy)
    	84.95s	 = Training   runtime
    	0.82s	 = Validation runtime
    	1072156.2	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: NeuralNetTorch_r79_BAG_L1 ... Training model for up to 102.61s of the 984.83s of remaining time.
    	Fitting NeuralNetTorch_r79_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.14%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/model.pkl
    	0.7764	 = Validation score   (accuracy)
    	74.62s	 = Training   runtime
    	12.42s	 = Validation runtime
    	70414.0	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: LightGBM_r131_BAG_L1 ... Training model for up to 24.79s of the 907.01s of remaining time.
    	Fitting LightGBM_r131_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.15%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/model.pkl
    	0.7763	 = Validation score   (accuracy)
    	3.13s	 = Training   runtime
    	0.32s	 = Validation runtime
    	2762569.2	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: NeuralNetFastAI_r191_BAG_L1 ... Training model for up to 19.60s of the 901.82s of remaining time.
    	Fitting NeuralNetFastAI_r191_BAG_L1 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetFastAI_r191_BAG_L1/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetFastAI_r191_BAG_L1/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=0.25%)
    	Time limit exceeded... Skipping NeuralNetFastAI_r191_BAG_L1.
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetFastAI_r191_BAG_L1/utils/model_template.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r9_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r96_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r22_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r33_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r42_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r137_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r102_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r13_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r195_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r188_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r145_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r89_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r30_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r130_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r86_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r50_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r11_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r194_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r172_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r69_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r103_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r14_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r161_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r143_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r70_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r156_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r196_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r39_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r167_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r95_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r41_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r98_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r15_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r158_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r86_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r37_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r197_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r49_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r49_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r143_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r127_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r134_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r34_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r94_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r143_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r128_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r111_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r31_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r4_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r65_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r88_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r30_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r49_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r5_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r87_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r71_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r143_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r178_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r166_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r31_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r185_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r160_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r60_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r15_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r135_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r22_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r69_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r6_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r138_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r121_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r172_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r180_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r76_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r197_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r121_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r127_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r16_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r194_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r12_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r135_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r4_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r126_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r36_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r100_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r163_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r198_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r187_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r19_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r95_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r34_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r42_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r1_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r89_BAG_L1 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/utils/oof.pkl
    Model configs that will be trained (in order):
    	WeightedEnsemble_L2: 	{'ag_args': {'valid_base': False, 'name_bag_suffix': '', 'model_type': <class 'autogluon.core.models.greedy_ensemble.greedy_weighted_ensemble_model.GreedyWeightedEnsembleModel'>, 'priority': 0}, 'ag_args_ensemble': {'save_bag_folds': True}}
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.00s of the 878.64s of remaining time.
    	Fitting WeightedEnsemble_L2 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/utils/model_template.pkl
    Subsampling to 1000000 samples to speedup ensemble selection...
    2025-02-23 11:38:15,129	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:38:15,130	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:38:15,130	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:38:15,131	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:38:15,131	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:38:15,131	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    2025-02-23 11:38:15,131	ERROR worker.py:422 -- Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): The worker died unexpectedly while executing this task. Check python-core-worker-*.log files for more information.
    Ensemble size: 2
    Ensemble weights: 
    [0.  0.  0.  0.  0.  0.  0.5 0.  0.  0.  0.5 0.  0. ]
    	6.23s	= Estimated out-of-fold prediction time...
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/model.pkl
    	Ensemble Weights: {'ExtraTreesEntr_BAG_L1': 0.5, 'CatBoost_r177_BAG_L1': 0.5}
    	0.7788	 = Validation score   (accuracy)
    	9.84s	 = Training   runtime
    	0.29s	 = Validation runtime
    	100169.7	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Model configs that will be trained (in order):
    	LightGBMXT_BAG_L2: 	{'extra_trees': True, 'ag_args': {'name_suffix': 'XT', 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    	LightGBM_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>, 'priority': 90}}
    	RandomForestGini_BAG_L2: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    	RandomForestEntr_BAG_L2: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>, 'priority': 80}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>, 'priority': 70}}
    	ExtraTreesGini_BAG_L2: 	{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    	ExtraTreesEntr_BAG_L2: 	{'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass'], 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>, 'priority': 60}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>, 'priority': 50}}
    	XGBoost_BAG_L2: 	{'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>, 'priority': 40}}
    	NeuralNetTorch_BAG_L2: 	{'ag_args': {'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>, 'priority': 25}}
    	LightGBMLarge_BAG_L2: 	{'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	CatBoost_r177_BAG_L2: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r79_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	LightGBM_r131_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.7023601671276614, 'learning_rate': 0.012144796373999013, 'min_data_in_leaf': 14, 'num_leaves': 53, 'ag_args': {'name_suffix': '_r131', 'priority': -3, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r191_BAG_L2: 	{'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r9_BAG_L2: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	LightGBM_r96_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.5636931414546802, 'learning_rate': 0.01518660230385841, 'min_data_in_leaf': 48, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r96', 'priority': -6, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r22_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	XGBoost_r33_BAG_L2: 	{'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r33', 'priority': -8, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	ExtraTrees_r42_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 18392, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r42', 'priority': -9, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_r137_BAG_L2: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.559174625782161, 'learning_rate': 0.04939557741379516, 'max_ctr_complexity': 3, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r137', 'priority': -10, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r102_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r13_BAG_L2: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3274013177541373, 'learning_rate': 0.017301189655111057, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r13', 'priority': -12, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	RandomForest_r195_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 37308, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r195', 'priority': -13, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r188_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.8282601210460099, 'learning_rate': 0.033929021353492905, 'min_data_in_leaf': 6, 'num_leaves': 127, 'ag_args': {'name_suffix': '_r188', 'priority': -14, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r145_BAG_L2: 	{'bs': 128, 'emb_drop': 0.44339037504795686, 'epochs': 31, 'layers': [400, 200, 100], 'lr': 0.008615195908919904, 'ps': 0.19220253419114286, 'ag_args': {'name_suffix': '_r145', 'priority': -15, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	XGBoost_r89_BAG_L2: 	{'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r89', 'priority': -16, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	NeuralNetTorch_r30_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.24622382571353768, 'hidden_size': 159, 'learning_rate': 0.008507536855608535, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.8201539594953562e-06, 'ag_args': {'name_suffix': '_r30', 'priority': -17, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	LightGBM_r130_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.6245777099925497, 'learning_rate': 0.04711573688184715, 'min_data_in_leaf': 56, 'num_leaves': 89, 'ag_args': {'name_suffix': '_r130', 'priority': -18, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r86_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.09976801642258049, 'hidden_size': 135, 'learning_rate': 0.001631450730978947, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 3.867683394425807e-05, 'ag_args': {'name_suffix': '_r86', 'priority': -19, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r50_BAG_L2: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7018061518087038, 'learning_rate': 0.07092851311746352, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r50', 'priority': -20, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r11_BAG_L2: 	{'bs': 128, 'emb_drop': 0.026897798530914306, 'epochs': 31, 'layers': [800, 400], 'lr': 0.08045277634470181, 'ps': 0.4569532219038436, 'ag_args': {'name_suffix': '_r11', 'priority': -21, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	XGBoost_r194_BAG_L2: 	{'colsample_bytree': 0.9090166528779192, 'enable_categorical': True, 'learning_rate': 0.09290221350439203, 'max_depth': 7, 'min_child_weight': 0.8041986915994078, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r194', 'priority': -22, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	ExtraTrees_r172_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 12845, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r172', 'priority': -23, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_r69_BAG_L2: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.0457098345001241, 'learning_rate': 0.050294288910022224, 'max_ctr_complexity': 5, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r69', 'priority': -24, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r103_BAG_L2: 	{'bs': 256, 'emb_drop': 0.1508701680951814, 'epochs': 46, 'layers': [400, 200], 'lr': 0.08794353125787312, 'ps': 0.19110623090573325, 'ag_args': {'name_suffix': '_r103', 'priority': -25, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r14_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.3905837860053583, 'hidden_size': 106, 'learning_rate': 0.0018297905295930797, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 9.178069874232892e-08, 'ag_args': {'name_suffix': '_r14', 'priority': -26, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	LightGBM_r161_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.5898927512279213, 'learning_rate': 0.010464516487486093, 'min_data_in_leaf': 11, 'num_leaves': 252, 'ag_args': {'name_suffix': '_r161', 'priority': -27, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r143_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.6239200452002372, 'epochs': 39, 'layers': [200, 100, 50], 'lr': 0.07170321592506483, 'ps': 0.670815151683455, 'ag_args': {'name_suffix': '_r143', 'priority': -28, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r70_BAG_L2: 	{'depth': 6, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.3584121369544215, 'learning_rate': 0.03743901034980473, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r70', 'priority': -29, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r156_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.5055288166864152, 'epochs': 44, 'layers': [400], 'lr': 0.0047762208542912405, 'ps': 0.06572612802222005, 'ag_args': {'name_suffix': '_r156', 'priority': -30, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	LightGBM_r196_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.5143401489640409, 'learning_rate': 0.00529479887023554, 'min_data_in_leaf': 6, 'num_leaves': 133, 'ag_args': {'name_suffix': '_r196', 'priority': -31, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	RandomForest_r39_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 28310, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r39', 'priority': -32, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	CatBoost_r167_BAG_L2: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.522712492188319, 'learning_rate': 0.08481607830570326, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r167', 'priority': -33, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r95_BAG_L2: 	{'bs': 128, 'emb_drop': 0.6656668277387758, 'epochs': 32, 'layers': [400, 200, 100], 'lr': 0.019326244622675428, 'ps': 0.04084945128641206, 'ag_args': {'name_suffix': '_r95', 'priority': -34, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r41_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.05488816803887784, 'hidden_size': 32, 'learning_rate': 0.0075612897834015985, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.652353009917866e-08, 'ag_args': {'name_suffix': '_r41', 'priority': -35, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	XGBoost_r98_BAG_L2: 	{'colsample_bytree': 0.516652313273348, 'enable_categorical': True, 'learning_rate': 0.007158072983547058, 'max_depth': 9, 'min_child_weight': 0.8567068904025429, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r98', 'priority': -36, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	LightGBM_r15_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.7421180622507277, 'learning_rate': 0.018603888565740096, 'min_data_in_leaf': 6, 'num_leaves': 22, 'ag_args': {'name_suffix': '_r15', 'priority': -37, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r158_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.01030258381183309, 'hidden_size': 111, 'learning_rate': 0.01845979186513771, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 0.00020238017476912164, 'ag_args': {'name_suffix': '_r158', 'priority': -38, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r86_BAG_L2: 	{'depth': 8, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6376578537958237, 'learning_rate': 0.032899230324940465, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r86', 'priority': -39, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r37_BAG_L2: 	{'bs': 512, 'emb_drop': 0.1567472816422661, 'epochs': 41, 'layers': [400, 200, 100], 'lr': 0.06831450078222204, 'ps': 0.4930900813464729, 'ag_args': {'name_suffix': '_r37', 'priority': -40, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r197_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.18109219857068798, 'hidden_size': 250, 'learning_rate': 0.00634181748507711, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 5.3861175580695396e-08, 'ag_args': {'name_suffix': '_r197', 'priority': -41, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r49_BAG_L2: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.353268454214423, 'learning_rate': 0.06028218319511302, 'max_ctr_complexity': 1, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r49', 'priority': -42, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	ExtraTrees_r49_BAG_L2: 	{'max_features': 'sqrt', 'max_leaf_nodes': 28532, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r49', 'priority': -43, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r143_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.9408897917880529, 'learning_rate': 0.01343464462043561, 'min_data_in_leaf': 21, 'num_leaves': 178, 'ag_args': {'name_suffix': '_r143', 'priority': -44, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	RandomForest_r127_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 38572, 'min_samples_leaf': 5, 'ag_args': {'name_suffix': '_r127', 'priority': -45, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_r134_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.006251885504130949, 'epochs': 47, 'layers': [800, 400], 'lr': 0.01329622020483052, 'ps': 0.2677080696008348, 'ag_args': {'name_suffix': '_r134', 'priority': -46, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	RandomForest_r34_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 18242, 'min_samples_leaf': 40, 'ag_args': {'name_suffix': '_r34', 'priority': -47, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r94_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.4341088458599442, 'learning_rate': 0.04034449862560467, 'min_data_in_leaf': 33, 'num_leaves': 16, 'ag_args': {'name_suffix': '_r94', 'priority': -48, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r143_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.1703783780377607, 'hidden_size': 212, 'learning_rate': 0.0004107199833213839, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 1.105439140660822e-07, 'ag_args': {'name_suffix': '_r143', 'priority': -49, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r128_BAG_L2: 	{'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.640921865280573, 'learning_rate': 0.036232951900213306, 'max_ctr_complexity': 3, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r128', 'priority': -50, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r111_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.6343202884164582, 'epochs': 21, 'layers': [400, 200], 'lr': 0.08479209380262258, 'ps': 0.48362560779595565, 'ag_args': {'name_suffix': '_r111', 'priority': -51, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r31_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.013288954106470907, 'hidden_size': 81, 'learning_rate': 0.005340914647396154, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 8.762168370775353e-05, 'ag_args': {'name_suffix': '_r31', 'priority': -52, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	ExtraTrees_r4_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 19935, 'min_samples_leaf': 20, 'ag_args': {'name_suffix': '_r4', 'priority': -53, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_r65_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.22771721361129746, 'epochs': 38, 'layers': [400], 'lr': 0.0005383511954451698, 'ps': 0.3734259772256502, 'ag_args': {'name_suffix': '_r65', 'priority': -54, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetFastAI_r88_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.4329361816589235, 'epochs': 50, 'layers': [400], 'lr': 0.09501311551121323, 'ps': 0.2863378667611431, 'ag_args': {'name_suffix': '_r88', 'priority': -55, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	LightGBM_r30_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.9773131270704629, 'learning_rate': 0.010534290864227067, 'min_data_in_leaf': 21, 'num_leaves': 111, 'ag_args': {'name_suffix': '_r30', 'priority': -56, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	XGBoost_r49_BAG_L2: 	{'colsample_bytree': 0.7452294043087835, 'enable_categorical': False, 'learning_rate': 0.038404229910104046, 'max_depth': 7, 'min_child_weight': 0.5564183327139662, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r49', 'priority': -57, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	CatBoost_r5_BAG_L2: 	{'depth': 4, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.894432181094842, 'learning_rate': 0.055078095725390575, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r5', 'priority': -58, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r87_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.36669080773207274, 'hidden_size': 95, 'learning_rate': 0.015280159186761077, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.3082489374636015e-08, 'ag_args': {'name_suffix': '_r87', 'priority': -59, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetTorch_r71_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.3027114570947557, 'hidden_size': 196, 'learning_rate': 0.006482759295309238, 'num_layers': 1, 'use_batchnorm': False, 'weight_decay': 1.2806509958776e-12, 'ag_args': {'name_suffix': '_r71', 'priority': -60, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	CatBoost_r143_BAG_L2: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 1.6761016245166451, 'learning_rate': 0.06566144806528762, 'max_ctr_complexity': 2, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r143', 'priority': -61, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	ExtraTrees_r178_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 29813, 'min_samples_leaf': 4, 'ag_args': {'name_suffix': '_r178', 'priority': -62, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	RandomForest_r166_BAG_L2: 	{'max_features': 'log2', 'max_leaf_nodes': 42644, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r166', 'priority': -63, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	XGBoost_r31_BAG_L2: 	{'colsample_bytree': 0.7506621909633511, 'enable_categorical': False, 'learning_rate': 0.009974712407899168, 'max_depth': 4, 'min_child_weight': 0.9238550485581797, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r31', 'priority': -64, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	NeuralNetTorch_r185_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.12166942295569863, 'hidden_size': 151, 'learning_rate': 0.0018866871631794007, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 9.190843763153802e-05, 'ag_args': {'name_suffix': '_r185', 'priority': -65, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r160_BAG_L2: 	{'bs': 128, 'emb_drop': 0.3171659718142149, 'epochs': 20, 'layers': [400, 200, 100], 'lr': 0.03087210106068273, 'ps': 0.5909644730871169, 'ag_args': {'name_suffix': '_r160', 'priority': -66, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r60_BAG_L2: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.3217885487525205, 'learning_rate': 0.05291587380674719, 'max_ctr_complexity': 5, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r60', 'priority': -67, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	RandomForest_r15_BAG_L2: 	{'max_features': 0.75, 'max_leaf_nodes': 36230, 'min_samples_leaf': 3, 'ag_args': {'name_suffix': '_r15', 'priority': -68, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	LightGBM_r135_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.8254432681390782, 'learning_rate': 0.031251656439648626, 'min_data_in_leaf': 50, 'num_leaves': 210, 'ag_args': {'name_suffix': '_r135', 'priority': -69, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	XGBoost_r22_BAG_L2: 	{'colsample_bytree': 0.6326947454697227, 'enable_categorical': False, 'learning_rate': 0.07792091886639502, 'max_depth': 6, 'min_child_weight': 1.0759464955561793, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r22', 'priority': -70, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	NeuralNetFastAI_r69_BAG_L2: 	{'bs': 128, 'emb_drop': 0.3209601865656554, 'epochs': 21, 'layers': [200, 100, 50], 'lr': 0.019935403046870463, 'ps': 0.19846319260751663, 'ag_args': {'name_suffix': '_r69', 'priority': -71, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r6_BAG_L2: 	{'depth': 4, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 1.5734131496361856, 'learning_rate': 0.08472519974533015, 'max_ctr_complexity': 3, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r6', 'priority': -72, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r138_BAG_L2: 	{'bs': 128, 'emb_drop': 0.08669109226243704, 'epochs': 45, 'layers': [800, 400], 'lr': 0.0041554361714983635, 'ps': 0.2669780074016213, 'ag_args': {'name_suffix': '_r138', 'priority': -73, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	LightGBM_r121_BAG_L2: 	{'extra_trees': False, 'feature_fraction': 0.5730390983988963, 'learning_rate': 0.010305352949119608, 'min_data_in_leaf': 10, 'num_leaves': 215, 'ag_args': {'name_suffix': '_r121', 'priority': -74, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetFastAI_r172_BAG_L2: 	{'bs': 512, 'emb_drop': 0.05604276533830355, 'epochs': 32, 'layers': [400], 'lr': 0.027320709383189166, 'ps': 0.022591301744255762, 'ag_args': {'name_suffix': '_r172', 'priority': -75, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r180_BAG_L2: 	{'depth': 7, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 4.43335055453705, 'learning_rate': 0.055406199833457785, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r180', 'priority': -76, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r76_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.006531401073483156, 'hidden_size': 192, 'learning_rate': 0.012418052210914356, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 3.0406866089493607e-05, 'ag_args': {'name_suffix': '_r76', 'priority': -77, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	ExtraTrees_r197_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 40459, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r197', 'priority': -78, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetTorch_r121_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.33926015213879396, 'hidden_size': 247, 'learning_rate': 0.0029983839090226075, 'num_layers': 5, 'use_batchnorm': False, 'weight_decay': 0.00038926240517691234, 'ag_args': {'name_suffix': '_r121', 'priority': -79, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r127_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.31956392388385874, 'epochs': 25, 'layers': [200, 100], 'lr': 0.08552736732040143, 'ps': 0.0934076022219228, 'ag_args': {'name_suffix': '_r127', 'priority': -80, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	RandomForest_r16_BAG_L2: 	{'max_features': 1.0, 'max_leaf_nodes': 48136, 'min_samples_leaf': 1, 'ag_args': {'name_suffix': '_r16', 'priority': -81, 'model_type': <class 'autogluon.tabular.models.rf.rf_model.RFModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetFastAI_r194_BAG_L2: 	{'bs': 256, 'emb_drop': 0.5117456464220826, 'epochs': 21, 'layers': [400, 200, 100], 'lr': 0.007212882302137526, 'ps': 0.2747013981281539, 'ag_args': {'name_suffix': '_r194', 'priority': -82, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r12_BAG_L2: 	{'depth': 7, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 4.835797074498082, 'learning_rate': 0.03534026385152556, 'max_ctr_complexity': 5, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r12', 'priority': -83, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetTorch_r135_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.06134755114373829, 'hidden_size': 144, 'learning_rate': 0.005834535148903801, 'num_layers': 5, 'use_batchnorm': True, 'weight_decay': 2.0826540090463355e-09, 'ag_args': {'name_suffix': '_r135', 'priority': -84, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r4_BAG_L2: 	{'bs': 256, 'emb_drop': 0.06099050979107849, 'epochs': 39, 'layers': [200], 'lr': 0.04119582873110387, 'ps': 0.5447097256648953, 'ag_args': {'name_suffix': '_r4', 'priority': -85, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	ExtraTrees_r126_BAG_L2: 	{'max_features': 'sqrt', 'max_leaf_nodes': 29702, 'min_samples_leaf': 2, 'ag_args': {'name_suffix': '_r126', 'priority': -86, 'model_type': <class 'autogluon.tabular.models.xt.xt_model.XTModel'>}, 'ag_args_ensemble': {'use_child_oof': True}}
    	NeuralNetTorch_r36_BAG_L2: 	{'activation': 'elu', 'dropout_prob': 0.3457125770744979, 'hidden_size': 37, 'learning_rate': 0.006435774191713849, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 2.4012185204155345e-08, 'ag_args': {'name_suffix': '_r36', 'priority': -87, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetFastAI_r100_BAG_L2: 	{'bs': 2048, 'emb_drop': 0.6960805527533755, 'epochs': 38, 'layers': [800, 400], 'lr': 0.0007278526871749883, 'ps': 0.20495582200836318, 'ag_args': {'name_suffix': '_r100', 'priority': -88, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	CatBoost_r163_BAG_L2: 	{'depth': 5, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.7454481983750014, 'learning_rate': 0.09328642499990342, 'max_ctr_complexity': 1, 'one_hot_max_size': 2, 'ag_args': {'name_suffix': '_r163', 'priority': -89, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	CatBoost_r198_BAG_L2: 	{'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 3.637071465711953, 'learning_rate': 0.04387418552563314, 'max_ctr_complexity': 4, 'one_hot_max_size': 5, 'ag_args': {'name_suffix': '_r198', 'priority': -90, 'model_type': <class 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel'>}}
    	NeuralNetFastAI_r187_BAG_L2: 	{'bs': 1024, 'emb_drop': 0.5074958658302495, 'epochs': 42, 'layers': [200, 100, 50], 'lr': 0.026342427824862867, 'ps': 0.34814978753283593, 'ag_args': {'name_suffix': '_r187', 'priority': -91, 'model_type': <class 'autogluon.tabular.models.fastainn.tabular_nn_fastai.NNFastAiTabularModel'>}}
    	NeuralNetTorch_r19_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.2211285919550286, 'hidden_size': 196, 'learning_rate': 0.011307978270179143, 'num_layers': 1, 'use_batchnorm': True, 'weight_decay': 1.8441764217351068e-06, 'ag_args': {'name_suffix': '_r19', 'priority': -92, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	XGBoost_r95_BAG_L2: 	{'colsample_bytree': 0.975937238416368, 'enable_categorical': False, 'learning_rate': 0.06634196266155237, 'max_depth': 5, 'min_child_weight': 1.4088437184127383, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r95', 'priority': -93, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	XGBoost_r34_BAG_L2: 	{'colsample_bytree': 0.546186944730449, 'enable_categorical': False, 'learning_rate': 0.029357102578825213, 'max_depth': 10, 'min_child_weight': 1.1532008198571337, 'ag_args': {'problem_types': ['binary', 'multiclass', 'regression', 'softclass'], 'name_suffix': '_r34', 'priority': -94, 'model_type': <class 'autogluon.tabular.models.xgboost.xgboost_model.XGBoostModel'>}}
    	LightGBM_r42_BAG_L2: 	{'extra_trees': True, 'feature_fraction': 0.4601361323873807, 'learning_rate': 0.07856777698860955, 'min_data_in_leaf': 12, 'num_leaves': 198, 'ag_args': {'name_suffix': '_r42', 'priority': -95, 'model_type': <class 'autogluon.tabular.models.lgb.lgb_model.LGBModel'>}}
    	NeuralNetTorch_r1_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.23713784729000734, 'hidden_size': 200, 'learning_rate': 0.00311256170909018, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 4.573016756474468e-08, 'ag_args': {'name_suffix': '_r1', 'priority': -96, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    	NeuralNetTorch_r89_BAG_L2: 	{'activation': 'relu', 'dropout_prob': 0.33567564890346097, 'hidden_size': 245, 'learning_rate': 0.006746560197328548, 'num_layers': 3, 'use_batchnorm': True, 'weight_decay': 1.6470047305392933e-10, 'ag_args': {'name_suffix': '_r89', 'priority': -97, 'model_type': <class 'autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch.TabularNeuralNetTorchModel'>}}
    Fitting 108 L2 models, fit_strategy="sequential" ...
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/utils/oof.pkl
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 868.22s of the 867.67s of remaining time.
    	Fitting LightGBMXT_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=3.80%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/model.pkl
    	0.7788	 = Validation score   (accuracy)
    	37.4s	 = Training   runtime
    	1.72s	 = Validation runtime
    	10050.3	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 828.14s of the 827.59s of remaining time.
    	Fitting LightGBM_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/utils/model_template.pkl
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (8 workers, per: cpus=4, gpus=0, memory=3.78%)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/model.pkl
    	0.7788	 = Validation score   (accuracy)
    	31.98s	 = Training   runtime
    	1.64s	 = Validation runtime
    	10058.9	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Fitting model: RandomForestGini_BAG_L2 ... Training model for up to 793.54s of the 792.99s of remaining time.
    	Fitting RandomForestGini_BAG_L2 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L2/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L2/utils/model_template.pkl
    	Warning: Reducing model 'n_estimators' from 300 -> 259 due to low time. Expected time usage reduced from 916.4s -> 792.8s...
    	518.84s	= Estimated out-of-fold prediction time...
    	Not enough time to generate out-of-fold predictions for model. Estimated time required was 518.84s compared to 129.56s of available time.
    	Time limit exceeded... Skipping RandomForestGini_BAG_L2.
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L2/utils/model_template.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForestEntr_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTreesGini_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTreesEntr_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBMLarge_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r177_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r79_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r131_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r191_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r9_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r96_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r22_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r33_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r42_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r137_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r102_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r13_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r195_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r188_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r145_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r89_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r30_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r130_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r86_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r50_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r11_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r194_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r172_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r69_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r103_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r14_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r161_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r143_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r70_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r156_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r196_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r39_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r167_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r95_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r41_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r98_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r15_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r158_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r86_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r37_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r197_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r49_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r49_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r143_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r127_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r134_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r34_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r94_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r143_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r128_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r111_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r31_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r4_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r65_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r88_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r30_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r49_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r5_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r87_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r71_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r143_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r178_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r166_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r31_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r185_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r160_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r60_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r15_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r135_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r22_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r69_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r6_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r138_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r121_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r172_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r180_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r76_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r197_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r121_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r127_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping RandomForest_r16_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r194_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r12_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r135_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r4_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping ExtraTrees_r126_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r36_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r100_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r163_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping CatBoost_r198_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetFastAI_r187_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r19_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r95_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping XGBoost_r34_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping LightGBM_r42_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r1_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Skipping NeuralNetTorch_r89_BAG_L2 due to lack of time remaining.
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/utils/oof.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/utils/oof.pkl
    Model configs that will be trained (in order):
    	WeightedEnsemble_L3: 	{'ag_args': {'valid_base': False, 'name_bag_suffix': '', 'model_type': <class 'autogluon.core.models.greedy_ensemble.greedy_weighted_ensemble_model.GreedyWeightedEnsembleModel'>, 'priority': 0}, 'ag_args_ensemble': {'save_bag_folds': True}}
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.00s of the -109.87s of remaining time.
    	Fitting WeightedEnsemble_L3 with 'num_gpus': 0, 'num_cpus': 32
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/utils/model_template.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/utils/model_template.pkl
    Subsampling to 1000000 samples to speedup ensemble selection...
    Ensemble size: 2
    Ensemble weights: 
    [0.  0.  0.  0.  0.  0.  0.5 0.  0.  0.  0.5 0.  0.  0.  0. ]
    	7.03s	= Estimated out-of-fold prediction time...
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/utils/oof.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/model.pkl
    	Ensemble Weights: {'ExtraTreesEntr_BAG_L1': 0.5, 'CatBoost_r177_BAG_L1': 0.5}
    	0.7788	 = Validation score   (accuracy)
    	11.36s	 = Training   runtime
    	0.29s	 = Validation runtime
    	100173.4	 = Inference  throughput (rows/s | 874382 batch size)
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    AutoGluon training complete, total runtime = 2801.34s ... Best model: WeightedEnsemble_L2 | Estimated inference throughput: 100169.7 rows/s (874382 batch size)
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Enabling decision threshold calibration (calibrate_decision_threshold='auto', metric is valid, problem_type is 'binary')
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/utils/data/y.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/utils/oof.pkl
    Subsampling y to 1000000 samples to speedup threshold calibration...
    Calibrating decision threshold to optimize metric accuracy | Checking 51 thresholds...
    	threshold: 0.500	| val: 0.7786	| NEW BEST
    	threshold: 0.480	| val: 0.7786
    	threshold: 0.520	| val: 0.7786
    	threshold: 0.460	| val: 0.7786
    	threshold: 0.540	| val: 0.7786
    	threshold: 0.440	| val: 0.7786
    	threshold: 0.560	| val: 0.7786
    	threshold: 0.420	| val: 0.7785
    	threshold: 0.580	| val: 0.7786
    	threshold: 0.400	| val: 0.7785
    	threshold: 0.600	| val: 0.7786
    	threshold: 0.380	| val: 0.7784
    	threshold: 0.620	| val: 0.7786
    	threshold: 0.360	| val: 0.7784
    	threshold: 0.640	| val: 0.7785
    	threshold: 0.340	| val: 0.2259
    	threshold: 0.660	| val: 0.7785
    	threshold: 0.320	| val: 0.2257
    	threshold: 0.680	| val: 0.7784
    	threshold: 0.300	| val: 0.2256
    	threshold: 0.700	| val: 0.7783
    	threshold: 0.280	| val: 0.2256
    	threshold: 0.720	| val: 0.7781
    	threshold: 0.260	| val: 0.2256
    	threshold: 0.740	| val: 0.7780
    	threshold: 0.240	| val: 0.2252
    	threshold: 0.760	| val: 0.7777
    	threshold: 0.220	| val: 0.2239
    	threshold: 0.780	| val: 0.7777
    	threshold: 0.200	| val: 0.2239
    	threshold: 0.800	| val: 0.7773
    	threshold: 0.180	| val: 0.2239
    	threshold: 0.820	| val: 0.7761
    	threshold: 0.160	| val: 0.2239
    	threshold: 0.840	| val: 0.7761
    	threshold: 0.140	| val: 0.2239
    	threshold: 0.860	| val: 0.7761
    	threshold: 0.120	| val: 0.2239
    	threshold: 0.880	| val: 0.7761
    	threshold: 0.100	| val: 0.2239
    	threshold: 0.900	| val: 0.7761
    	threshold: 0.080	| val: 0.2239
    	threshold: 0.920	| val: 0.7761
    	threshold: 0.060	| val: 0.2239
    	threshold: 0.940	| val: 0.7761
    	threshold: 0.040	| val: 0.2239
    	threshold: 0.960	| val: 0.7761
    	threshold: 0.020	| val: 0.2239
    	threshold: 0.980	| val: 0.7761
    	threshold: 0.000	| val: 0.2239
    	threshold: 1.000	| val: 0.7761
    Calibrating decision threshold via fine-grained search | Checking 38 thresholds...
    	threshold: 0.501	| val: 0.7786
    	threshold: 0.502	| val: 0.7786
    	threshold: 0.503	| val: 0.7786
    	threshold: 0.504	| val: 0.7786
    	threshold: 0.505	| val: 0.7786
    	threshold: 0.506	| val: 0.7786
    	threshold: 0.507	| val: 0.7786
    	threshold: 0.508	| val: 0.7786
    	threshold: 0.509	| val: 0.7786
    	threshold: 0.510	| val: 0.7786
    	threshold: 0.511	| val: 0.7786
    	threshold: 0.512	| val: 0.7786
    	threshold: 0.513	| val: 0.7786
    	threshold: 0.514	| val: 0.7786
    	threshold: 0.515	| val: 0.7786
    	threshold: 0.516	| val: 0.7786
    	threshold: 0.517	| val: 0.7786
    	threshold: 0.518	| val: 0.7786
    	threshold: 0.519	| val: 0.7786
    	threshold: 0.499	| val: 0.7786
    	threshold: 0.498	| val: 0.7786
    	threshold: 0.497	| val: 0.7786
    	threshold: 0.496	| val: 0.7786
    	threshold: 0.495	| val: 0.7786
    	threshold: 0.494	| val: 0.7786
    	threshold: 0.493	| val: 0.7786
    	threshold: 0.492	| val: 0.7786
    	threshold: 0.491	| val: 0.7786
    	threshold: 0.490	| val: 0.7786
    	threshold: 0.489	| val: 0.7786
    	threshold: 0.488	| val: 0.7786
    	threshold: 0.487	| val: 0.7786
    	threshold: 0.486	| val: 0.7786
    	threshold: 0.485	| val: 0.7786
    	threshold: 0.484	| val: 0.7786
    	threshold: 0.483	| val: 0.7786
    	threshold: 0.482	| val: 0.7786
    	threshold: 0.481	| val: 0.7786
    	Base Threshold: 0.500	| val: 0.7786
    	Best Threshold: 0.500	| val: 0.7786
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/trainer.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/learner.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/predictor.pkl
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/version.txt with contents "1.2"
    Saving /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/metadata.json
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248")
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/model.pkl


    *** Summary of fit() ***
    Estimated performance of each model:
                            model  score_val eval_metric  pred_time_val     fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0         WeightedEnsemble_L3   0.778789    accuracy      64.120703   191.468206                0.286449          11.359117            3       True         17
    1         WeightedEnsemble_L2   0.778789    accuracy      64.123274   189.950677                0.289020           9.841588            2       True         14
    2       ExtraTreesGini_BAG_L1   0.778788    accuracy      62.622783    93.800065               62.622783          93.800065            1       True          6
    3       ExtraTreesEntr_BAG_L1   0.778788    accuracy      63.018718    95.155787               63.018718          95.155787            1       True          7
    4     RandomForestEntr_BAG_L1   0.778788    accuracy      56.229006    73.767857               56.229006          73.767857            1       True          4
    5     RandomForestGini_BAG_L1   0.778788    accuracy      56.238548    73.897823               56.238548          73.897823            1       True          3
    6             LightGBM_BAG_L2   0.778763    accuracy     295.271434  1471.467147                1.641614          31.981683            2       True         16
    7           LightGBMXT_BAG_L2   0.778763    accuracy     295.345976  1476.882533                1.716156          37.397069            2       True         15
    8        CatBoost_r177_BAG_L1   0.778507    accuracy       0.815536    84.953301                0.815536          84.953301            1       True         11
    9             CatBoost_BAG_L1   0.778507    accuracy       0.975786    86.297275                0.975786          86.297275            1       True          5
    10      NeuralNetTorch_BAG_L1   0.776506    accuracy      33.144672   809.801182               33.144672         809.801182            1       True          9
    11  NeuralNetTorch_r79_BAG_L1   0.776417    accuracy      12.417727    74.623855               12.417727          74.623855            1       True         12
    12             XGBoost_BAG_L1   0.776313    accuracy       7.047609    33.821504                7.047609          33.821504            1       True          8
    13          LightGBMXT_BAG_L1   0.776285    accuracy       0.235625     3.281832                0.235625           3.281832            1       True          1
    14       LightGBMLarge_BAG_L1   0.776285    accuracy       0.281605     3.839002                0.281605           3.839002            1       True         10
    15            LightGBM_BAG_L1   0.776285    accuracy       0.285695     3.112715                0.285695           3.112715            1       True          2
    16       LightGBM_r131_BAG_L1   0.776285    accuracy       0.316510     3.133264                0.316510           3.133264            1       True         13
    Number of models trained: 17
    Types of models trained:
    {'StackerEnsembleModel_TabularNeuralNetTorch', 'StackerEnsembleModel_XT', 'WeightedEnsembleModel', 'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_XGBoost', 'StackerEnsembleModel_LGB', 'StackerEnsembleModel_RF'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', []) : 1 | ['URL']
    *** End of fit() summary ***


    /home/apic/miniconda3/envs/malicious_url_venv/lib/python3.10/site-packages/autogluon/core/utils/plots.py:169: UserWarning: AutoGluon summary plots cannot be created because bokeh is not installed. To see plots, please do: "pip install bokeh==2.0.1"
      warnings.warn('AutoGluon summary plots cannot be created because bokeh is not installed. To see plots, please do: "pip install bokeh==2.0.1"')



```python
predictor.leaderboard(train_df, silent=True)
```

    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/model.pkl





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>score_test</th>
      <th>score_val</th>
      <th>eval_metric</th>
      <th>pred_time_test</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_test_marginal</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CatBoost_r177_BAG_L1</td>
      <td>0.778793</td>
      <td>0.778507</td>
      <td>accuracy</td>
      <td>1.212548</td>
      <td>0.815536</td>
      <td>84.953301</td>
      <td>1.212548</td>
      <td>0.815536</td>
      <td>84.953301</td>
      <td>1</td>
      <td>True</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CatBoost_BAG_L1</td>
      <td>0.778793</td>
      <td>0.778507</td>
      <td>accuracy</td>
      <td>1.328939</td>
      <td>0.975786</td>
      <td>86.297275</td>
      <td>1.328939</td>
      <td>0.975786</td>
      <td>86.297275</td>
      <td>1</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestEntr_BAG_L1</td>
      <td>0.778793</td>
      <td>0.778788</td>
      <td>accuracy</td>
      <td>8.363371</td>
      <td>56.229006</td>
      <td>73.767857</td>
      <td>8.363371</td>
      <td>56.229006</td>
      <td>73.767857</td>
      <td>1</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RandomForestGini_BAG_L1</td>
      <td>0.778793</td>
      <td>0.778788</td>
      <td>accuracy</td>
      <td>8.401056</td>
      <td>56.238548</td>
      <td>73.897823</td>
      <td>8.401056</td>
      <td>56.238548</td>
      <td>73.897823</td>
      <td>1</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ExtraTreesGini_BAG_L1</td>
      <td>0.778793</td>
      <td>0.778788</td>
      <td>accuracy</td>
      <td>8.462774</td>
      <td>62.622783</td>
      <td>93.800065</td>
      <td>8.462774</td>
      <td>62.622783</td>
      <td>93.800065</td>
      <td>1</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ExtraTreesEntr_BAG_L1</td>
      <td>0.778793</td>
      <td>0.778788</td>
      <td>accuracy</td>
      <td>8.661296</td>
      <td>63.018718</td>
      <td>95.155787</td>
      <td>8.661296</td>
      <td>63.018718</td>
      <td>95.155787</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>WeightedEnsemble_L2</td>
      <td>0.778793</td>
      <td>0.778789</td>
      <td>accuracy</td>
      <td>9.916466</td>
      <td>64.123274</td>
      <td>189.950677</td>
      <td>0.042623</td>
      <td>0.289020</td>
      <td>9.841588</td>
      <td>2</td>
      <td>True</td>
      <td>14</td>
    </tr>
    <tr>
      <th>7</th>
      <td>WeightedEnsemble_L3</td>
      <td>0.778793</td>
      <td>0.778789</td>
      <td>accuracy</td>
      <td>9.920033</td>
      <td>64.120703</td>
      <td>191.468206</td>
      <td>0.046189</td>
      <td>0.286449</td>
      <td>11.359117</td>
      <td>3</td>
      <td>True</td>
      <td>17</td>
    </tr>
    <tr>
      <th>8</th>
      <td>LightGBM_BAG_L2</td>
      <td>0.778793</td>
      <td>0.778763</td>
      <td>accuracy</td>
      <td>384.732949</td>
      <td>295.271434</td>
      <td>1471.467147</td>
      <td>3.332280</td>
      <td>1.641614</td>
      <td>31.981683</td>
      <td>2</td>
      <td>True</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LightGBMXT_BAG_L2</td>
      <td>0.778793</td>
      <td>0.778763</td>
      <td>accuracy</td>
      <td>385.185134</td>
      <td>295.345976</td>
      <td>1476.882533</td>
      <td>3.784466</td>
      <td>1.716156</td>
      <td>37.397069</td>
      <td>2</td>
      <td>True</td>
      <td>15</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NeuralNetTorch_BAG_L1</td>
      <td>0.776506</td>
      <td>0.776506</td>
      <td>accuracy</td>
      <td>210.172528</td>
      <td>33.144672</td>
      <td>809.801182</td>
      <td>210.172528</td>
      <td>33.144672</td>
      <td>809.801182</td>
      <td>1</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NeuralNetTorch_r79_BAG_L1</td>
      <td>0.776483</td>
      <td>0.776417</td>
      <td>accuracy</td>
      <td>105.525325</td>
      <td>12.417727</td>
      <td>74.623855</td>
      <td>105.525325</td>
      <td>12.417727</td>
      <td>74.623855</td>
      <td>1</td>
      <td>True</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>LightGBMLarge_BAG_L1</td>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>accuracy</td>
      <td>0.470964</td>
      <td>0.281605</td>
      <td>3.839002</td>
      <td>0.470964</td>
      <td>0.281605</td>
      <td>3.839002</td>
      <td>1</td>
      <td>True</td>
      <td>10</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LightGBM_r131_BAG_L1</td>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>accuracy</td>
      <td>0.484887</td>
      <td>0.316510</td>
      <td>3.133264</td>
      <td>0.484887</td>
      <td>0.316510</td>
      <td>3.133264</td>
      <td>1</td>
      <td>True</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LightGBM_BAG_L1</td>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>accuracy</td>
      <td>0.496650</td>
      <td>0.285695</td>
      <td>3.112715</td>
      <td>0.496650</td>
      <td>0.285695</td>
      <td>3.112715</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>LightGBMXT_BAG_L1</td>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>accuracy</td>
      <td>0.528022</td>
      <td>0.235625</td>
      <td>3.281832</td>
      <td>0.528022</td>
      <td>0.235625</td>
      <td>3.281832</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>XGBoost_BAG_L1</td>
      <td>0.776285</td>
      <td>0.776313</td>
      <td>accuracy</td>
      <td>27.292310</td>
      <td>7.047609</td>
      <td>33.821504</td>
      <td>27.292310</td>
      <td>7.047609</td>
      <td>33.821504</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred = predictor.predict_proba_multi(test_df)
model_names = pred.keys()

df_dict = {}
for name in model_names:
    df_dict[name] = pred[name][0]
    
pred_df = pd.DataFrame(df_dict)
pred_df
```

    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/model.pkl





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LightGBMXT_BAG_L1</th>
      <th>LightGBM_BAG_L1</th>
      <th>RandomForestGini_BAG_L1</th>
      <th>RandomForestEntr_BAG_L1</th>
      <th>CatBoost_BAG_L1</th>
      <th>ExtraTreesGini_BAG_L1</th>
      <th>ExtraTreesEntr_BAG_L1</th>
      <th>XGBoost_BAG_L1</th>
      <th>NeuralNetTorch_BAG_L1</th>
      <th>LightGBMLarge_BAG_L1</th>
      <th>CatBoost_r177_BAG_L1</th>
      <th>NeuralNetTorch_r79_BAG_L1</th>
      <th>LightGBM_r131_BAG_L1</th>
      <th>WeightedEnsemble_L2</th>
      <th>LightGBMXT_BAG_L2</th>
      <th>LightGBM_BAG_L2</th>
      <th>WeightedEnsemble_L3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1747684</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>1747685</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>1747686</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>1747687</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
    <tr>
      <th>1747688</th>
      <td>0.776285</td>
      <td>0.776285</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.515667</td>
      <td>0.777719</td>
      <td>0.777719</td>
      <td>0.76064</td>
      <td>0.777587</td>
      <td>0.776285</td>
      <td>0.521502</td>
      <td>0.775217</td>
      <td>0.776285</td>
      <td>0.64961</td>
      <td>0.777511</td>
      <td>0.776917</td>
      <td>0.64961</td>
    </tr>
  </tbody>
</table>
<p>1747689 rows × 17 columns</p>
</div>




```python
pred = predictor.predict_proba_multi(test_df)
model_name = predictor.model_best

submission_df = pd.read_csv('./data/sample_submission.csv')
submission_df['probability'] = pred[model_name][0]
submission_df.to_csv('./prediction/ensemble_predictions.csv', index=False)
```

    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/RandomForestEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesGini_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/ExtraTreesEntr_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/XGBoost_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMLarge_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/CatBoost_r177_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/NeuralNetTorch_r79_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_r131_BAG_L1/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBMXT_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/LightGBM_BAG_L2/model.pkl
    Loading: /home/apic/python/dacon_project/malicious_url/AutogluonModels/ag-20250223_015248/models/WeightedEnsemble_L3/model.pkl

