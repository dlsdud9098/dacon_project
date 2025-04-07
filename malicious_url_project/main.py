import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
from sklearn.utils.class_weight import compute_class_weight
from glob import glob
import os
from itertools import combinations

def load_data(train_path, test_path):
    # 데이터 불러오기
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # [.] => .
    train_df['URL'] = train_df['URL'].apply(lambda x: x.replace('[.]', '.'))
    test_df['URL'] = test_df['URL'].apply(lambda x: x.replace('[.]', '.'))
    
    return train_df, test_df

def model_fit(train_df, test_df, checkpoint, checkpoint_name):
    predictor = MultiModalPredictor(label='label', problem_type='binary')
    
    # 라벨별 가중치 계산
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_df['label']), y=train_df['label'].values)
    weights /= weights.sum()
    weights = list(weights)
    
    predictor.fit(
        train_data=train_df,
        presets='best_quality',
        time_limit=None,
        column_types = {'URL':'text'},
        seed=42,
        hyperparameters={
            "model.hf_text.checkpoint_name": checkpoint,
            "env.per_gpu_batch_size": 32,
            "optimization.patience": 3,
            "optimization.loss_function": "focal_loss",
            "optimization.focal_loss.alpha": weights,
        }
    )
    
    model_save(predictor, checkpoint_name, test_df)
    
    return predictor
    
def model_save(predictor, checkpoint_name, test_df):
    preds = predictor.predict_proba(
        test_df
    )
    
    predictor.save(f'./model/{checkpoint_name}')
    
    submission_df = pd.read_csv('./data/sample_submission.csv')
    submission_df['probability'] = preds[1]
    submission_df.to_csv(f'./prediction/{checkpoint_name}.csv', index=False)
    
def model_ensemble(checkpoint_names, test_df):
    pairs = list(combinations(checkpoint_names, 2)) 
    
    for pair in pairs:
        pair_1_path = os.path.join(f'./model/{pairs[0]}')
        pair_2_path = os.path.join(f'./model/{pairs[1]}')
        
        pair_1 = MultiModalPredictor.load(pair_1_path)
        pair_2 = MultiModalPredictor.load(pair_2_path)
        
        # 최고 성능 데이터를 추적
        max_data = 0
        best_weight_1 = 0
        best_weight_2 = 0
        
        # 가중치 조합 탐색
        for model_weight in np.arange(0.1, 1, 0.1):
            weight_1 = model_weight
            weight_2 = 1 - model_weight
            
            # 예측 결과 계산
            predictions_1 = pair_1.predict_proba(test_df)[:,1]
            predictions_2 = pair_2.predict_proba(test_df)[:,1]
            
            
            # 가중 평균 앙상블 계산
            final_predictions = (predictions_1 * weight_1) + (predictions_2 * weight_2)
            
            # 예제: 평가 점수 계산 (여기서는 단순 합으로 가정)
            score = final_predictions.sum()  # 실제로는 정확도, F1-score 등을 사용
            
            
            # 최고 점수와 가중치 업데이트
            if score > max_data:
                max_data = score
                best_weight_1 = weight_1
                best_weight_2 = weight_2
                
        print(f'Model_1: {pair[0]}, Weight: {best_weight_1}\nModel_2: {pair[1]}, Weight: {best_weight_2}\nBest_score: {max_data}')
        
    
        
        

if __name__ == '__main__':
    train = './data/train.csv'
    test = './data/test.csv'
    
    model_checkpoints = ['r3ddkahili/final-complete-malicious-url-model',
                        'CrabInHoney/urlbert-tiny-v3-malicious-url-classifier',
                        'kmack/malicious-url-detection']
    
    train_df, test_df = load_data(train, test)
    
    for model_checkpoint in model_checkpoints:
        checkpoint_name = model_checkpoint.split('/')[1].replace('-', '_')
        predictor = model_fit(train_df, test_df, model_checkpoint, checkpoint_name)
    model_ensemble(model_checkpoints, test_df)