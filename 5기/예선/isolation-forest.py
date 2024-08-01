'''
데이터를 공개할 수 없어 모델링파트 코드만 작성했습니다.
'''

import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#%%

model = IsolationForest(n_estimators=100, contamination='auto', random_state=RANDOM_STATE)

#%%

features = []

for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue

if "Set ID" in features:
    features.remove("Set ID")

train_x = df_train[features]
train_y = df_train["target"]

model.fit(train_x, train_y)

#%%

df_test_y = pd.read_csv(os.path.join("submission.csv"))

#%%

df_test = pd.merge(X, df_test_y, "inner", on="Set ID")
df_test_x = df_test[features]

for col in df_test_x.columns:
    try:
        df_test_x.loc[:, col] = df_test_x[col].astype(int)
    except:
        continue

#%%
    
test_pred = model.predict(df_test_x)
test_pred

# -1을 'AbNormal', 1을 'Normal'로 변환
test_pred_labels = np.where(test_pred == -1, 'AbNormal', 'Normal')
test_pred_labels
#%%

# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["target"] = test_pred_labels
# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)
