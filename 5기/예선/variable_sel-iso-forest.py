# ## 3. 모델 학습
# 

# ### 모델 정의
# 

# In[59]:


model = IsolationForest(n_estimators=100, contamination='auto', random_state=RANDOM_STATE)


# ### 모델 학습
# 

# In[60]:


features = []

for col in df_train.columns:
    try:
        df_train[col] = df_train[col].astype(int)
        features.append(col)
    except:
        continue

train_x = df_train[features]
train_y = df_train["target"]

model.fit(train_x, train_y)


# ### 검증

# In[85]:


# valid set target y
df_val_y = df_val.iloc[:, -1]

# valid set X
df_val_x = df_val[features]


# In[92]:


val_pred = model.predict(df_val_x)

# target type -1, 1 <-> abnormal, normal
val_pred_labels = np.where(val_pred == -1, 'AbNormal', 'Normal')

# Calculate accuracy
accuracy = accuracy_score(df_val_y, val_pred_labels)
print(f"Accuracy: {accuracy:.4f}")

# Calculate F1 score
f1 = f1_score(df_val_y, val_pred_labels, pos_label='AbNormal')  # Assuming 'AbNormal' is the positive class
print(f"F1 Score: {f1:.4f}")


# ## 4. 제출하기
# 

# ### 테스트 데이터 예측
# 

# 테스트 데이터 불러오기
# 

# In[61]:


test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))


# In[62]:


df_test_x = test_data[features]

for col in df_test_x.columns:
    try:
        df_test_x.loc[:, col] = df_test_x[col].astype(int)
    except:
        continue


# In[63]:


test_pred = model.predict(df_test_x)
test_pred


# In[65]:


# -1을 'AbNormal', 1을 'Normal'로 변환
test_pred_labels = np.where(test_pred == -1, 'AbNormal', 'Normal')
test_pred_labels


# In[66]:


test_pred_series = pd.Series(test_pred_labels)

test_pred_series.value_counts()


# ### 제출 파일 작성
# 

# In[67]:


# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
# df_sub["target"] = test_pred
df_sub["target"] = test_pred_labels
# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)
