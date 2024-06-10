import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential

# TODO: [지시사항 1번] 첫번째 모델을 완성하세요.
def build_model1():
    model = Sequential()
    # 입력 단어 10개, 벡터 길이는 5
    model.add(layers.Embedding(
        input_dim = 10,
    output_dim = 5)
    )
    # simpleRNN hidden state 크기는 5
    model.add(layers.SimpleRNN(3))
    
    return model

# TODO: [지시사항 2번] 두번째 모델을 완성하세요.
def build_model2():
    model = Sequential()
    # Embedding layer: 입력 단어의 총 개수는 256개, 각 단어의 벡터 길이는 100
    model.add(layers.Embedding(
        input_dim = 256,
        output_dim = 100))
    # SimpleRNN layer: hidden state의 크기는 20
    model.add(
        layers.SimpleRNN(20)
        )
    # Dense layer: 노드 개수는 10개, 활성화 함수는 softmax
    model.add(layers.Dense(10, activation='softmax'))
    
    return model
    
def main():
    model1 = build_model1()
    print("=" * 20, "첫번째 모델", "=" * 20)
    model1.summary()
    
    print()
    
    model2 = build_model2()
    print("=" * 20, "두번째 모델", "=" * 20)
    model2.summary()

if __name__ == "__main__":
    main()
