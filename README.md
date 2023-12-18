### Project Title

NLP 혐오 표현 분류 프로젝트 

### Overview

- 기간  |  2021. 08 ~ 2021. 08
- 팀원  |  개인 프로젝트
- 플랫폼 |  Python, Pytorch, Colab notebook

### Background 

Pytorch 프레임워크로 NLP 프로젝트
Tokenization과 Embedding 방법론을 비교하고, CNN과 BERT 모델 구현, KoBERT 사용까지 NLP 프로세스를 단계적으로 밟아 나갈 예정임

### Goal

1. Tokenization : 한국어 Tokenizer인 Konlpy의 Mecab과 Subword Tokenizer로써 Huggingface의 Wordpiece Tokenizer를 CNN 모델에 적용하여 각각의 성능을 비교
2. Embedding : Word2vec, Fasttext, Glove의 개념을 익힌 뒤, Pre-trained Embedding을 CNN 모델에 적용한 결과를 비교
3. BERT : 논문 <Pre-training of Deep Bidirectional Transformers for Language Understanding>를 토대로 Bert 모델을 구현하고 성능을 확인
4. KoBERT : SKT Brain의 한국어 BERT 모델인 KoBERT를 사용하여 최종 성능을 평가

### Dataset

한국어 혐오 표현 데이터셋 : 7,896 train data/471 validation data/974 test data로 구성되어 있으며, 
Normal(혐오표현 없음)/Offensive(공격적 표현)/Hate(혐오표현) 총 3개로 라벨링 되어 있음

### Theories

1.  분류 문제에 있어 CNN 모델은 기준 모델보다 성능이 10% 이상 높을 것이다
2.  Mecab보다 Wordpiece Tokenizer를 사용한 모델의 성능이 5% 이상 높을 것이다
3.  Pre-trained Embedding을 사용한 결과가 그렇지 않은 결과보다 성능이 10% 이상 높을 것이다
4.  BERT 모델은 CNN 모델보다 성능이 10% 이상 높을 것이다
5.  KoBERT 모델은 F1 Score 0.8 이상을 달성할 것이다

### EDA

- Train set은 Normal(0)이, Validation set은 Offensive(1)의 수가 많으며, 클래스의 분포가 불균형하므로 F1 Score를 평가지표로 사용
<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/d42f3ed5-32e7-478f-802d-c90881078bf6" alt="text" width="number" />
</p>

### Base Model

- TD-IDF Vectorization을 사용한 Multinomial naive bayes 및 SVM 모델

<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/8b3fd7ea-cdf4-4ed0-84a3-ca8c332ab7f4" alt="text" width="number" />
</p>

   
### Tokenization

1. 한국어 자연어 처리를 위한 라이브러리인 Konlpy의 Mecab과, Mecab 전처리 후 Subword Tokenizer인 Huggingface의 Bertwordpiece Tokenizer를 사용하여 성능 비교 
2. Subword Tokenizer를 사용할 경우 '나는 아침밥을 먹었다.'라는 문장에서 '아침밥'을 '아침'과 '밥'으로 분절할 수 있어, OOV(Out Of Vocabulary) 문제를 해결할 수 있음

### Word Embedding

1. 기준이 되는 Embedding은 Pytorch에서 제공하는 Embedding Layer이며, Word2Vec, Fasttext, Glove의 한국어 Pre-trained Embedding을 사용하여 성능을 비교할 것임
2. Pre-trained Embedding
  - Word2Vec : 분산 표현(Distributed Representation)의 일종으로 단어의 의미를 여러 차원으로 분산하여 단어간 유사도를 계산할 수 있으며, CBOW와 Skip-gram 방법론으로 나뉜다.
  - Fasttext : 내부 단어(subword)를 고려하여 각 단어를 글자 단위 n-gram의 구성으로 취급하기 때문에 OOV 문제도 해결할 수 있으며, 이러한 장점 때문에 한국어 Embedding에 잘 활용됨
  - Glove :  LSA의 메커니즘인 카운트 기반의 방법과 Word2Vec의 메커니즘인 예측 기반의 방법론 두 가지를 모두 사용하는 방법
  
### CNN Model

1. 문장은 Embedding Layer를 거쳐 행렬로 변환이 됨
2. 커널은 사이즈가 4인 커널 2개, 3인 커널 2개, 2인 커널 2개를 사용(총 6개)
3. 각 Region 마다 2개의 Feature map을 형성
4. Maxpooling 후 Softmax를 거쳐 3 Classes의 Ouput 출력

<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/934838a3-baac-493c-b2f5-7a13d7255dcd" alt="text" width="number" />
</p>

### CNN Text Classification 결과

1. Pre-trained Embedding을 적용하지 않은 기준으로, Wordpiece Tokenizer를 사용한 CNN의 모델의 성능은 그렇지 않은 모델과 비슷한 성능을 보임
2. Mecab Tokenizer를 사용한 CNN 모델의 성능은 Glove Embedding을 사용했을 때의 성능이 가장 높음
3. Wordpiece Tokenizer를 사용한 모델의 성능은 Fasttext를 사용했을 때 가장 높은 성능을 보임

<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/b47f4dbf-ae0e-4beb-bc54-d9aeb2aa143a" alt="text" width="number" />
</p>

### BERT Concept

1. Transformer의 Encoder 부분만을 사용
2. 사전 훈련 Embedding을 통해 목표의 성능을 올리는 모델
3. 대량 말뭉치를 토대로 BERT 인코더에서 Embedding 한 후, 데이터에 맞게 Fine Tuning하여 Task를 수행
4. Fine Tuning 과정에서 복잡한 모델을 사용하지 않고, 간단한 DNN 모델로도 좋은 성능을 내는 것으로 알려짐
       
<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/3611c0a3-187d-46c7-b696-25db229e7ce8" alt="text" width="number" />
</p>

### BERT Modeling

1.  Token Embedding : Word Piece Embedding 방식을 사용하여 OOV 문제를 해결
2.  Segment Embedding : 두개의 문장을 구분자([SEP])를 넣어 구분하고 그 두 문장을 하나의 Segment로 지정하여 입력
3.  Position Embedding : Token의 위치정보 Embedding
   
<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/9f429239-70bd-48ee-93b1-c3da7b05fcb9" alt="text" width="number" />
</p>


### Pre-training

1.  MLM(Masked Language Model) : [MASK]된 단어를 예측하는 방식이며, 전체 단어의 15%를 선택한 후 그중 80%는 [MASK], 10%는 현재 단어 유지, 10%는 임의의 단어로 대체
2. NSP(Next Sentence Prediction) :  첫 번째([CLS]) Token으로 문장 A와 문장 B의 관계를 예측
3. 소스코드에서는 각각 projection_lm, projection_cls로 표현
4. RAM 문제로 100000개의 데이터만으로 소량 사전학습을 진행
5. 10 epochs Traing 결과 loss 약 6.7

<figure class="half">  
  <a href="link"><img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/b549f2e0-7c34-470b-8530-20a10d546e3a"></a>  
  <a href="link"><img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/ef83fc41-7258-43b2-becf-364de4a5515a"></a>  
</figure>
<p align="center">
  <img src="https://github.com/mugan1/pytorch_hate_speech/assets/71809159/7b27d75c-0d8f-4279-b1f3-f117fd4ef5bf" alt="text" width="number" />
</p>


