## DQACR

Dailogue-based multiple choice QA task에서 Commonsense를 고려하여 문제를 푸는 language model이다.

## Quick Links

  - [DQACR이란?](#what_is_dqacr)
  - [데이터 셋(Dataset)](#dataset)
  - [실험 결과](#result)
  - [Environment Setting](#environment_setting)
  - [Hyperparameter](#hyperparameter)
  - [Train](#train)
  - [Evaluation](#evaluation)

## DQACR이란?

기존 Pre-trained Language Model을 활용한 dialogue-based Multiple-choice QA task에는 치명적인 문제점 2개가 존재한다.
* input sequence가 일정 길이보다 길 경우, 뒷 부분의 dialogue history를 truncate한다.
* 문제를 푸는 과정에서 commonsense를 잘 고려하지 못한다. 

우리는 Semantic Search와 Continual Learning을 사용하여 위의 문제점을 보완한 Dialogue-based Multiple-choice QA agent를 만들었다.

## 데이터 셋(Dataset)
Dialogue-based Multiple-choice QA dataset으로 [DREAM](https://dataset.org/dream/) 을 사용하였다.
DREAM속 일부 문제는 commonsense reasoning을 요구하고, 둘 이상 문장의 reasoning을 요구한다.

우리는 다음과 같은 과정으로 하나의 데이터를 처리하였다.
1. [ConceptNet](https://conceptnet.io/)에서 Semantic Search를 통해 각 candidate answer과 가장 관련성이 높은 sentence를 찾는다.
2. Question과 각 dialogue utterance를 가지고 Semantic Search를 하여 maximum sequence length가 될 때까지 relevant utterance만 선택하여 input에 넣는다.
   1. 이때 dialogue contextual flow를 유지하기 위해 각 utterance는 순서를 유지하였다.
3. 하나의 candidate answer에 대해 input은 다음과 같아짐
   1. `[CLS] ConceptNet information [SEP] selected dialogue history [SEP] question candidate_answer [SEP]`

관련 내용은 `DQACR_preprocess_data.py`에 있다.

## 실험 결과

| Model                    | Accuracy |
|:-------------------------|:--------:|
| ALBERT-xxlarge(baseline) |   88.5   |
| ALBERT-xxlarge(DQACR)    |  90.05   |

## Environment Setting

아래와 같은 라이브러리를 사용하여 실험을 진행하였다.

| Module                | Version |
|:----------------------|:-------:|
| transformers          | 4.17.0  |
| torch                 |  1.7.1  |
| tqdm                  | 4.63.0  |
| datasets              | 1.18.3  |
| attrdict              |  2.0.1  |
| scikit-learn          |  1.0.2  |
| sentence_transformers |  2.2.0  |

## Hyperparameter

아래와 같은 하이퍼파라미터를 사용하여 실험을 진행하였다.

| Hyperparameter | Value |
|:---------------|:-----:|
| Batch size     |   9   |
| Epoch          |   3   |
| Weight decay   | 0.01  |
| Learning rate  | 1e-5  |

## Train

* DQACR의 경우 config.json에서 `model`을 DQACR로 설정해주면 된다.
* Baseline(ALBERT-xxlarge)의 경우 config.json에서 model을 DQACR이외의 문자열로 지정하면 된다.
* `train_and_inference.py` 실행시 `--mode train`으로 지정해주거나 따로 argparase 인자를 지정하지 않으면 된다. (default =  train)

 `python train_and_inference.py --mode train`

## Evaluation
* `train_and_inference.py` 실행시 `--mode`를 `train`이외의 것으로 지정해주면 된다. 
  * 하지만 명확함을 위해 `inference`나 `test`로 하는 것을 추천한다.
