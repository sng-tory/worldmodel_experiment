# INHA AI Participant Package

이 폴더는 참가자에게 전달할 파일만 모은 배포본입니다.
실행은 `challenge_kit` 폴더 안에서 합니다.

```bash
cd challenge_kit
poetry install
```

## 포함 내용

- 학습 코드: `challenge_kit/scripts/train_diffusion.py`
- 제출용 CSV 생성 코드: `challenge_kit/scripts/eval/make_submission_feature_csv.py`
- 모델/라이브러리 코드: `challenge_kit/src`, `challenge_kit/libs`, `shared_libs/video_utils`
- 학습 데이터 위치: `data/train/`
- 테스트 입력 위치: `data/eval/images/`, `data/eval/actions/`
- 모델 파라미터 위치: `checkpoints/`

## 체크포인트 파일명

- `checkpoints/backbone.ckpt`: pretrained backbone/base model
- `checkpoints/baseline_diffusion.ckpt`: baseline diffusion checkpoint
- `checkpoints/action_extractor.ckpt`: 제출 CSV feature 추출용 action extractor checkpoint

`action_extractor.ckpt`는 참가자가 새로 학습하는 모델이 아닙니다. `make_submission_feature_csv.py`가 이 checkpoint를 로드해서 제출 영상의 action feature/action_mae 값을 CSV에 넣습니다.

## diffusion 모델 학습 실행 예시

```bash
cd challenge_kit
poetry run bash scripts/train.sh \
  --config configs/train/inha_action_diffusion_11M.yaml \
  --script scripts/train_diffusion.py
```

## 제출용 feature CSV 생성

모델을 이용해서 생성 영상만들고 csv까지 생성 수행:

```bash
cd challenge_kit
poetry run python scripts/eval/make_submission_feature_csv.py \
  --output-csv ../outputs/submission_features.csv
```


## 제출 파일

최종 제출 파일은 아래 CSV입니다.

```text
outputs/submission_features.csv
```

## 추론/GPU 제한 안내

대회 운영 정책에 맞춰 추론 환경, GPU 수, 시간 제한을 별도로 고지하세요. 참가자는 동일한 `eval/images`, `eval/actions` 입력에서 `sample_id`별 영상을 생성해야 하며, 제출 CSV는 제공된 `make_submission_feature_csv.py`로 생성하는 것을 기준으로 합니다.