# INHA AI Participant Package

이 폴더는 참가자에게 전달할 파일만 모은 배포본입니다.
실행은 `challenge_kit` 폴더 안에서 합니다.

```bash
cd challenge_kit
poetry install
```

## 포함 내용

- 학습 코드: `challenge_kit/scripts/train_diffusion.py`, `challenge_kit/scripts/train_action_extractor.py`
- 전처리 코드: `challenge_kit/scripts/preprocess_so100_stride.py`
- 제출용 CSV 생성 코드: `challenge_kit/scripts/eval/make_submission_feature_csv.py`
- 모델/라이브러리 코드: `challenge_kit/src`, `challenge_kit/libs`, `shared_libs/video_utils`
- 학습 데이터 위치: `data/so100_stride5/`
- 테스트 입력 위치: `data/challenge/images/`, `data/challenge/actions/`
- 모델 파라미터 위치: `checkpoints/`

## 체크포인트 파일명

- `checkpoints/backbone.ckpt`: pretrained backbone/base model
- `checkpoints/baseline_diffusion.ckpt`: baseline diffusion checkpoint
- `checkpoints/action_extractor.ckpt`: feature CSV 생성용 action extractor checkpoint

## 학습 실행 예시

```bash
cd challenge_kit
poetry run bash scripts/train.sh \
  --config configs/train/inha_action_diffusion_11M.yaml \
  --script scripts/train_diffusion.py
```

## 제출용 feature CSV 생성

생성까지 함께 수행할 때:

```bash
cd challenge_kit
poetry run python scripts/eval/make_submission_feature_csv.py \
  --output-csv ../outputs/submission_features.csv
```

이미 `sample_000000.mp4` 형식의 예측 영상을 만든 경우:

```bash
cd challenge_kit
poetry run python scripts/eval/make_submission_feature_csv.py \
  --skip-generation \
  --prediction-root ../outputs/predictions/videos \
  --output-csv ../outputs/submission_features.csv
```

## 제출 파일

최종 제출 파일은 아래 CSV입니다.

```text
outputs/submission_features.csv
```

## 추론/GPU 제한 안내

대회 운영 정책에 맞춰 추론 환경, GPU 수, 시간 제한을 별도로 고지하세요. 참가자는 동일한 `challenge/images`, `challenge/actions` 입력에서 `sample_id`별 영상을 생성해야 하며, 제출 CSV는 제공된 `make_submission_feature_csv.py`로 생성하는 것을 기준으로 합니다.