# SVLA Challenge Participant Release

실행 위치는 `svla_challenge_kit` 폴더입니다.

```bash
cd svla_challenge_kit
poetry install
```

데이터와 체크포인트는 아래 위치에 복사합니다.

- 학습 데이터셋: `../data/so100_stride5/`
- 테스트 입력: `../data/challenge/images/`, `../data/challenge/actions/`
- backbone 체크포인트: `../checkpoints/model.ckpt`
- baseline diffusion 체크포인트: `../checkpoints/baseline_diffusion.ckpt`
- action extractor 체크포인트: `../checkpoints/action_extractor.ckpt`

학습 예시는 다음과 같습니다.

```bash
bash scripts/train.sh --config configs/train/svla_action_diffusion_11M.yaml --script scripts/train_diffusion.py
```

제출용 feature CSV 생성 예시는 다음과 같습니다.

```bash
poetry run python scripts/eval/make_submission_feature_csv.py \
  --checkpoint ../checkpoints/baseline_diffusion.ckpt \
  --output-csv ../outputs/submission_features.csv
```

이미 생성한 `sample_000000.mp4` 형식의 영상 폴더가 있으면 다음처럼 generation을 건너뜁니다.

```bash
poetry run python scripts/eval/make_submission_feature_csv.py \
  --skip-generation \
  --prediction-root ../outputs/predictions/videos \
  --output-csv ../outputs/submission_features.csv
```