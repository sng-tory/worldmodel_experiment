# SVLA Challenge Dacon Private Kit

실행 위치는 `svla_scoring_kit` 폴더입니다.

```bash
cd svla_scoring_kit
poetry install
```

운영진 데이터와 모델은 아래 위치에 복사합니다.

- 정답 영상: `../data/answers/videos/sample_000000.mp4`
- challenge action: `../data/challenge/actions/sample_000000.npy`
- challenge image: `../data/challenge/images/sample_000000.png`
- action extractor 체크포인트: `../checkpoints/action_extractor.ckpt`
- action normalization stats: `../stats/so100_action_statistics.json`
- public id 목록: `../csv/public_ids.txt`

정답 feature CSV 생성:

```bash
poetry run python scripts/eval/make_answer_feature_csv.py
```

참가자 제출 CSV 채점:

```bash
poetry run python scripts/eval/score_feature_csv.py \
  --submission-csv ../submissions/team_a/submission_features.csv
```

출력은 `public_score`, `private_score`, 전체 summary JSON입니다. 점수는 낮을수록 좋습니다.