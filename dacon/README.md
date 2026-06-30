# DACON Organizer Package

이 폴더는 데이콘/운영진에게 전달할 비공개 채점용 파일만 모은 배포본입니다.
실행은 `scoring_kit` 폴더 안에서 합니다.

```bash
cd scoring_kit
poetry install
```

## 포함 내용

- 정답 feature CSV 위치: `private_csv/answer_features.csv`
- public/private split 설정: `private_csv/public_ids.txt`
- 최종 채점 코드: `scoring_kit/scripts/eval/score_feature_csv.py`
- 정답 CSV 재생성 코드: `scoring_kit/scripts/eval/make_answer_feature_csv.py`
- feature extractor 공통 코드: `scoring_kit/scripts/eval/feature_csv_utils.py`

## 정답 CSV 생성 또는 재생성

운영진이 정답 영상에서 feature CSV를 다시 만들 때만 사용합니다.

```bash
cd scoring_kit
poetry run python scripts/eval/make_answer_feature_csv.py
```

이때 내부적으로 DINO/FVD feature와 action extractor 기반 `action_mae` scalar를 저장합니다.

## 최종 채점

참가자가 제출한 `submission_features.csv`를 `submissions/<team_name>/submission_features.csv`에 둔 뒤 실행합니다.

```bash
cd scoring_kit
poetry run python scripts/eval/score_feature_csv.py \
  --submission-csv ../submissions/team_name/submission_features.csv
```

터미널 출력은 아래 두 줄만 나옵니다.

```text
public_score=...
private_score=...
```

상세 결과를 파일로 남기려면 다음 옵션을 추가합니다.

```bash
  --details-csv ../outputs/team_name_details.csv \
  --summary-csv ../outputs/team_name_summary.csv
```

## public/private 설정 방식

`private_csv/public_ids.txt`에 public score로 계산할 sample_id를 한 줄에 하나씩 적습니다.

```text
sample_000000
sample_000001
```

- `public_ids.txt`에 있는 sample만 public score에 사용합니다.
- 나머지 common sample은 private score에 사용합니다.
- 숫자만 적어도 `sample_000001` 형식으로 자동 변환됩니다.

## 점수 산식 요약

점수는 낮을수록 좋습니다.

```text
score = 0.4 * DINO_component + 0.3 * FVD_component + 0.3 * Action_component
```

Action Error Ratio는 split별 평균 MAE 기준입니다.

```text
AER = mean(submission_action_mae) / mean(answer_action_mae)
Action_component = clamp((AER - 1) / 4, 0, 1)
```

## 제한/운영 안내

참가자에게는 `private_csv`, `private_data`, `private_checkpoints`, `scoring_kit/scripts/eval/score_feature_csv.py`를 제공하지 않는 것을 권장합니다. 추론 방식, GPU 수, 외부 데이터 사용 가능 여부, 시간 제한은 대회 공지 문서에서 별도로 고지해야 합니다.