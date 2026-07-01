# DACON Organizer Package

이 폴더는 데이콘/운영진에게 전달할 채점용 최소 패키지입니다.
운영진은 참가자 제출 CSV와 운영진이 전달받은 정답 CSV를 비교해서 public/private 점수만 산출합니다.

## 폴더 구조

```text
dacon/
  scoring_kit/
    requirements.txt
    scripts/eval/score_feature_csv.py

  private_csv/
    answer_features.csv
    public_ids.txt

  submissions/
    team_name/submission_features.csv

  outputs/
```

## 설치 및 실행

가상환경을 만든 뒤 `requirements.txt`만 설치하면 됩니다.

```bash
cd scoring_kit
pip install -r requirements.txt
python scripts/eval/score_feature_csv.py \
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

## 운영진이 넣어야 하는 파일

```text
submissions/<team_name>/submission_features.csv
```

`answer_features.csv`는 사전에 생성해서 데이콘에 전달합니다.

## public/private 설정 방식

`private_csv/public_ids.txt`에 public score로 계산할 sample_id를 한 줄에 하나씩 적습니다.
숫자만 적어도 자동으로 `sample_000001` 형식으로 변환됩니다.

```text
0
1
2
sample_000033
```

- `public_ids.txt`에 있는 sample만 public score에 사용합니다.
- 나머지 common sample은 private score에 사용합니다.

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
