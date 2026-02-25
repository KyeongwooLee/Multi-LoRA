# Evaluation Pipeline README

## 개요

`Evaluation/evaluation.py`는 질문/정답/모델 응답 데이터셋을 입력받아  
`gemini-2.5-flash-lite`(기본값)로 LLM-as-judge 평가를 수행합니다.

핵심 원칙:
- 평가 기준은 코드의 `default_criteria()`를 **고정 사용**
- `project_docs` 기반 기준 자동 생성 없음
- heuristic/offline 평가 없음 (Gemini만 사용)

---

## 평가 기준

`default_criteria()` 기준:

1. `reference_correctness` (0.5)
- 정답(reference)과 사실/논리 일치도

2. `explanation_clarity` (0.2)
- 설명의 명확성, 이해 용이성

3. `hallucination_control` (0.2)
- 정답에 없는 왜곡/과장/허위 정보 제어

4. `personalization_alignment` (0.1)
- style_label 등 개인화 스타일 정렬도

점수 체계:
- 항목별 기준 점수: `1~5`
- 종합 점수: `0~100`
- 공식: `overall_score = (sum(weight * score_1_to_5) / 5) * 100`
- 합격 기준: `overall_score >= 70.0`

---

## 입력 데이터셋 형식

배열(JSON Array) 또는 객체(JSON Object) 지원.

필수 필드:
- `question`
- `reference_answer`
- `model_answer`

선택 필드:
- `id` (없으면 자동 생성)
- `style_label`

예시:

```json
[
  {
    "id": "sample_001",
    "question": "질문",
    "reference_answer": "정답",
    "model_answer": "모델 응답",
    "style_label": "direct_instruction"
  }
]
```

---

## 실행 방법

환경 변수:
- `GENAI_API_KEY` 필요

실행:

```bash
GENAI_API_KEY=YOUR_KEY python3 Evaluation/evaluation.py \
  --dataset Evaluation/sample_eval_dataset.json \
  --output-dir Evaluation/outputs \
  --judge-model gemini-2.5-flash-lite \
  --temperature 0.0
```

---

## 출력 파일

기본 출력 경로: `Evaluation/outputs`

1. `criteria.json`
- 고정 평가 기준 정보

2. `item_results.json`
- 각 샘플별 기준 점수/이슈/개선 팁

3. `final_report.json`
- 메타데이터 + 기준 + 요약 통계 + 상세 결과

---

## 주요 옵션

- `--dataset`: 평가 데이터셋 경로
- `--output-dir`: 결과 저장 경로
- `--judge-model`: Gemini 모델명
- `--temperature`: Gemini 생성 온도
