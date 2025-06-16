하이퍼 파라미터 : vocab_size , tokenizer_type , max_length


# Custom Tokenizer Project (한국어 텍스트용)

이 프로젝트는 한국어 텍스트 전처리를 위한 커스텀 토크나이저를 구현한 것입니다. 
기본적으로 HuggingFace 스타일을 따르며, 다양한 토크나이징 방식과 vocab 학습 기능을 포함합니다.

---

##  주요 기능

1. **토크나이징 방식 지원**
   - `freq`: 띄어쓰기 기준 단어 단위 토큰화
   - `okt`: Konlpy의 Okt 형태소 분석기 기반 토큰화
   - `bpe`: Byte Pair Encoding 방식의 서브워드 기반 토큰화

2. **Vocab 학습**
   - `train_freq()`: 빈도 기반 토큰 사전 생성
   - `train_okt()`: 형태소 기반 토큰 사전 생성
   - `train_bpe()`: BPE 병합 규칙 기반 서브워드 사전 생성

3. **특수 토큰 관리** __init__
   - `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` 자동 포함

4. **Vocab 저장 및 불러오기** (필요 시 run_and_debug 에서 사용.)
   - JSON 형식으로 vocab을 저장 및 재활용 가능

---

##  현재까지 완료된 작업

🔹 핵심 클래스 및 기능 구현
- CustomTokenizer 클래스 전체 구조 작성됨
- __init__: 특수 토큰 포함한 초기화 구현 완료 (vocab_size, tokenizer_type 사용)
- tokenize(): freq, okt, bpe 방식 각각 구현
- train_freq(), train_okt(), train_bpe() 각 방식에 따른 vocab 학습 구현 완료
- apply_bpe_merges(): BPE 병합 방식 구현
- encode(): 텍스트를 input_ids, attention_mask, token_type_ids로 변환 완성
- __call__(): HuggingFace 스타일의 callable 구현 완료 (return_tensors 포함)
- decode(): input_ids → 텍스트 복원 기능 구현 완료
- save_vocab() / load_vocab(): vocab 저장 및 불러오기 구현 완료

🔹 테스트 코드
- run_and_debug.py에서 freq, bpe 방식에 대한 테스트 성공적으로 수행됨
- encode, __call__, decode 결과 출력 확인 가능
- okt 주석 처리 된 것을tokenize() 및 train_okt()는 코드상 구현 완료



## 🔧 향후 개발 과제

- 토크나이저 테스트 자동화
- KoBERT 프로젝트 연동
- 03 _Custom_KoBERT_Tokenizer_Project.ipynb , 04_Custom_KoBERT_Fine_Tuning_Project.ipynb를 연결 후 우리 tokenizer로 매끄럽게 이어지는지 확인.

---

우선 주어진 각자의 과업은 tokenizer.py 에서 코드를 작성하시고, run_and_debug.ipynb에서 직접 테스트 해보는 방법으로 진행하시면 될 듯합니다.
