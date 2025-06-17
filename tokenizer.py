import json
from collections import Counter
from konlpy.tag import Okt
import torch

class CustomTokenizer:
    # vocab, 특수 토큰 등 초기화
    def __init__(self, vocab_size=3000, tokenizer_type ='freq'):
        self.vocab_size = vocab_size
        self.vocab={}
        self.special_tokens_map = {
        '[UNK]': 0,
        '[PAD]': 1,
        '[CLS]': 2,
        '[SEP]': 3,
        '[MASK]': 4,
        }

        self.special_tokens = list(self.special_tokens_map.keys())

        for token, idx in self.special_tokens_map.items():
            self.vocab[token]=idx

        #어떤 방식으로 vocab을 생성할건지 설정. 기본값 = 빈도로 설정. option으로 설정.
        self.tokenizer_type = tokenizer_type
           
    # 단순 단어 빈도 기반 vocab 생성           
    def train_freq(self, corpus):
        """빈도 기반 vocab 생성"""
        
        # 1. 토큰의 빈도수 count
        count = Counter()
        for line in corpus:
            tokens = self.tokenize(line) # text.split()
            count.update(tokens)

        # 2. 자주 등장한 토큰을 vocab_size 에 맞게 추가.
        num_add = self.vocab_size - len(self.special_tokens)
        for token, _ in count.most_common(num_add): #most_common(num_add)는 가장 많이 등장한 토큰 n개를 뽑는 함수.
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        print(f"[train_freq] vacab 생성 완료. 총 {len(self.vocab)}개 토큰 등록.")

    def train_bpe(self, corpus):
        """BPE 원리 기반 서브워드 vocab 생성"""

        # 1. 초기 서브워드 카운트 (공백 있는 버전으로 저장)
        subword_counter = Counter()
        for line in corpus:
            tokens = self.tokenize(line)  # e.g., ['나', '는', '학', '교', '에', '</w>']
            joined = ' '.join(tokens)     # '나 는 학 교 에 </w>'
            subword_counter[joined] += 1

        # 2. BPE 병합 반복
        merges = []
        num_add = self.vocab_size - len(self.vocab)
        while len(self.vocab) < self.vocab_size and len(merges) < num_add:
            # 빈도 높은 쌍 탐색
            pair_counter = Counter()
            for token_seq, freq in subword_counter.items():
                symbols = token_seq.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counter[pair] += freq

            if not pair_counter:
                break

            best_pair = pair_counter.most_common(1)[0][0]
            merges.append(best_pair)

            # 병합 적용
            new_counter = Counter()
            pattern = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            for token_seq, freq in subword_counter.items():
                new_token_seq = token_seq.replace(pattern, replacement)
                new_counter[new_token_seq] += freq
            subword_counter = new_counter

            # vocab 추가
            if replacement not in self.vocab:
                self.vocab[replacement] = len(self.vocab)

        print(f"[train_bpe] vocab 생성 완료. 총 {len(self.vocab)}개 토큰 등록.")
        
    def train_okt(self, corpus):
        """형태소 분석기(Okt) + 빈도수에 기반한 vocab 생성."""
        okt = Okt()
 
        # 1. 형태소 분석 기반 토큰 카운트
        count = Counter()
        for line in corpus:
            tokens = self.tokenize(line)
            count.update(tokens)
        
        # 2. 자주 사용하는 토큰을 vocab에 추가(빈도 기반 추가)
        num_add = self.vocab_size - len(self.special_tokens)
        for token , _ in count.most_common(num_add):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        print(f"[train_okt] vocab 생성 완료. 총 {len(self.vocab)}개 토큰 등록.")

    # 문장 토큰화                     
    def tokenize(self, text):
        if self.tokenizer_type == 'freq':
            return text.strip().split()

        elif self.tokenizer_type == 'okt':
            from konlpy.tag import Okt
            okt = Okt()
            return okt.morphs(text)

        elif self.tokenizer_type == 'bpe':
            tokens = []
            for word in text.strip().split():
               chars = list(word) + ['</w>']
               tokens.extend(chars)
            return tokens

        else:
            raise ValueError(f"[tokenize] 알 수 없는 tokenizer_type: {self.tokenizer_type}")

    def apply_bpe_merges(self, tokens):
        merged = tokens[:]
        made_merge = True
        
        while made_merge:
            made_merge = False
            
            for i in range(len(merged) - 1):
                pair = merged[i] + merged[i + 1]
                
                if pair in self.vocab:
                    merged = merged[:i] + [pair] + merged[i + 2:]
                    made_merge = True
                    break
                
        return merged


             
    def encode(self, text, text_pair=None, max_length=32, padding=True, truncation=True):
        """"
        주어진 텍스트를 BERT 모델 입력 형식 (input_ids, attention_mask, token_type_ids)으로 인코딩한다.

        Args:
            text (str): 입력 텍스트. 하나 이상의 문장으로 구성될 수 있음.
            text_pair (str, optional): 두 번째 문장. 기본값은 None. 사용하지 않으면 단일 문장 처리됨.
            max_length (int): 출력 시퀀스의 최대 길이. 기본값은 32.
            truncation (bool): max_length보다 길 경우 자를지 여부. 기본값은 True.

        Returns:
            Tuple[List[int], List[int], List[int]]:
                - input_ids: 토큰을 vocab 인덱스로 매핑한 리스트.
                - attention_mask: 실제 토큰은 1, 패딩된 토큰은 0으로 표시.
                - token_type_ids: 문장 구분을 위한 타입 인덱스. 현재 모두 0으로 구성됨.

        Notes:
            - BPE 토크나이저를 사용하는 경우, 토큰화 후 merge 규칙이 적용된다.
            - 입력 텍스트는 '[CLS]'로 시작하고 각 문장은 '[SEP]'로 끝난다.
            - text_pair가 있을 경우 두 번째 문장도 '[SEP]'로 끝나며 token_type_id는 1로 설정된다.
            - vocab에 없는 토큰은 '[UNK]'로 처리된다.
            - 패딩은 '[PAD]'로 채워지며 attention_mask에서 0으로 처리된다.

        Preconditions:
            - self.vocab은 학습이 완료되어 있어야 함.
            - self.tokenizer_type은 'bpe', 'freq', 'okt' 중 하나여야 함.
            - text는 str 타입이어야 하며 None이 아니어야 함.

        Postconditions:
            - 리턴되는 세 개의 리스트 길이는 모두 max_length와 동일함.
            - 인코딩된 input_ids에는 반드시 '[CLS]'(2)와 '[SEP]'(3) 토큰이 포함되어야 함.
            - attention_mask는 input_ids에서 실제 토큰은 1, 패딩은 0으로 표시됨.
            - token_type_ids는 현재는 모든 문장에 대해 0으로 고정됨.
        """
        # 1. 토큰화
        tokens_a = self.tokenize(text)
        tokens_b = self.tokenize(text_pair) if text_pair else []
        
        # 1.1 BPE 병합 적용
        if self.tokenizer_type == 'bpe':
            tokens_a = self.apply_bpe_merges(tokens_a)
            if text_pair:
                tokens_b = self.apply_bpe_merges(tokens_b)
                

        # 2. [CLS], [SEP] 추가
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        token_type_ids = [0] * len(tokens)

        if text_pair:
            tokens += tokens_b + ['[SEP]']
            token_type_ids += [1] * (len(tokens_b) + 1)

        # 3. 토큰 → ID (vocab에 없으면 [UNK])
        input_ids = [self.vocab.get(tok, self.vocab['[UNK]']) for tok in tokens]

        # 4. attention_mask
        attention_mask = [1] * len(input_ids)

        # 5. 길이 맞추기
        if padding and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.vocab['[PAD]']] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len
            
        elif truncation:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
            
        else:
            raise ValueError("길이 초과: truncation=False일 때 max_length보다 긴 입력입니다.")

        return tokens, input_ids, attention_mask, token_type_ids

    # tokenizer(text) 호출 시 동작         

    def __call__(self, text, text_pair=None, max_length=32, return_tensors='pt', padding=True, truncation=True):
        """
        텍스트를 BERT 입력 형식으로 인코딩하고, 딕셔너리 형태로 반환한다.
        tokenizer(text)처럼 바로 호출할 수 있도록 정의된 special method이다.

        Args:
            text (str): 입력 문장.
            text_pair (str, optional): 두 번째 문장. 문장쌍 입력이 필요한 경우 사용.
            max_length (int): 출력 길이 제한. 기본값은 32.
            return_tensors (str): 반환 타입. 'pt'는 PyTorch 텐서 형태로 반환. 'np' 또는 'list'는 미지원 (현재는 'pt'만 지원).

        Returns:
            dict: 다음 key를 포함한 딕셔너리 반환
                - 'input_ids': torch.LongTensor of token ids
                - 'attention_mask': torch.LongTensor (1은 실제 토큰, 0은 패딩)
                - 'token_type_ids': torch.LongTensor (문장 구분용, 현재 두 번째 문장이 있을 경우 1로 설정됨)

        Notes:
            - 내부적으로 self.encode()를 호출하여 토큰 인덱스를 생성한다.
            - return_tensors가 'pt'인 경우 PyTorch 텐서로 변환한다.
            - 향후 TensorFlow, NumPy 대응은 확장 가능.

        Preconditions:
            - self.encode() 함수가 정상 작동해야 함.
            - text는 str 타입이어야 하며 None이 아니어야 함.
            - vocab이 사전에 학습되어 있어야 함.

        Postconditions:
            - 반환되는 딕셔너리는 3개의 key ('input_ids', 'attention_mask', 'token_type_ids')를 포함.
            - 각 값은 동일한 길이의 torch.Tensor로 구성됨.
        """
        #input_ids, attention_mask, token_type_ids = self.encode(text, text_pair, max_length)
        input_ids, attention_mask, token_type_ids = self.encode(
            text, text_pair, max_length=max_length, padding=padding, truncation=truncation
        )
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask]),
                'token_type_ids': torch.tensor([token_type_ids])
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

    # 숫자 → 텍스트 복원      
    def decode(self, input_ids):
        """
        숫자 ID 시퀀스를 원래 텍스트로 디코딩한다.

        Args:
            input_ids (List[int]): 토큰 ID 리스트. 일반적으로 모델 출력이나 tokenizer.encode() 결과.

        Returns:
            str: 복원된 자연어 문장.

        Process:
            1. input_ids를 토큰 문자열로 변환 (ID → 토큰)
            2. [PAD], [CLS], [SEP], [UNK] 등 특수 토큰 제거
            3. BPE 토크나이저일 경우, </w>로 끝나는 서브워드들을 공백 기준으로 재결합
            4. freq 또는 okt 방식일 경우, 단순 공백 연결

        Notes:
            - BPE의 경우 </w>는 서브워드 경계를 의미하며, 이를 기준으로 단어 경계를 복원함.
            - vocab이 존재하지 않거나, 잘못된 input_ids가 들어오면 [UNK]로 복원될 수 있음.

        Preconditions:
            - self.vocab은 학습되어 있어야 하며, input_ids 내 ID들은 해당 vocab에 존재하거나 예외처리 되어야 함.
            - self.special_tokens는 [PAD], [CLS], [SEP], [UNK], [MASK] 등을 포함해야 함.

        Postconditions:
            - 출력 문자열은 사람이 읽을 수 있는 형태의 문장임 (완전한 원문 복원은 아님).
        """
        # 1. ID → 토큰
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(i, '[UNK]') for i in input_ids]

        # 2. 특수 토큰 제거
        tokens = [t for t in tokens if t not in self.special_tokens]

        # 3. BPE 처리
        if self.tokenizer_type == 'bpe':
            text = ''
            for token in tokens:
                if token.endswith('</w>'):
                    text += token.replace('</w>', '') + ' '
                else:
                    text += token
            return text.strip()

        else:  # freq / okt
            return ' '.join(tokens).strip()
        
    #실험 및 재현을 위한 저장
    def save_vocab(self, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"[save_vocab] vocab 저장 완료 -> {file_path}")

    def load_vocab(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        print(f"[load_vocab] vocab 로드 완료 -> {file_path}")