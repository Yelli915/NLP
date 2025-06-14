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
        '[PAD]': 0,
        '[UNK]': 1,
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

    # input_ids 등 생성                  
    def encode(self, text, text_pair=None, max_length=32, padding=True, truncation=True):
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

        return input_ids, attention_mask, token_type_ids

    # tokenizer(text) 호출 시 동작         
    def __call__(self, text, text_pair=None, max_length=32, return_tensors='pt', padding=True, truncation=True):
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