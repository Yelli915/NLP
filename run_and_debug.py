# #bpe 아이디어 기반 vocab 생성.
from tokenizer import CustomTokenizer

# corpus = [
#     "나는 학교에 간다",
#     "학교에 다시 간다",
#     "오늘은 날씨가 좋다",
#     "나는 오늘 학교에 갔다"
# ]

# tokenizer = CustomTokenizer(vocab_size=50, tokenizer_type='bpe')
# tokenizer.train_bpe(corpus)

# for token, idx in tokenizer.vocab.items():
#     print(f"{idx:>2} : {token}")


# tok = tokenizer.tokenize("학교에 다시 간다.오늘은 날씨가 좋다.")
# print(tok)

from tokenizer import CustomTokenizer

def test_tokenizer(tokenizer_type):
    print(f"\n🧪 테스트 중: {tokenizer_type.upper()} 방식")
    
    corpus = [
        "나는 학교에 갔다.",
        "오늘은 날씨가 좋다.",
        "학교에 다시 갔다.",
        "나는 오늘 학교에 갔다."
    ]
    
    test_text = "나는 오늘 학교에 갔다."
    
    tokenizer = CustomTokenizer(vocab_size=50, tokenizer_type=tokenizer_type)
    
    if tokenizer_type == 'freq':
        tokenizer.train_freq(corpus)
    elif tokenizer_type == 'okt':
        tokenizer.train_okt(corpus)
    elif tokenizer_type == 'bpe':
        tokenizer.train_bpe(corpus)
    
    print("📘 Vocab:")
    for token, idx in tokenizer.vocab.items():
        print(f"{idx:>2} : {token}")
    
    input_ids, attention_mask, token_type_ids = tokenizer.encode(test_text, max_length=20)
    print("\n✅ encode 결과:")
    print("input_ids      :", input_ids)
    print("attention_mask :", attention_mask)
    print("token_type_ids :", token_type_ids)
    
    inputs = tokenizer(test_text, max_length=20, return_tensors='pt')
    print("✅ __call__() 결과:")
    for k, v in inputs.items():
        print(f"{k:15}: {v}")
    
    print("✅ decode 결과:")
    print(tokenizer.decode(input_ids))

# 세 방식 모두 실행
for mode in ['freq', 'okt', 'bpe']:
    test_tokenizer(mode)

