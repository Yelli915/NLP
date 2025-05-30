#bpe 아이디어 기반 vocab 생성.
from tokenizer import CustomTokenizer

corpus = [
    "나는 학교에 간다",
    "학교에 다시 간다",
    "오늘은 날씨가 좋다",
    "나는 오늘 학교에 갔다"
]

tokenizer = CustomTokenizer(vocab_size=50, tokenizer_type='bpe')
tokenizer.train_bpe(corpus)

for token, idx in tokenizer.vocab.items():
    print(f"{idx:>2} : {token}")
