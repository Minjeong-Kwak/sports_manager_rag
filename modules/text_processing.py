import tiktoken
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API 키가 로드되지 않았습니다. .env 파일을 확인하세요.")

# OpenAI API 설정
client = OpenAI(api_key=OPENAI_API_KEY)

# 임베딩 모델 설정
EMBEDDING_MODEL = "text-embedding-3-small"

# OpenAI 토큰 인코더 로드
encoding = tiktoken.get_encoding("cl100k_base")

# 텍스트 정제 함수 (중복 제거, 특수문자 제거, 의미 단위 정리 등)
def clean_text(text):
    text = re.sub(r"[\t\r]+", " ", text)                     # 탭/캐리지리턴 제거
    text = re.sub(r"\s{2,}", " ", text)                       # 이중 공백 제거
    text = re.sub(r'["\'*\[\]<>▶◆■●▪→⇒①②③④⑤⑥⑦⑧⑨⑩]', "", text)  # 불필요한 기호 제거
    text = re.sub(r"\n+", " ", text)                         # 줄바꿈 제거
    text = re.sub(r"\s+", " ", text)                         # 공백 정리
    return text.strip()

# 텍스트를 청크로 나누는 함수 (오버랩 적용)
def chunk_text(questions, answers, general_texts, max_length=300, overlap=50):
    question_chunks = []
    question_answer_pairs = []
    general_chunks = []

    def tokenize_and_chunk(text):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(encoding.encode(word))
            if current_length + word_length > max_length:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(encoding.encode(chunk_text)) > 10:  # 최소 10토큰 이상만 유지
                        chunks.append(chunk_text)
                current_chunk = current_chunk[-overlap // 2:] + [word]
                current_length = sum(len(encoding.encode(w)) for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(encoding.encode(chunk_text)) > 10:
                chunks.append(chunk_text)
        return chunks

    # 문제-정답 처리
    for i, question in enumerate(questions):
        question_clean = clean_text(question)
        if not question_clean:
            continue
        answer = answers[i] if i < len(answers) else None
        chunks = tokenize_and_chunk(question_clean)
        for chunk in chunks:
            question_chunks.append(chunk)
            question_answer_pairs.append({"question": chunk, "answer": answer})

    # 일반 텍스트 처리
    for text in general_texts:
        text_clean = clean_text(text)
        if not text_clean:
            continue
        chunks = tokenize_and_chunk(text_clean)
        general_chunks.extend(chunks)

    return question_chunks, question_answer_pairs, general_chunks

# 텍스트를 OpenAI 임베딩 벡터로 변환하는 함수
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding
