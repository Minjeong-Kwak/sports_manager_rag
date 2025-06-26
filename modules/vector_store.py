import os
import faiss
import numpy as np
import json
from dotenv import load_dotenv
from openai import OpenAI
from modules.text_processing import get_embedding
from rank_bm25 import BM25Okapi
from modules import vector_store
import re
import tiktoken

def normalize_text(text):
    return re.sub(r"\s+", "", text.strip())

encoding = tiktoken.get_encoding("cl100k_base")


# ✅ 글로벌 변수로 벡터 인덱스 캐싱
FAISS_INDEX = None
BM25_CORPUS = []

# ✅ 텍스트에서 숫자와 수식을 추출하는 함수 추가
def extract_numbers_and_formula(text):
    """주어진 텍스트에서 숫자와 간단한 수식을 추출"""
    numbers = re.findall(r"\d+\.?\d*", text)
    return numbers

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API 설정
client = OpenAI(api_key=OPENAI_API_KEY)

# FAISS 벡터 데이터 저장 경로
FAISS_INDEX_PATH = "embeddings/faiss_index"
METADATA_PATH = "embeddings/metadata.json"

# FAISS 인덱스를 저장할 폴더 생성
os.makedirs("embeddings", exist_ok=True)

# ✅ BM25 검색 인덱스
bm25_corpus = []
bm25_index = None

# ✅ FAISS 인덱스 로드 함수
def load_faiss_index():
    global FAISS_INDEX
    if not os.path.exists(FAISS_INDEX_PATH):
        print("❌ FAISS 인덱스를 로드할 수 없습니다.")
        return None
    FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
    return FAISS_INDEX

# BM25 인덱스 로드 함수
def load_bm25_corpus():
    global BM25_CORPUS, bm25_index
    if not os.path.exists(METADATA_PATH):
        print("❌ BM25 metadata.json 파일을 찾을 수 없습니다.")
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        BM25_CORPUS = json.load(f)
    bm25_index = BM25Okapi([doc.split() for doc in BM25_CORPUS])  # ✅ 인덱스도 재생성
    return BM25_CORPUS

# ✅ FAISS + BM25 검색을 위한 인덱스 생성
def create_faiss_index(question_answer_pairs, general_chunks):
    global bm25_corpus, bm25_index

    print("🔍 FAISS 및 BM25 인덱스 생성 중...")

    qa_texts = [q["question"] for q in question_answer_pairs] if question_answer_pairs else []
    general_texts = general_chunks if general_chunks else []

    if not qa_texts and not general_texts:
        print("⚠️ 인덱스를 생성할 데이터가 없습니다. 빈 인덱스를 생성합니다.")

    dummy_text = "기본 더미 데이터"
    dummy_embedding = get_embedding(dummy_text)

    qa_embeddings = np.array([get_embedding(text) for text in qa_texts], dtype=np.float32) if qa_texts else np.array([dummy_embedding], dtype=np.float32)
    general_embeddings = np.array([get_embedding(text) for text in general_texts], dtype=np.float32) if general_texts else np.empty((0, qa_embeddings.shape[1]), dtype=np.float32)

    qa_embeddings /= np.linalg.norm(qa_embeddings, axis=1, keepdims=True) + 1e-10
    if general_embeddings.shape[0] > 0:
        general_embeddings /= np.linalg.norm(general_embeddings, axis=1, keepdims=True) + 1e-10

    index = faiss.IndexFlatIP(qa_embeddings.shape[1])

    print(f"🟢 벡터 추가 중... (총 {qa_embeddings.shape[0] + general_embeddings.shape[0]}개)")
    all_embeddings = np.vstack((qa_embeddings, general_embeddings)) if general_embeddings.shape[0] > 0 else qa_embeddings
    index.add(all_embeddings)

    print("✅ FAISS 인덱스 저장 중...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("✅ FAISS 인덱스 저장 완료!")

    bm25_corpus = qa_texts + general_texts if qa_texts or general_texts else [dummy_text]
    bm25_index = BM25Okapi([doc.split() for doc in bm25_corpus])

    print("✅ BM25 키워드 검색 인덱스 생성 완료!")

    return index, bm25_corpus

# ✅ FAISS + BM25 검색 실행
def search_faiss(query, top_k=7, filter_type=None):
    global FAISS_INDEX, BM25_CORPUS
    if FAISS_INDEX is None:
        print("❌ 인덱스가 로드되지 않았습니다. main.py를 먼저 실행하세요.")
        return []

    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    query_embedding /= np.linalg.norm(query_embedding)

    raw_k = top_k * 4
    distances, indices = FAISS_INDEX.search(query_embedding, raw_k)

    candidate_texts = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(BM25_CORPUS):
            similarity = 1 - dist
            if similarity < 0.3:
                continue
            text = BM25_CORPUS[idx]
            candidate_texts.append(text)

        # ✅ BM25 점수로 재정렬 (normalize_text 기준으로 정렬)
    if bm25_index is not None:
        query_tokens = query.split()
        scores = bm25_index.get_scores(query_tokens)

        ranked = sorted(
            candidate_texts,
            key=lambda x: scores[BM25_CORPUS.index(x)],
            reverse=True
        )
    else:
        ranked = candidate_texts  # fallback

    results = [{"type": "text", "text": t} for t in ranked[:top_k]]
    return results


# ✅ 수치 계산이 필요한 경우 처리하는 함수
def execute_calculation(search_results):
    for res in search_results:
        if res["type"] == "qa":
            numbers = extract_numbers_and_formula(res["question"])
            if numbers:
                try:
                    numbers = [float(num) for num in numbers]
                    if "유동비율" in res["question"]:
                        result = (numbers[0] / numbers[1]) * 100
                        return f"유동비율은 {result:.2f}%입니다."
                except Exception as e:
                    return f"❌ 계산 중 오류 발생: {e}"
    return None

# 문제 검색
def find_similar_questions(query):
    search_results = vector_store.search_faiss(query, top_k=5)
    similar_questions = []

    for result in search_results:
        if "question" in result:
            similar_questions.append(result["question"])

    return similar_questions if similar_questions else ["유사한 문제가 없습니다."]

MAX_TEXT_TOKENS = 2000

def trim_text(text, max_tokens=MAX_TEXT_TOKENS):
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:max_tokens])

# ✅ GPT 기반 응답 생성 함수
def generate_response(query, search_results):
    calculation_result = execute_calculation(search_results)
    if calculation_result:
        return calculation_result

    for result in search_results:
        if result["type"] == "text":
            result["text"] = trim_text(result["text"])

    valid_answers = []
    general_info = []
    limited_results = search_results

    for res in limited_results:
        if res["type"] == "qa" and res["answer"]:
            valid_answers.append(f"문제: {res['question']}\n정답: {res['answer']}")
        elif res["type"] == "text":
            general_info.append(res["text"])

    if not valid_answers and not general_info:
        return "❌ 관련된 정보를 찾을 수 없습니다. 질문을 더 구체적으로 입력해 주세요."

    context = "\n\n".join(valid_answers + general_info)
    context = context[:8000]

    prompt = f"""당신은 스포츠경영관리사 시험을 돕는 AI입니다.  
    사용자의 질문: "{query}"  generate

    📌 **역할:**  
    - 스포츠경영관리사 시험 합격을 목표로 하는 수험생을 지원합니다.  
    - 신뢰할 수 있는 정보를 제공하며, 정확하고 논리적인 답변을 작성해야 합니다.
    - 수험생이 원할경우 문제를 생성하여 제공합니다.  

    📌 **응답 가이드:**  
    - 반드시 검색된 정보를 기반으로 답변하세요.  
    - 개념이 필요한 경우 설명을 보충하세요.  
    - 문장은 명확하고 실용적으로 작성해야 합니다.
    - 답변할 수 없는 정보가 입력된다면 사실대로 답변할 수 없다고 대답해야 합니다.
    - 이모지를 활용하여 답변해도 됩니다.

    🔍 **참고 정보:**  
    {context}  

    ✍️ **답변:**"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 스포츠경영관리사 전문가이며, 검색된 정보를 최우선으로 활용하여 답변해야 합니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    return response.choices[0].message.content
