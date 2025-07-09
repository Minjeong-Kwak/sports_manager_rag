from modules.pdf_loader import extract_questions_and_answers
from modules.vector_store import create_faiss_index, search_faiss, generate_response
from modules.problem_solver import solve_text_problem, solve_image_problem, solve_pdf_problem, generate_mcq
import modules.vector_store as vector_store
from modules.text_processing import chunk_text

import os
import time
import sys
import io
import json
import faiss
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def type_out(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # 마지막에 개행

# ✅ 진행 과정 즉시 출력하는 함수
def print_progress(message):
    print(f"{message}...", end="\r", flush=True)

print_progress("🔍 [1] PDF에서 문제, 정답, 일반 텍스트 추출 중")
questions, answers, general_texts = extract_questions_and_answers()
print(f"✅ [1] 완료! (문제: {len(questions)}개, 정답: {len(answers)}개, 일반 텍스트: {len(general_texts)}개)")

# ✅ JSON 파일로 저장
os.makedirs("output", exist_ok=True)
with open("output/questions.json", "w", encoding="utf-8") as fq:
    json.dump(questions, fq, ensure_ascii=False, indent=2)
with open("output/answers.json", "w", encoding="utf-8") as fa:
    json.dump(answers, fa, ensure_ascii=False, indent=2)
    
print_progress("🔍 [2] 청크 분할 중")
question_chunks, question_answer_pairs, general_chunks = chunk_text(
    questions, answers, general_texts, max_length=300, overlap=50
)

print(f"✅ [2] 완료! (문제 청크: {len(question_chunks)}개, 일반 텍스트 청크: {len(general_chunks)}개)")

# ✅ 인덱스가 이미 존재하면 재사용, 없으면 생성
if not os.path.exists("embeddings/faiss_index") or not os.path.exists("embeddings/metadata.json"):
    print_progress("🔍 [3] FAISS 및 BM25 인덱스 생성 중")
    vector_store.FAISS_INDEX, vector_store.BM25_CORPUS = create_faiss_index(
        question_answer_pairs, general_chunks
    )
    faiss.write_index(vector_store.FAISS_INDEX, "embeddings/faiss_index")
    with open("embeddings/metadata.json", "w", encoding="utf-8") as f:
        json.dump(vector_store.BM25_CORPUS, f, ensure_ascii=False, indent=2)
    print("✅ [3] 완료!")
else:
    print("✅ [3] 인덱스가 이미 존재합니다. 재사용합니다.")
    vector_store.FAISS_INDEX = faiss.read_index("embeddings/faiss_index")
    with open("embeddings/metadata.json", "r", encoding="utf-8") as f:
        vector_store.BM25_CORPUS = json.load(f)

print("✅ [3] 완료!")

def main():
    gpt_response = ""
    results = []
    
    while True:
        print("\n🔍 검색할 질문을 입력하거나, 'solve'를 입력하면 문제를 풀어드립니다.")
        print("   'generate'를 입력하면 객관식 문제를 생성합니다. (종료하려면 'exit' 입력)")
        query = input("입력: ")

        if query.lower() == "exit":
            print("🔚 프로그램을 종료합니다.")
            break

        elif query.lower() == "solve":
            print("\n📌 문제 풀이 방식을 선택하세요:")
            print("1. 텍스트 입력")
            print("2. 이미지 파일 업로드")
            print("3. PDF 파일 업로드")

            choice = input("선택 (1, 2, 3): ")

            if choice == "1":
                problem_text = input("\n✏️ 문제를 입력하세요: ")
                solution = solve_text_problem(problem_text)
                type_out("\n🤖 RAG 답변:")
                type_out(solution)

            elif choice == "2":
                image_path = input("\n📂 이미지 파일 경로를 입력하세요: ")
                solution = solve_image_problem(image_path) if os.path.exists(image_path) else "❌ 파일을 찾을 수 없습니다."
                type_out("\n🤖 RAG 답변:")
                type_out(solution)

            elif choice == "3":
                pdf_path = input("\n📂 PDF 파일 경로를 입력하세요: ")
                solution = solve_pdf_problem(pdf_path) if os.path.exists(pdf_path) else "❌ 파일을 찾을 수 없습니다."
                type_out("\n🤖 RAG 답변:")
                type_out(solution)

            else:
                print("❌ 올바른 선택이 아닙니다.")

        elif query.lower() == "generate":
            print("\n📌 객관식 문제 생성 모드입니다.")
            question = input("🔹 생성할 문제의 키워드를 입력하세요: ")

            # 🔹 관련된 정보를 검색하여 참고 자료로 사용
            search_results = search_faiss(question, top_k=1)
            reference_text = search_results[0]["text"] if search_results else "관련된 정보를 찾을 수 없습니다."

            # 🔹 객관식 문제 생성 실행
            mcq = generate_mcq(question, reference_text)

            # ✅ 문제와 정답/해설 분리
            lines = mcq.split("\n")
            question_part = "\n".join(lines[:6])  # 문제 + 보기만 출력
            answer_part = "\n".join(lines[6:])  # 정답 + 해설

            type_out("\n✅ 생성된 객관식 문제:\n")
            type_out(question_part)

            # 사용자 정답 입력 받기
            user_answer = input("\n📝 정답을 입력하세요 (1~4): ")

            type_out("\n📌 정답 및 해설:\n")
            type_out(answer_part)

            # ✅ 피드백 기능을 위해 gpt_response, results 기본값 설정
            gpt_response = mcq
            results = [{"type": "generated_mcq", "text": mcq}]

        else:
            start_time = time.time()
            print(f"\n🔍 [5] FAISS 검색 실행 중 (쿼리: {query})")

            try:
                results = search_faiss(query, top_k=3)
                print("✅ [5] 완료!")

                type_out("\n📌 검색된 결과:")
                for res in results:
                    if res["type"] == "qa":
                        type_out(f"📖 문제: {res['question']}")
                        type_out(f"✅ 정답: {res['answer']}\n")
                    elif res["type"] == "text":
                        type_out(f"📄 일반 텍스트: {res['text']}\n")

                gpt_response = generate_response(query, results) if results else "❌ 관련된 정보를 찾을 수 없습니다."
                type_out("\n🤖 RAG 답변:")
                type_out(gpt_response)

                execution_time = time.time() - start_time
                log_interaction(query, results, gpt_response, execution_time)

            except Exception as e:
                execution_time = time.time() - start_time
                print(f"❌ 오류 발생: {e}")
                log_interaction(query, [], "❌ 오류 발생", execution_time, str(e))
                gpt_response = "❌ 오류 발생"
                results = []  # ✅ 예외 발생 시 빈 리스트 설정

        # ✅ 1. 모든 질문과 답변이 끝난 후 피드백 요청
        new_query = interactive_feedback(query, gpt_response, results)

        # ✅ 2. 피드백이 끝난 후, 새로운 질문을 다시 받도록 루프 유지
        if new_query:  
            query = new_query

# ✅ main() 실행 코드 추가 (위치는 main() 함수 정의 이후)
if __name__ == "__main__":
    print("✅ [4] 사용자 입력을 대기 중...")  # ✅ main()이 실행되는지 확인
    main()
