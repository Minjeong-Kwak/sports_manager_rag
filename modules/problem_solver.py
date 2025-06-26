import os
import re
import pytesseract
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image
from modules.vector_store import find_similar_questions

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI API 설정
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ 텍스트 입력 문제 풀이
from modules.vector_store import search_faiss  # 추가 필요

def solve_text_problem(problem_text):
    search_results = search_faiss(problem_text, top_k=3)
    context = "\n".join([r["text"] for r in search_results])

    prompt = f"""당신은 스포츠경영관리사 문제를 푸는 AI입니다.
    
문제: {problem_text}

📚 참고 정보:
{context}

✍️ 풀이 및 정답:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 스포츠경영관리사 시험 문제를 푸는 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ✅ 이미지 문제 풀이 (OCR)
def solve_image_problem(image_path):
    try:
        # 이미지에서 텍스트 추출 (OCR)
        image = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return "❌ 이미지에서 문제를 인식하지 못했습니다."

        print(f"🔍 OCR 인식된 문제:\n{extracted_text.strip()}")

        # GPT를 활용해 문제 풀이
        return solve_text_problem(extracted_text.strip())
    
    except Exception as e:
        return f"❌ 이미지 문제 풀이 중 오류 발생: {e}"

# ✅ PDF 문제 풀이
def solve_pdf_problem(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        extracted_text = ""

        # 모든 페이지에서 텍스트 추출
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"

        if not extracted_text.strip():
            return "❌ PDF에서 문제를 인식하지 못했습니다."

        print(f"📖 PDF에서 추출된 문제:\n{extracted_text.strip()}")

        # GPT를 활용해 문제 풀이
        return solve_text_problem(extracted_text.strip())

    except Exception as e:
        return f"❌ PDF 문제 풀이 중 오류 발생: {e}"

#문제 생성 코드
def generate_mcq(question_text, reference_text):
    from modules.vector_store import find_similar_questions

    similar_questions = find_similar_questions(question_text)
    similar_question_text = "\n".join(similar_questions) if similar_questions else "유사한 문제가 없습니다."

    prompt = f"""당신은 스포츠경영관리사 시험 출제 전문가입니다.

다음 참고 정보를 바탕으로 '중급 이상 난이도의' 객관식 문제를 1문항 생성하세요.

📌 조건:
- 다양한 유형의 문제를 출제하세요. 예를 들어:
  - 보기 중 맞는 답 1개를 고르는 문제
  - 보기 중 틀린 답 1개를 고르는 문제제
  - 개념과 정의의 짝짓기 문제
  - 빈칸에 들어갈 개념 추론 문제
  - 보기(ㄱ, ㄴ, ㄷ) 중 해당하는 것을 모두 고르는 문제
  - 사례 기반 추론 문제
  - 연결형 문제

- 보기는 실제 시험에 나올 법한 헷갈리는 선지로 구성하세요.
- 맞는 답을 고르거나 틀린 답을 고르는 문제의 정답 외 오답은 혼란을 줄 수 있는 보기여야 합니다.
- 기존 문제와 똑같은 문장을 반복하지 마세요.
- 선택지는 내용상 유사하게 보여야 하지만, 정확히 알지 않으면 오답이 되도록 구성하세요.

🔍 참고 정보:
{reference_text}

📌 문제 유형 예시 (참고만 하세요. 그대로 출제하지 마세요):

1. 문제. 환경분석에 사용되는 SWOT 분석 요인을 바르게 짝지은 것은?
   1) 내부환경 : S-T  
   2) 외부환경 : S-W  
   3) 내부환경 : W-O  
   4) 외부환경 : O-T

2. 문제. 다음 ( )에 알맞는 것은?  
   ---기업의 사명에 '혼'을 불어넣는 역할을 하는 (    )은/는 "기업이 미래에 달성하고자 하는 기업상"이다...  
   1) 미션  
   2) 비전  
   3) 사업포트폴리오  
   4) 성장 벡터

3. 문제. BCG 매트릭스에서 question mark 사업 단위에 적합한 전략 유형을 모두 고른 것은?  
   ㄱ. 유지전략(hold) ㄴ. 증대전략(build) ㄷ. 수확전략(harvest) ㄹ. 철수전략(divest)  
   1) ㄱ  
   2) ㄱ, ㄷ  
   3) ㄴ, ㄷ  
   4) ㄴ, ㄷ, ㄹ

📤 출력 형식 (고정):
질문: [문제 내용]  
보기:  
1) ...  
2) ...  
3) ...  
4) ...  
정답: [정답 번호]  
해설: [정답에 대한 설명]
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "당신은 스포츠경영관리사 시험 문제 출제 전문가입니다."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

