import os
import re
from pypdf import PdfReader
import pytesseract

# Tesseract 실행 경로를 직접 지정합니다.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# PDF 파일이 저장된 폴더 경로
PDF_FOLDER = "data/"

def extract_questions_and_answers():
    questions = []
    answers = []
    general_texts = []  # 문제 형식이 아닌 일반 텍스트 저장

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ PDF 파일을 찾을 수 없습니다. data/ 폴더를 확인하세요.")
        return [], [], []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"📖 파일 읽는 중: {pdf_file}")

        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    lines = text.split("\n")
                    current_question = ""
                    page_has_qa = False  # 해당 페이지에 문제-정답 형식이 있는지 여부

                    for line in lines:
                        line = line.strip()
                        # 정답 라인: "정답", "답", "A" 등으로 시작하는 경우
                        if re.match(r"^(정답|답|A)\b", line):
                            if current_question:
                                questions.append(current_question)
                                answers.append(line)
                                current_question = ""
                                page_has_qa = True
                        # 문제 라인: "문제"라는 단어가 있을 수도 있고, 숫자로 시작하는 경우도 포함
                        elif re.match(r"^(문제\s*)?\d+[\.\)]\s+", line):
                            if current_question:
                                # 이전 문제에 대해 정답이 없는 경우 빈 문자열로 추가
                                questions.append(current_question)
                                answers.append("")
                            current_question = line
                        else:
                            # 이미 문제 라인이 시작된 경우, 같은 문제의 나머지 텍스트로 취급하여 이어 붙임
                            if current_question:
                                current_question += " " + line
                    # 페이지에서 QA 형식이 감지된 경우 남은 current_question 저장
                    if current_question and page_has_qa:
                        questions.append(current_question)
                        answers.append("")
                    # 만약 QA 형식이 아니라면 전체 텍스트를 일반 텍스트로 저장
                    if not page_has_qa:
                        general_texts.append(text.strip())

    print("✅ 문제, 정답 및 일반 텍스트 추출 완료!")
    return questions, answers, general_texts
