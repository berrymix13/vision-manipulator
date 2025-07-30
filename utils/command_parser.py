## LLM to Json

import os, re
import json 
import openai
from dotenv import load_dotenv

# 환경 변수 로딩
load_dotenv("/home/ros/llm_robot/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

PROMPT_PATH = "/home/ros/llm_robot/prompt/llm2yolo_json.txt"

def strip_codefence(text: str) -> str:
    """
    ```json\n{...}\n```  또는  ```\n{...}\n```  형태에서 {...} 부분만 돌려준다.
    코드펜스가 없으면 원문 그대로 반환.
    """
    # ```json … ```  또는  ``` … ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    return m.group(1) if m else text.strip()



# GPT-4o에게 코드 요청
def parse_command(user_input: str) -> dict:
    """ 자연어 명령을 JSON 형식으로 변환 """
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.2,
        max_tokens=500
    )

    raw = response.choices[0].message.content
    json_str = strip_codefence(raw)

    try:
        return json.loads(json_str)
    except Exception as e:
        print("❌ JSON 파싱 오류:", e)
        print("📄 GPT 응답 원문:\n", raw)
        return {}