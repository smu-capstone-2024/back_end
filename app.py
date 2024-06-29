from flask import Flask, request, jsonify
import requests
import redis
import openai
import json
from flask_cors import CORS
import uuid
import time
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI
#host.docker.internal
app = Flask(__name__)
CORS(app, support_credentials=True)
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

openai.api_key = 'api_key'
client = OpenAI(api_key=os.environ.get("api_key"))


def ask_gpt(prompt):
    print("ask_gpt")
    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except openai.error.InvalidRequestError as e:
        return f"Error: {str(e)}"

def LLM_filtering(question):
    cpu_rank = {
        'AMD Ryzen Threadripper PRO 7995WX': 0,
        'AMD Ryzen Threadripper 7980X': 1,
        'AMD Ryzen Threadripper 7970X': 11,
        'AMD Ryzen Threadripper PRO 7975WX': 13,
        'AMD Ryzen Threadripper PRO 5995WX': 14,
        'AMD Ryzen Threadripper PRO 7965WX': 23,
        'AMD Ryzen Threadripper 7960X': 24,
        'Intel Core i9-14900KS': 56,
        'AMD Ryzen 9 7950X': 60,
        'AMD Ryzen 9 7950X3D': 63,
        'Intel Core i9-13900KS': 67,
        'Intel Core i7-14700K': 98,
        'AMD Ryzen 9 7900X': 107,
        'AMD Ryzen 9 7900X3D': 111,
        'Intel Core i7-13700K': 136,
        'Intel Core i5-14600KF': 181,
        'Intel Core i5-13600K': 194,
        'AMD Ryzen 7 7700X': 209,
        'AMD Ryzen 7 7800X3D': 225,
        'Intel Core i5-14500': 247,
        'Intel Core i5-13500': 257,
        'AMD Ryzen 5 7600X': 312,
        'Intel Core i5-12600K': 332,
        'Apple M3 Pro 12 Core': 335,
        'AMD Ryzen 5 7500F': 355,
        'AMD Ryzen 7 5700X': 359,
        'Intel Core i5-14400': 361,
        'AMD Ryzen 7 5700X3D': 366,
        'Intel Core i5-13400F': 399,
        'AMD Ryzen 5 5600X': 504,
        'Intel Core i5-12400F': 595,
        'Intel Core i3-12100E': 844
    }

    gpu_rank = {
        'GeForce RTX 4090': 0,
        'GeForce RTX 4080': 1,
        'GeForce RTX 4070 Ti': 4,
        'Radeon RX 7900 XTX': 6,
        'Radeon PRO W7800': 11,
        'RTX 6000 Ada Generation': 14,
        'Radeon RX 6800 XT': 25,
        'GeForce RTX 3070 Ti': 28,
        'GeForce RTX 4060 Ti 16GB': 31,
        'GeForce RTX 4060 Ti': 32,
        'NVIDIA A10': 37,
        'RTX A5500': 43,
        'GeForce RTX 3060 Ti': 47,
        'Radeon RX 6700 XT': 53,
        'Radeon RX 6750 GRE 12GB': 67,
        'Radeon PRO W7600': 72,
        'Radeon RX 6650 XT': 80,
        'GeForce RTX 3060 12GB': 81,
        'Radeon RX 6600 XT': 88,
        'GeForce RTX 3060 8GB': 106,
        'RTX A2000 12GB': 132,
        'GeForce GTX 1660 Ti': 152,
        'GeForce RTX 3050 8GB': 153,
        'GeForce RTX 3050 OEM': 170,
        'L4': 183,
        'Intel Arc A580': 184,
        'GeForce RTX 3050 6GB': 196,
        'GeForce GTX 1650 SUPER': 207
    }

    prompt = f"""
    다음 질문을 분석하여 포함된 CPU 및 GPU를 확인하고, 질문을 재정의해 주세요.

    1. 만약 질문에 CPU나 GPU가 없다면, 질문을 그대로 출력해 주세요.

    2. 질문 안에 '이상', '이하', '더 좋은', '더 안 좋은' 등의 조건이 있는 경우:
        - 해당 조건을 해석하여 목록에 있는 항목과 비교하여 가장 비슷한 항목을 찾아 주세요. 
        - 만약 목록에 없는 경우에도 '2060이상을 추천해줘'와 같은 조건이 있다면, 목록에서 해당 조건을 만족하는 항목을 찾아 질문을 재정의해 주세요.
        - 목록에 더 이상 좋은 항목이나 더 안 좋은 항목이 없는 경우, '조건을 만족하는 항목이 없습니다.다시 질문하여 견적을 추천 받으세요.'라고 출력해 주세요.
        - '이상','이하','더 좋은' ,'더 안좋은'등의 조건이 있다면 해당 단어도 그대로 재정의 질문에 넣어서 출력해 주세요.

    3. 질문 안에 '포함된', '들어간' 등의 표현이 있는 경우: '이상', '이하', '더 좋은', '더 안 좋은' 등의 조건이 있는 경우 예외가 존재함 
        - 질문에 포함된 CPU와 GPU가 목록에 있는 경우, 정확한 이름으로 대체하여 질문을 재정의해 주세요.
        - 목록에 없는 경우, '해당 CPU나 GPU는 목록에 없습니다. 다시 질문하여 견적을 추천 받으세요.'라고 출력해 주세요.

    4. 재정의된 질문이 있다면, 다른 것은 출력하지 말고 무조건 재정의된 질문만 출력해 주세요.
            
    5. 재정의된 질문이 없다면, '조건을 만족하는 항목이 없습니다.다시 질문하여 견적을 추천 받으세요.'라고 출력해 주세요.
        예시를 꼭 참고 바랍니다!!
            예시
            - "2060이상이 포함된 컴퓨터 견적을 추천해줘"와 같은 조건이 있다면, 목록에서 그래픽카드 2060이상인 항목을 키값을 찾아서 넣어 재정의하면 됩니다. - 예시: GeForce RTX 3060 8GB이상이 포함된 컴퓨터 견적을 추천해줘
            - "2060이 포함된(들어간) 컴퓨터 견적을 추천해줘"하면 그래픽카드 2060은 목록에 없으므로 해당 CPU나 GPU는 목록에 없습니다. 다시 질문하여 견적을 추천 받으세요.라고 출력해 주세요.
            - "3060이상이 들어간 컴퓨터 견적 추천해줘"하면 목록에서 그래픽카드 3060이상인 항목을 찾아서 하지만 목록에 있으므로 넣어 재정의하면 됩니다. - 예시: GeForce RTX 3060 12GB이상이 들어간 컴퓨터 견적 추천해줘
            - "3060이 포함된(들어간) 컴퓨터 견적을 추천해줘"하면 그래픽카드 3060은 목록에 있으므로 목록에서 키값을 똑같이 기입해 재정의해주세요. - 키값: GeForce RTX 3060 12GB이상이 들어간 컴퓨터 견적 추천해줘
            - "i7-11세대이상이 포함된 견적을 추천해줘"하면 목록에서 i7-11세대이상인 항목을 찾아서 똑같이 넣어 재정의하면 됩니다. - 키값: Intel Core i7-13700K이상이 포함된 견적을 추천해줘
            - "i7-11세대이 포함된(들어간) 컴퓨터 견적을 추천해줘"하면 i7-11세대은 목록에 없으므로 해당 CPU나 GPU는 목록에 없습니다. 다시 질문하여 견적을 추천 받으세요.라고 출력해 주세요.
    6. 문자열이 짤리지 않도록 주의해 주세요.
    7. 다른 텍스트를 포함하지 않고 재정의된 질문이나 질문만 출력해줘야 합니다.
    
    질문: {question}

    CPU 목록:
    {list(cpu_rank.keys())}

    GPU 목록:
    {list(gpu_rank.keys())}
    """

    response = ask_gpt(prompt)

    if "포함되지 않았음" in response:
        return response
    else:
        return response

def call_by_LLm(session_id, origin_text):
    cpu_rank = {
        'AMD Ryzen Threadripper PRO 7995WX': 0,
        'AMD Ryzen Threadripper 7980X': 1,
        'AMD Ryzen Threadripper 7970X': 11,
        'AMD Ryzen Threadripper PRO 7975WX': 13,
        'AMD Ryzen Threadripper PRO 5995WX': 14,
        'AMD Ryzen Threadripper PRO 7965WX': 23,
        'AMD Ryzen Threadripper 7960X': 24,
        'Intel Core i9-14900KS': 56,
        'AMD Ryzen 9 7950X': 60,
        'AMD Ryzen 9 7950X3D': 63,
        'Intel Core i9-13900KS': 67,
        'Intel Core i7-14700K': 98,
        'AMD Ryzen 9 7900X': 107,
        'AMD Ryzen 9 7900X3D': 111,
        'Intel Core i7-13700K': 136,
        'Intel Core i5-14600KF': 181,
        'Intel Core i5-13600K': 194,
        'AMD Ryzen 7 7700X': 209,
        'AMD Ryzen 7 7800X3D': 225,
        'Intel Core i5-14500': 247,
        'Intel Core i5-13500': 257,
        'AMD Ryzen 5 7600X': 312,
        'Intel Core i5-12600K': 332,
        'Apple M3 Pro 12 Core': 335,
        'AMD Ryzen 5 7500F': 355,
        'AMD Ryzen 7 5700X': 359,
        'Intel Core i5-14400': 361,
        'AMD Ryzen 7 5700X3D': 366,
        'Intel Core i5-13400F': 399,
        'AMD Ryzen 5 5600X': 504,
        'Intel Core i5-12400F': 595,
        'Intel Core i3-12100E': 844
    }

    gpu_rank = {
        'GeForce RTX 4090': 0,
        'GeForce RTX 4080': 1,
        'GeForce RTX 4070 Ti': 4,
        'Radeon RX 7900 XTX': 6,
        'Radeon PRO W7800': 11,
        'RTX 6000 Ada Generation': 14,
        'Radeon RX 6800 XT': 25,
        'GeForce RTX 3070 Ti': 28,
        'GeForce RTX 4060 Ti 16GB': 31,
        'GeForce RTX 4060 Ti': 32,
        'NVIDIA A10': 37,
        'RTX A5500': 43,
        'GeForce RTX 3060 Ti': 47,
        'Radeon RX 6700 XT': 53,
        'Radeon RX 6750 GRE 12GB': 67,
        'Radeon PRO W7600': 72,
        'Radeon RX 6650 XT': 80,
        'GeForce RTX 3060 12GB': 81,
        'Radeon RX 6600 XT': 88,
        'GeForce RTX 3060 8GB': 106,
        'RTX A2000 12GB': 132,
        'GeForce GTX 1660 Ti': 152,
        'GeForce RTX 3050 8GB': 153,
        'GeForce RTX 3050 OEM': 170,
        'L4': 183,
        'Intel Arc A580': 184,
        'GeForce RTX 3050 6GB': 196,
        'GeForce GTX 1650 SUPER': 207
    }
    history = redis_client.get(session_id)
    if history:
        conversation = json.loads(history)
    else:
        conversation = {'chat_history': []}

    current_turn = {'user': "User: " + origin_text}
    conversation['chat_history'].append(current_turn)

    prompt_text = "\n".join([
        f"{turn['user']}\n{turn.get('ai', '')}"
        for turn in conversation['chat_history']
    ])
    print(prompt_text)
    
    try:
        prompt = f"""
            지금까지의 대화 내역은 다음과 같습니다:{prompt_text}
            견적 외의 대화 내역의 일부 답변이 틀릴 수 있습니다. 
            user: {origin_text}
            와 같이 질문이 들어왔습니다.

            규칙1. 사용자가 추천이나 비교를 요청하는 경우 전체 대화 기록과 사용자의 질문을 고려하여 해당 언어로 상세하고 정확한 답변을 제공하세요.
            규칙2. 특히 사용자가 더 나은 CPU 또는 GPU를 요청하는 경우에는 제공된 사양에서 현재 사양보다 성능이 더 좋은 옵션을 나열하세요.
                {cpu_rank}의 숫자가 낮을수록 좋고, {gpu_rank}의 숫자가 낮을수록 좋습니다.
            규칙3. 대화 내역에서 AI가 알 수 없는 잘못 답변 한 것 같으면 고쳐주세요.
            규칙4. user의 질문에서 AI가 이전 대화에서 잘못 대답을 했다 한다면 AI가 잘못 대답한 것 이므로 대화 내역을 통해 흐름을 보고 고쳐서 다시 제공해 주세요.
            규칙5. 요약을 물어보면 대화 내역의 견적 내용에서 요약을 가져와서 대답해주세요.
            규칙6. 컴퓨터 관련 지식 외의 질문은 답하지 마세요.
            """
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[
                {"role": "system", "content": "귀하는 컴퓨터 하드웨어 및 기술 사양을 전문으로 하는 전문 AI 도우미입니다. 여러분의 임무는 대화 기록을 바탕으로 간결하고 정확하며 맥락에 맞는 답변을 제공하는 것입니다."},
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        answer = response.choices[0].message.content
        print("파인튜닝 답변: "+answer)
        prompt = f"""
            지금까지의 대화 내역은 다음과 같습니다:{prompt_text}
            견적 외의 대화 내역의 일부 답변이 틀릴 수 있습니다. 
            (User): {origin_text}
            와 같이 질문이 들어왔습니다.
            그에 대한 답변으로
            (AI): {answer} 
            과 같이 답변을 했습니다.
            
            규칙 1. 이 답변이 제대로 나온 것 같다면 그대로 답변만 해주세요.
            규칙 2. 답변이 제대로 나오지 않거나 오류 및 오타가 있다면 고쳐서 답변만 해 주세요.
            규칙 3. 사용자가 질문한 {origin_text}에 대해 답변 {answer}가 올바르지 않다면 고쳐서 답변만 해 주세요.
            """
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[
                {"role": "system", "content": "귀하는 컴퓨터 하드웨어 및 기술 사양을 전문으로 하는 전문 AI 도우미입니다. 여러분의 임무는 대화 기록을 바탕으로 간결하고 정확하며 맥락에 맞는 답변을 제공하는 것입니다."},
                {"role": "user", "content": f"{prompt}"}
            ]
        )
        answer = response.choices[0].message.content
        print("수정한 답변: "+answer)

        conversation['chat_history'][-1]['ai'] = "(AI): " + answer

    

        if len(conversation['chat_history']) > 2:
            conversation['chat_history'] = conversation['chat_history'][:1] + conversation['chat_history'][-1:]
        redis_client.set(session_id, json.dumps(conversation))
        return answer
    except Exception as e:
        print(f"Error calling the API: {e}")
        return "API 호출 중 에러 발생"
    
def fetch_details(part, details):
    user_message = f"{details['제품명']}의 상세정보를 통해 장점과 단점을 전문적이지만 컴퓨터에 대해 잘 모르는 사람이 이해할 수 있도록 쉽고 친근하게 설명해줘."
    response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant adept at providing technical specifications and performance data."},
            {"role": "user", "content": user_message}
        ]
    )
    return part, details['제품명'], details['가격'], response.choices[0].message.content


def call_by_Rag_LLm(session_id, text, detail, quotes):
    data1 = quotes["quote_description"]
    data2 = quotes["quote_feedback"]
    price = quotes["total_price"]
    current_turn = {'user': "(User): " + text}
    conversation = {'chat_history': []}
    conversation['chat_history'].append(current_turn)
    start_time = time.time()
    try:
        answers = []
        
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_details, part, details) for part, details in detail.items()]
            for future in as_completed(futures):
                part, product_name, price, response_content = future.result()
                answers.append(f"{part}: {product_name} - {price}\n{response_content}\n")
        answers_str = "\n".join(answers)
        print(answers_str)
        print("견적 상세: " + data1)
        print("견적 피드백: " + data2)

        review_data = f"""
        USER: {text}
        의 사용자 질문에 대해 {answers_str}와 같은 견적을 추천해 줄 때 추천 이유를 설명해주세요.
        해당 견적의 총 가격은 {price}입니다.
        이제 이것에 대해서 설명을 해야 합니다.
        이게 견적 설명 데이터: {data1}
        이건 견적 피드백의 예시 데이터: {data2}
        이 견적 설명 데이터와 피드백의 예시 데이터를 참고해서 주관적인 생각의 데이터를 너의 객관적인 견해로 {text}의 사용자 질문에 대한 견적 추천 이유를 참고해서 설명해야 합니다.
        
        명시해야할 규칙으로
        규칙1. {text}의 USER 질문에 대한 해당 견적을 추천하는 이유가 들어가야 합니다.
        규칙2. 견적 추천 이유에서 너무 길거나 짧은 설명이면 안되며 해당 부품의 장점과 개선 가능한 점을 말해 주어야 합니다.
        규칙3. 너무 전문적인 부품 설명보단 해당 부품의 ~ 스펙으로 인해 ~의 성능이 제공되어 추천한다. 와 같이 전문적인 답변을 쉬운 말로 풀어서 설명해야 됩니다.
        규칙4. 답변은 전문가가 말해주는 것처럼 답변만 주고 부가적인 쓸모없는 말은 하지 말아야 합니다.
        """
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[{"role": "system", "content": "You are an expert assistant specializing in computer hardware and technical specifications. Your task is to provide concise, accurate, and contextually relevant answers based on the conversation history."},
                      {"role": "user", "content": review_data}]
        )
        review = response.choices[0].message.content
        print("재정의: " + review)

        review_data = f"""
        USER: {text}
        의 사용자 질문에 대해 {answers_str}의 견적을 추천 했다
        해당 견적의 총 가격은 {price}입니다.
        {review}의 견적 리뷰가 있습니다.
        해당 견적 리뷰에서 더 좋은 사양을 원하신다면~ 의 사양을 높이거나 가성비를 원하신다면 ~의 사양 낮추는 방법을 추천하는 답변만 주세요.
        """
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[{"role": "system", "content": "You are an expert assistant specializing in computer hardware and technical specifications. Your task is to provide concise, accurate, and contextually relevant answers based on the conversation history."},
                      {"role": "user", "content": review_data}]
        )
        review2 = response.choices[0].message.content
        print("재정의2: " + review2)
        final_review = f"해당 상품의 추천 이유는 다음과 같습니다.\n{review}\n{review2}"
        conversation['chat_history'][-1]['ai'] = f"(AI): 추천하는 견적은 다음과 같습니다.[\n" + \
            f"cpu_name: {detail['CPU']['제품명']}, cpu_price: {detail['CPU']['가격']}, 수량: {detail['CPU']['수량']}\n" + \
            f"mother_name: {detail['메인보드']['제품명']}, mother_price: {detail['메인보드']['가격']}, 수량: {detail['메인보드']['수량']}\n" + \
            f"memory_name: {detail['메모리']['제품명']}, memory_price: {detail['메모리']['가격']}, 수량: {detail['메모리']['수량']}\n" + \
            f"gpu_name: {detail['그래픽카드']['제품명']}, gpu_price: {detail['그래픽카드']['가격']}, 수량: {detail['그래픽카드']['수량']}\n" + \
            f"ssd_name: {detail['SSD']['제품명']}, ssd_price: {detail['SSD']['가격']}, 수량: {detail['SSD']['수량']}\n" + \
            f"case_name: {detail['케이스']['제품명']}, case_price: {detail['케이스']['가격']}, 수량: {detail['케이스']['수량']}\n" + \
            f"power_name: {detail['파워서플라이']['제품명']}, power_price: {detail['파워서플라이']['가격']}, 수량: {detail['파워서플라이']['수량']}\n" + \
            f"cpu_Cooler_name: {detail['CPU쿨러']['제품명']}, cpu_Cooler_price: {detail['CPU쿨러']['가격']}, 수량: {detail['CPU쿨러']['수량']}\n" + \
            f"총가격: {price}]\n"+\
            f"요약: " + review
        redis_client.set(session_id, json.dumps(conversation))
        end_time = time.time() 
        print(f"{end_time - start_time:.2f} seconds") 
        return final_review
    except Exception as e:
        print(f"Error calling the API: {e}")
        return "API 호출 중 에러 발생"

@app.route('/compare_quote', methods=['POST'])
def compare_quote():
    print("compare_quote")
    data = request.json
    text = data['text']
    data_1 = json.loads(data['data1'])
    data_2= json.loads(data['data2'])
    data1 = data_1["parts_price"]
    data2 = data_2["parts_price"]
    total1 = data_1["total_price"]
    total2 = data_2["total_price"]
    c_bench1 = data_1["cpu_benchmarkscore"]
    c_bench2 = data_2["cpu_benchmarkscore"]
    g_bench1 = data_1["gpu_benchmarkscore"]
    g_bench2 = data_2["gpu_benchmarkscore"]
    index1 = "A"
    index2 = "B"
    quote1 = f"""
        {index1}견적
            CPU - {data1["CPU"]["제품명"]}, 가격: {data1["CPU"]["가격"]}, 수량: {data1["CPU"]["수량"]}, 벤치마크 점수: {c_bench1}
            CPU쿨러 - {data1["CPU쿨러"]["제품명"]}, 가격: {data1["CPU쿨러"]["가격"]}, 수량: {data1["CPU쿨러"]["수량"]}
            SSD - {data1["SSD"]["제품명"]}, 가격: {data1["SSD"]["가격"]}, 수량: {data1["SSD"]["수량"]}
            그래픽카드 - {data1["그래픽카드"]["제품명"]}, 가격: {data1["그래픽카드"]["가격"]}, 수량: {data1["그래픽카드"]["수량"]}, 벤치마크 점수: {g_bench1}
            메모리 - {data1["메모리"]["제품명"]}, 가격: {data1["메모리"]["가격"]}, 수량: {data1["메모리"]["수량"]}
            메인보드 - {data1["메인보드"]["제품명"]}, 가격: {data1["메인보드"]["가격"]}, 수량: {data1["메인보드"]["수량"]}
            케이스 - {data1["케이스"]["제품명"]}, 가격: {data1["케이스"]["가격"]}, 수량: {data1["케이스"]["수량"]}
            파워서플라이 - {data1["파워서플라이"]["제품명"]}, 가격: {data1["파워서플라이"]["가격"]}, 수량: {data1["파워서플라이"]["수량"]}
            견적 총 가격: {total1}
            """
    quote2 = f"""
        {index2}견적
            CPU - {data2["CPU"]["제품명"]}, 가격: {data2["CPU"]["가격"]}, 수량: {data2["CPU"]["수량"]}, 벤치마크 점수: {c_bench2}
            CPU쿨러 - {data2["CPU쿨러"]["제품명"]}, 가격: {data2["CPU쿨러"]["가격"]}, 수량: {data2["CPU쿨러"]["수량"]}
            SSD - {data2["SSD"]["제품명"]}, 가격: {data2["SSD"]["가격"]}, 수량: {data2["SSD"]["수량"]}
            그래픽카드 - {data2["그래픽카드"]["제품명"]}, 가격: {data2["그래픽카드"]["가격"]}, 수량: {data2["그래픽카드"]["수량"]}, 벤치마크 점수: {g_bench2}
            메모리 - {data2["메모리"]["제품명"]}, 가격: {data2["메모리"]["가격"]}, 수량: {data2["메모리"]["수량"]}
            메인보드 - {data2["메인보드"]["제품명"]}, 가격: {data2["메인보드"]["가격"]}, 수량: {data2["메인보드"]["수량"]}
            케이스 - {data2["케이스"]["제품명"]}, 가격: {data2["케이스"]["가격"]}, 수량: {data2["케이스"]["수량"]}
            파워서플라이 - {data2["파워서플라이"]["제품명"]}, 가격: {data2["파워서플라이"]["가격"]}, 수량: {data2["파워서플라이"]["수량"]}
            견적 총 가격: {total2}
            """
    try:
        user_message = f"""
        "{text}"의 목적으로 {index1}견적과 {index2}견적 중 고민중이야
            {quote1}
            
            {quote2}
            이 두 견적의 차이점에 대해 전문적이지만 컴퓨터에 대해 잘 모르는 사람이 이해할 수 있도록 쉽게 설명해줘야 합니다.

            규칙1. 각 견적에서 부품을 세세히 비교하여 정확한 비교 분석으로 견적{index1}의 ~(어떠한 부품)보다 견적{index2}의 ~의 ~가 더 빠르고 효율적이다 와 같은 비교 분석 데이터를 해야 합니다.
            규칙2. 답변은 전문가가 말해주는 것처럼 답변만 주고 부가적인 쓸모없는 말은 하지 말아야 합니다.
            규칙3. 마지막으로 설명한 내용을 정리해 주는 말로 견적{index1}은 ~인 사람에게 추천하고 견적{index2}는 ~인 사람에게 추천한다 라고 말해줘야 합니다.
            규칙4. CPU 제품과 그래픽 카드 제품은 벤치마크 점수를 기준으로 성능 비교를 해야 합니다.
            """
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": user_message}]
        )
        answer = response.choices[0].message.content
        print(answer)
        user_message = f"""
        "{text}"질문으로 
            {quote1}
            {quote2}
        중 고민인데
            {answer}의 답변을 받았어

        규칙 1. 이 답변이 제대로 나온 것 같다면 그대로 답변만 해주세요.
        규칙 2. 답변이 제대로 나오지 않거나 오류(가격이나 어휘 등) 및 오타가 있다면 고쳐서 답변만 해 주세요.
        규칙 3. 마지막으로 설명한 내용을 정리해 주는 말로 견적{index1}은 ~인 사람에게 추천하고 견적{index2}는 ~인 사람에게 추천한다 라고 말해줘야 합니다.
        규칙 4. CPU 제품과 그래픽 카드 제품은 벤치마크 점수를 기준으로 성능 비교를 해야 합니다.
        """
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:capstone::9Ms8oKur",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": user_message}]
        )
        answer = response.choices[0].message.content

        return jsonify({'status': 'success', 'data': answer})
    except Exception as e:
        print(f"Error calling the API: {e}")
        return jsonify({'status': 'error', 'data': "error"})

@app.route('/get-quotes', methods=['POST'])
def get_quotes():
    print("get-quotes")
    data = request.json
    origin_text = data['origin_text']
    filtering_text = LLM_filtering(origin_text)
    print(filtering_text)
    try:
        headers = {"Content-Type": "application/json"}
        print("랭체인 실행")
        response = requests.post('http://192.168.205.57:5000/query', headers=headers, json={"text": origin_text})
        response.raise_for_status()
        quotes = response.json()
        print("랭체인 실행 완료")
    except requests.exceptions.RequestException as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    return jsonify(quotes)

@app.route('/get-quote-detail', methods=['POST'])
def get_quote_detail():
    print("get-quote-detail")
    data = request.json
    quotes = data['quotes']
    origin_text = data['text']
    session_id = data.get('session_id')
    detail = quotes["parts_price"]
    session_id = str(uuid.uuid4())
    print("디스크랩션, 피드백 = " + quotes["quote_description"] + quotes["quote_feedback"])
    result = call_by_Rag_LLm(session_id, origin_text, detail, quotes)
    return jsonify({'status': 'success', 'session_id': session_id, 'data': result, 'data2' : None,
                    'cpu_name': detail['CPU']["제품명"], 'cpu_price': detail["CPU"]["가격"], 
                    'mother_name': detail["메인보드"]["제품명"], 'mother_price': detail["메인보드"]["가격"],
                    'memory_name': detail["메모리"]["제품명"], 'memory_price': detail["메모리"]["가격"], 
                    'gpu_name': detail["그래픽카드"]["제품명"], 'gpu_price': detail["그래픽카드"]["가격"],
                    'ssd_name': detail["SSD"]["제품명"], 'ssd_price': detail["SSD"]["가격"], 
                    'case_name': detail["케이스"]["제품명"], 'case_price': detail["케이스"]["가격"],
                    'power_name': detail["파워서플라이"]["제품명"], 'power_price': detail["파워서플라이"]["가격"], 
                    'cpu_Cooler_name': detail["CPU쿨러"]["제품명"], 'cpu_Cooler_price': detail["CPU쿨러"]["가격"],
                    'total_price': quotes["total_price"]})

@app.route('/reset-chat', methods=['POST'])
def reset_chat():
    data = request.json
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({'status': 'error', 'message': 'No session_id provided'}), 400

    try:
        history = redis_client.get(session_id)
        if history:
            conversation = json.loads(history)
            conversation['chat_history'] = conversation['chat_history'][:1]
            redis_client.set(session_id, json.dumps(conversation))
            return jsonify({'status': 'success', 'message': 'Chat history reset'}), 200
        else:
            return jsonify({'status': 'error', 'message': 'No history found for session_id'}), 404
    except Exception as e:
        print(f"Error resetting chat history: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/get-response', methods=['POST'])
def get_response():
    data = request.json
    origin_text = data['origin_text']
    session_id = data.get('session_id')
    if session_id is None:
        return jsonify({'status': 'success', 'session_id': session_id, 'data': "견적을 먼저 추천 받아야 합니다"})
    result = call_by_LLm(session_id, origin_text)
    return jsonify({'status': 'success', 'session_id': session_id, 'data': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
