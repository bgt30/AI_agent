import os
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from transformers import pipeline
from konlpy.tag import Okt

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# GPT-4o-mini 모델 초기화
llm = ChatOpenAI(
    temperature=0,  # 모델의 응답 온도 설정
    model_name="gpt-4o-mini",  # 사용할 모델 이름
    openai_api_key=openai_api_key  # OpenAI API 키
)

# YouTube API 클라이언트 초기화
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# 분류용 프롬프트 템플릿 (AI 관련 질문을 분류하기 위한 템플릿)
classify_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    당신은 인공지능(AI)과 관련된 질문에 답변을 제공하고 유튜브 동영상을 추천하는 AI 어시스턴트입니다.

    관련성 판단:
       사용자의 질문: "{query}"
       - 질문이 AI, 인공지능 또는 그 응용 분야와 직접적 혹은 간접적으로 관련이 있는지 판단하세요.
       - 관련이 있다면 "AI-RELATED"로 시작하세요.
       - 관련이 없다면 "NOT-AI-RELATED"로 시작하고, 더 이상 진행하지 마세요.
    """
)
# LLMChain을 이용해 AI 관련성 판단 모델 생성
classify_chain = LLMChain(llm=llm, prompt=classify_prompt)

# Hugging Face 모델 초기화
nlp_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# 형태소 분석기 초기화
okt = Okt()

# 질문 분석 및 쿼리 정제 함수
def refine_query_with_nlp_and_tokenizer(query):
    # 1. NLP로 키워드 추출
    embeddings = nlp_model(query)
    keywords = " ".join([re.sub(r'[^\w\s]', '', word) for word in query.split()])

    # 2. 형태소 분석기로 조사/접미사 제거
    tokens = okt.nouns(query)
    refined_tokens = " ".join(tokens)

    # NLP 기반 키워드와 형태소 분석 키워드를 결합
    return f"{keywords} {refined_tokens}".strip()

# 유튜브 댓글을 가져오는 함수
def get_top_comments(video_id, max_comments=5):
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            order="relevance"
        )
        response = request.execute()
        comments = [
            item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            for item in response["items"]
        ]
        return comments
    except Exception:
        return ["댓글을 불러올 수 없습니다."]

# 유튜브 영상을 검색하고 댓글을 가져오는 함수
def search_youtube_videos_with_comments(query, max_results=5, max_comments=3):
    refined_query = refine_query_with_nlp_and_tokenizer(query)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([refined_query])

    request = youtube.search().list(
        part="snippet",
        q=refined_query,
        type="video",
        maxResults=max_results,
        order="viewCount"
    )
    response = request.execute()
    videos = []
    if "items" in response:
        for item in response["items"]:
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            video_id = item["id"]["videoId"]

            video_text = title + " " + description
            video_text = refine_query_with_nlp_and_tokenizer(video_text)
            video_tfidf_matrix = tfidf_vectorizer.transform([video_text])

            similarity = cosine_similarity(tfidf_matrix, video_tfidf_matrix)[0][0]
            comments = get_top_comments(video_id, max_comments)

            videos.append({
                "title": title,
                "description": description,
                "video_id": video_id,
                "similarity": similarity,
                "comments": comments
            })

    videos = sorted(videos, key=lambda x: x["similarity"], reverse=True)
    return videos

# Streamlit UI 설정
st.title("유튜브 영상 추천 AI🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("질문을 입력하세요! 종료하려면 '종료'를 입력하세요."):
    if user_input.lower() == "exit":
        st.info("채팅을 종료합니다.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                classification = classify_chain.run({"query": user_input})
                if classification.startswith("NOT-AI-RELATED"):
                    error_message = "AI 관련 질문을 해주세요"
                    st.subheader(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                else:
                    videos = search_youtube_videos_with_comments(user_input)
                    if videos:
                        st.subheader("관련 유튜브 영상 추천:")
                        assistant_response = ""
                        for video in videos:
                            st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
                            video_info = f"**설명:** {video['description']}\n"
                            comments_info = "\n".join([f"  - {comment}" for comment in video["comments"]])
                            full_info = f"{video_info}\n\n**댓글:**\n{comments_info}\n\n---"
                            assistant_response += full_info
                            st.markdown(full_info)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    else:
                        error_message = "관련 영상을 찾을 수 없습니다."
                        st.subheader(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                st.error(f"RAG 호출 중 오류가 발생했습니다: {str(e)}")
