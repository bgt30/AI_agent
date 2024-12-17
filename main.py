# 필요한 라이브러리들을 임포트합니다.
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

# 텍스트에서 특수 문자를 제거하고 소문자로 변환하는 함수 (TF-IDF 처리를 위한 전처리)
def extract_keywords(text):
    text = re.sub(r'[^\w\s]', '', text)  # 특수 문자는 제거
    return text.lower()  # 소문자로 변환하여 반환

# 유튜브 댓글을 가져오는 함수
def get_top_comments(video_id, max_comments=5):
    try:
        # 유튜브 API를 통해 댓글 목록을 가져옴
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            order="relevance"
        )
        response = request.execute()
        # 댓글 목록 추출
        comments = [
            item["snippet"]["topLevelComment"]["snippet"]["textOriginal"]
            for item in response["items"]
        ]
        return comments
    except Exception:
        # 댓글을 가져올 수 없는 경우
        return ["댓글을 불러올 수 없습니다."]
    
# 유튜브 영상을 검색하고 댓글을 가져오는 함수
def search_youtube_videos_with_comments(query, max_results=5, max_comments=3):
    # 사용자 쿼리를 키워드 추출 후 TF-IDF 방식으로 벡터화
    refined_query = extract_keywords(query)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([refined_query])

    # 유튜브 검색 API 호출
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results,
        order="viewCount"  # 조회수 기준으로 정렬
    )
    response = request.execute()
    videos = []
    if "items" in response:
        for item in response["items"]:
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            video_id = item["id"]["videoId"]

            # 영상의 제목과 설명을 결합하여 벡터화
            video_text = title + " " + description
            video_text = extract_keywords(video_text)
            video_tfidf_matrix = tfidf_vectorizer.transform([video_text])

            # 코사인 유사도 계산
            similarity = cosine_similarity(tfidf_matrix, video_tfidf_matrix)[0][0]

            # 댓글을 가져옴
            comments = get_top_comments(video_id, max_comments)

            # 영상 정보와 유사도, 댓글을 리스트에 저장
            videos.append({
                "title": title,
                "description": description,
                "video_id": video_id,
                "similarity": similarity,
                "comments": comments
            })

    # 유사도 순으로 영상 정렬
    videos = sorted(videos, key=lambda x: x["similarity"], reverse=True)
    return videos


# Streamlit UI 설정
st.title("AI YouTube Video Recommendation")  # 앱 제목

# 채팅 기록을 위한 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록을 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("질문을 입력하세요! 종료하려면 '종료'를 입력하세요."):
    # 종료 조건 처리
    if user_input.lower() == "exit":
        st.info("채팅을 종료합니다.")
    else:
        # 사용자 메시지를 세션에 기록
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG 모델을 호출하여 AI 관련성 판단
        with st.chat_message("assistant"):
            try:
                # GPT 모델로 질문의 관련성 분류
                classification = classify_chain.run({"query": user_input})
                if classification.startswith("NOT-AI-RELATED"):
                    # AI 관련이 없는 질문인 경우 처리
                    error_message = "AI 관련 질문을 해주세요"
                    st.subheader(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                else:
                    # 유튜브 검색 실행
                    videos = search_youtube_videos_with_comments(user_input)
                    if videos:
                        st.subheader("🔍 관련 유튜브 동영상 추천:")
                        assistant_response = ""
                        for video in videos:
                            # 유튜브 동영상 삽입 (링크를 통해 바로 실행)
                            st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
                            
                            # 유튜브 영상 설명만 출력
                            video_info = f"**설명:** {video['description']}\n"
                            
                            # 댓글 정보 출력
                            comments_info = "\n".join([f"  - {comment}" for comment in video["comments"]])
                            
                            # 영상 정보와 댓글을 합쳐서 출력할 내용 준비
                            full_info = f"{video_info}\n\n**댓글:**\n{comments_info}\n\n---"
                            
                            # 화면에 출력할 내용에 추가
                            assistant_response += full_info
                            
                            # 스트림릿에 해당 정보를 출력
                            st.markdown(full_info)
                        
                        # 어시스턴트의 응답을 채팅 기록에 추가
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                    else:
                        # 관련 영상을 찾을 수 없는 경우
                        error_message = "관련 영상을 찾을 수 없습니다."
                        st.subheader(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

            except Exception as e:
                # RAG 호출 중 오류 발생 시 처리
                st.error(f"RAG 호출 중 오류가 발생했습니다: {str(e)}")


'''
사용자 질문 정제하는 거 보완
이전 내용 기억하는 거 추가
'''