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

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Initialize GPT-4o-mini model
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o-mini",
    openai_api_key=openai_api_key
)

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Classification prompt template
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
classify_chain = LLMChain(llm=llm, prompt=classify_prompt)

# TF-IDF and cosine similarity functions
def extract_keywords(text):
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

# 인기 댓글 가져오기 함수
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
    
def search_youtube_videos_with_comments(query, max_results=5, max_comments=3):
    # Refine query with TF-IDF
    refined_query = extract_keywords(query)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([refined_query])

    request = youtube.search().list(
        part="snippet",
        q=query,
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

            # Vectorize title and description
            video_text = title + " " + description
            video_text = extract_keywords(video_text)
            video_tfidf_matrix = tfidf_vectorizer.transform([video_text])

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix, video_tfidf_matrix)[0][0]

            # Get top comments
            comments = get_top_comments(video_id, max_comments)

            videos.append({
                "title": title,
                "description": description,
                "video_id": video_id,
                "similarity": similarity,
                "comments": comments
            })

    # Sort by similarity
    videos = sorted(videos, key=lambda x: x["similarity"], reverse=True)
    return videos

# Streamlit UI 설정
st.title("AI YouTube Video Recommendation")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("질문을 입력하세요! 종료하려면 '종료'를 입력하세요."):
    # 종료 조건 처리
    if user_input.lower() == "exit":
        st.info("채팅을 종료합니다.")
    else:
        # 사용자 메시지를 히스토리에 추가
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG 모델 호출
        with st.chat_message("assistant"):
            try:
                # GPT 모델로 관련성 판단
                classification = classify_chain.run({"query": user_input})
                if classification.startswith("NOT-AI-RELATED"):
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
                            # 유튜브 동영상 삽입
                            st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
                            
                            # 유튜브 영상 정보 (설명만 남김)
                            video_info = f"**설명:** {video['description']}\n"
                            
                            # 댓글 정보
                            comments_info = "\n".join([f"  - {comment}" for comment in video["comments"]])
                            
                            # Combine all information
                            full_info = f"{video_info}\n\n**댓글:**\n{comments_info}\n\n---"
                            
                            # Append to assistant response
                            assistant_response += full_info
                            
                            # Display video information and comments in Streamlit
                            st.markdown(full_info)
                        
                        # Append the assistant's response to the chat history
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                    else:
                        error_message = "관련 영상을 찾을 수 없습니다."
                        st.subheader(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

            except Exception as e:
                st.error(f"RAG 호출 중 오류가 발생했습니다: {str(e)}")