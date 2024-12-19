import os
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timedelta
import kss

# 환경 변수 로드
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# YouTube API 클라이언트 초기화
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# CLIP 모델 로드
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 최근 N일 동안 업로드된 영상만 검색
def get_recent_date_filter(days=365):
    recent_date = datetime.utcnow() - timedelta(days=days)
    return recent_date.strftime("%Y-%m-%dT%H:%M:%SZ")

# 질문 분석 및 쿼리 정제 함수 (KSS 활용)
def refine_query(query):
    # KSS로 문장 분리
    sentences = kss.split_sentences(query)
    # 각 문장을 단어로 분리하고 결합
    refined_query = " ".join(sentences)
    return refined_query

# 이미지 임베딩 생성 함수
def get_image_embedding(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_inputs = clip_processor(images=image, return_tensors="pt")
        image_embedding = clip_model.get_image_features(**image_inputs)
        return image_embedding.detach().numpy()
    except Exception as e:
        print(f"이미지 임베딩 생성 오류: {e}")
        return None

# 텍스트 임베딩 생성 함수
def get_text_embedding(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding.detach().numpy()

# 유튜브 댓글 가져오기
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
            for item in response.get("items", [])
        ]
        return comments
    except Exception as e:
        print(f"댓글 불러오기 오류: {e}")
        return ["댓글을 불러올 수 없습니다."]

# 유튜브 동영상 검색 및 멀티모달 유사도 계산
def search_youtube_videos_with_multimodal(query, max_results=10, max_comments=3):
    ai_related_keywords = '''인공지능 (AI)
기계 학습 (머신러닝)
딥러닝
인공 신경망
지도 학습
비지도 학습
강화 학습
반지도 학습
전이 학습
생성형 인공지능
자연어 처리 (NLP)
컴퓨터 비전
로보틱스
인지 컴퓨팅
인공 일반 지능 (AGI)
인공 협소 지능 (ANI)
인공 초지능 (ASI)'''
    refined_query = f"{refine_query(query)} {ai_related_keywords}"
    text_embedding = get_text_embedding(refined_query)

    request = youtube.search().list(
        part="snippet",
        q=refined_query,
        type="video",
        maxResults=max_results,
        order="viewCount",
        publishedAfter=get_recent_date_filter(365)  # 최근 1년 내 업로드된 영상만 검색
    )
    response = request.execute()
    videos = []

    if "items" in response:
        for item in response["items"]:
            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            video_id = item["id"]["videoId"]
            thumbnail_url = item["snippet"]["thumbnails"]["high"]["url"]

            # 텍스트 임베딩 생성
            video_text = title + " " + description
            video_text_embedding = get_text_embedding(video_text)

            # 이미지 임베딩 생성
            image_embedding = get_image_embedding(thumbnail_url)

            # 멀티모달 유사도 계산
            if image_embedding is not None:
                text_similarity = cosine_similarity(text_embedding, video_text_embedding)[0][0]
                image_similarity = cosine_similarity(text_embedding, image_embedding)[0][0]
                multimodal_similarity = (text_similarity + image_similarity) / 2  # 단순 평균
            else:
                multimodal_similarity = 0  # 이미지 분석 실패 시 유사도 0 처리

            # 댓글 가져오기
            comments = get_top_comments(video_id, max_comments)

            videos.append({
                "title": title,
                "description": description,
                "video_id": video_id,
                "similarity": multimodal_similarity,
                "thumbnail_url": thumbnail_url,
                "comments": comments
            })

    # 상위 N개의 결과만 반환
    videos = sorted(videos, key=lambda x: x["similarity"], reverse=True)[:max_results]
    return videos

# Streamlit UI 설정
st.title("유튜브 영상 검색 AI🤖")

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
                videos = search_youtube_videos_with_multimodal(user_input)
                if videos:
                    st.subheader("관련 유튜브 영상 추천:")
                    for video in videos:
                        st.video(f"https://www.youtube.com/watch?v={video['video_id']}")
                        st.image(video["thumbnail_url"], caption=video["title"])
                        st.markdown(f"**설명:** {video['description']}")
                        st.markdown("**댓글:**")
                        for comment in video["comments"]:
                            st.markdown(f"- {comment}")
                        st.markdown("---")
                else:
                    st.subheader("관련 영상을 찾을 수 없습니다.")
            except Exception as e:
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")
