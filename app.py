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
                            
                            # 유튜브 영상 설명 출력
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