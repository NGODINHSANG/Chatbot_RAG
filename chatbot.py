import streamlit as st
from rag_processor import rag_search, inspect_faiss, get_all_chunks, get_vector_db_info

st.title("Chatbot RAG với DeepSeek")
st.write("Hỏi bất kỳ câu hỏi nào liên quan đến dữ liệu mẫu!")

tab1, tab2 = st.tabs(["Chat", "Xem Vector DB"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Nhập câu hỏi của bạn:"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            with st.spinner("Đang xử lý..."):
                answer, context = rag_search(query)
                st.markdown(f"**Câu trả lời:** {answer}")
            st.session_state.messages.append({"role": "assistant", "content": answer})


with tab2:
    st.subheader("Nội dung trong Vector DB (FAISS)")
    st.write("Dưới đây là tất cả các đoạn dữ liệu đã được lưu trong FAISS:")
    chunks = get_all_chunks()
    for i, chunk in enumerate(chunks):
        st.write(f"**Chunk {i+1}:** {chunk}")
    
    info = get_vector_db_info()
    st.write(f"**Tổng số đoạn:** {info['total_chunks']}")
    st.write(f"**Kích thước vector:** {info['vector_dimension']}")

    st.subheader("Kiểm tra FAISS với câu hỏi")
    test_query = st.text_input("Nhập câu hỏi để kiểm tra FAISS:")
    if st.button("Kiểm tra"):
        result = inspect_faiss(test_query)
        st.write(f"**Câu hỏi:** {result['query']}")
        st.write(f"**Chỉ số đoạn tìm được:** {result['indices']}")
        st.write(f"**Khoảng cách (distance):** {result['distances']}")
        st.write(f"**Nội dung đoạn:**")
        for ctx in result['contexts']:
            st.write(f"- {ctx}")