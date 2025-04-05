import PyPDF2
from docx import Document
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file_paths):
    all_chunks = []
    for file_path in file_paths:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".pdf":
            with open(file_path, "rb") as file:
                pdf = PyPDF2.PdfReader(file)
                text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif file_extension == ".docx":
            doc = Document(file_path)
            text = "".join(para.text for para in doc.paragraphs if para.text)
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
        else:
            print(f"Định dạng file {file_extension} không được hỗ trợ!")
            continue
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        all_chunks.extend(chunks)
    return all_chunks

def create_and_save_faiss_index(file_paths, index_file="faiss_index.bin", chunks_file="chunks.pkl"):
    
    chunks = load_data(file_paths) if file_paths else [
        "Câu hỏi: Chính sách bảo hành sản phẩm là gì? Câu trả lời: Chính sách bảo hành kéo dài 12 tháng kể từ ngày mua.",
        "Câu hỏi: Làm sao để được bảo hành? Câu trả lời: Khách hàng cần giữ hóa đơn mua hàng để được bảo hành.",
        "Câu hỏi: Sản phẩm hư do lỗi người dùng thì sao? Câu trả lời: Sản phẩm hư hỏng do lỗi người dùng sẽ không được bảo hành."
    ]
    
    
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"Đã tạo và lưu FAISS index vào {index_file} với {len(chunks)} đoạn.")

if __name__ == "__main__":

    file_paths = [
        "data.pdf",
    ]
    create_and_save_faiss_index(file_paths)