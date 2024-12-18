import streamlit as st
from knn_base import predict_student_result

# Cấu hình giao diện Streamlit
st.title("Phân Loại Kết Quả Học Sinh")
st.subheader("Nhập điểm các môn học để dự đoán")

# Các input để nhập điểm
hindi = st.number_input("Điểm môn Hindi:", min_value=0.0, max_value=100.0, step=0.1)
english = st.number_input("Điểm môn English:", min_value=0.0, max_value=100.0, step=0.1)
science = st.number_input("Điểm môn Science:", min_value=0.0, max_value=100.0, step=0.1)
maths = st.number_input("Điểm môn Maths:", min_value=0.0, max_value=100.0, step=0.1)
history = st.number_input("Điểm môn History:", min_value=0.0, max_value=100.0, step=0.1)
geography = st.number_input("Điểm môn Geography:", min_value=0.0, max_value=100.0, step=0.1)
total_score = hindi + english + science + maths + history + geography
st.write(f"**Tổng điểm:** {total_score}")

# Nút để thực hiện dự đoán
if st.button("Dự đoán"):
    # Gọi hàm dự đoán từ file knn_classifier.py
    prediction = predict_student_result(hindi, english, science, maths, history, geography, total_score)
    st.success(f"Kết quả dự đoán: {prediction}")