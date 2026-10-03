from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


# Cấu hình trang
st.set_page_config(
    page_title="Demo Phân Loại Hoa Iris",
    page_icon="🌸",
    layout="wide",
)


# Tải dữ liệu và huấn luyện mô hình một lần
@st.cache_resource
def load_model():
    iris_data = load_iris()
    classifier = GaussianNB()
    classifier.fit(iris_data.data, iris_data.target)
    return iris_data, classifier


# Tính độ chính xác tham khảo trên tập Test 80/20
@st.cache_resource
def calculate_test_accuracy():
    iris_data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris_data.data,
        iris_data.target,
        test_size=0.2,
        random_state=42,
        stratify=iris_data.target,
    )

    test_model = GaussianNB()
    test_model.fit(X_train, y_train)
    predictions = test_model.predict(X_test)

    return accuracy_score(y_test, predictions)


iris, model = load_model()
test_accuracy = calculate_test_accuracy()


# Ảnh sẽ được thêm vào thư mục images trong repository sau
image_dir = Path(__file__).parent / "images"

species_images = {
    "setosa": image_dir / (
        "1790997729001_3952417606131047985_"
        "g7586510699556540576_e3084215504a0fedd6d558ea6c050578.jpg"
    ),
    "versicolor": image_dir / (
        "1790997729290_3952417606131047985_"
        "g7586510699556540576_c7e6b309ff702208f70fd86757b5f52c.jpg"
    ),
    "virginica": image_dir / (
        "1790997729742_3952417606131047985_"
        "g7586510699556540576_ed6311813705d3b92fff61f0683d18c0.jpg"
    ),
}


# Tạo bảng dữ liệu để xem trong ứng dụng
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target
iris_df["species"] = [
    iris.target_names[target] for target in iris.target
]


# Tiêu đề và hướng dẫn
st.title("🌸 Demo Phân Loại Hoa Iris Dataset")
st.subheader("Mô hình: Gaussian Naive Bayes")
st.write(
    "Nhập kích thước lá đài và cánh hoa, sau đó nhấn "
    "**Dự đoán ngay** để xem kết quả."
)


# Thông tin mô hình ở thanh bên
with st.sidebar:
    st.header("Thông tin mô hình")
    st.write("**Thuật toán:** Gaussian Naive Bayes")
    st.write("**Dữ liệu:** Iris Dataset")
    st.metric("Độ chính xác trên tập Test", f"{test_accuracy:.2%}")
    st.caption("Chia tập dữ liệu 80/20, random_state=42, stratify=y.")


# Chia giao diện thành cột nhập liệu và cột kết quả
input_col, result_col = st.columns(2)

with input_col:
    st.header("① Thông số đầu vào")

    with st.form("iris_prediction_form"):
        sepal_length = st.slider(
            "Chiều dài lá đài (Sepal Length - cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.1,
            step=0.1,
        )
        sepal_width = st.slider(
            "Chiều rộng lá đài (Sepal Width - cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.5,
            step=0.1,
        )
        petal_length = st.slider(
            "Chiều dài cánh hoa (Petal Length - cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.4,
            step=0.1,
        )
        petal_width = st.slider(
            "Chiều rộng cánh hoa (Petal Width - cm)",
            min_value=0.1,
            max_value=2.5,
            value=0.2,
            step=0.1,
        )

        submitted = st.form_submit_button(
            "🔍 Dự đoán ngay",
            use_container_width=True,
        )


# Chỉ dự đoán khi người dùng nhấn nút
if submitted:
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )
    predicted_index = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.session_state["iris_result"] = {
        "species": iris.target_names[predicted_index],
        "probabilities": probabilities.tolist(),
    }


# Hiển thị kết quả
with result_col:
    st.header("② Kết quả phân loại")

    result = st.session_state.get("iris_result")

    if result is None:
        st.info("Nhập thông số bên trái rồi nhấn **Dự đoán ngay**.")
    else:
        predicted_species = result["species"]
        probabilities = result["probabilities"]

        st.success(f"🌼 Loài hoa dự đoán: **{predicted_species.upper()}**")

        st.subheader("Xác suất của từng loài")
        probability_df = pd.DataFrame(
            {"Xác suất": probabilities},
            index=[name.title() for name in iris.target_names],
        )
        st.bar_chart(probability_df)

        st.subheader(f"Hình ảnh hoa {predicted_species.title()}")
        image_path = species_images[predicted_species]

        if image_path.is_file():
            st.image(
                str(image_path),
                caption=f"Iris {predicted_species.title()}",
                use_container_width=True,
            )
        else:
            st.warning(
                "Chưa tìm thấy ảnh. Sau khi lưu app.py, hãy tải ảnh vào "
                "thư mục images trong repository."
            )


# Bonus: xem bảng dữ liệu Iris
with st.expander("Xem bảng dữ liệu Iris"):
    st.dataframe(iris_df, use_container_width=True)
