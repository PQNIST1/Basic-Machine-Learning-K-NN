import numpy as np

# Hàm tính khoảng cách Euclid giữa hai điểm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Hàm dự đoán điểm tiêu dùng của khách hàng mới dựa trên k điểm láng giềng gần nhất
def predict_spending_score(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_scores = [y_train[i] for i in k_indices]
    return np.mean(k_nearest_scores)

# Hàm phân loại khách hàng dựa trên tuổi, thu nhập và dự đoán điểm tiêu dùng

def classify_customer(x_test, predicted_spending_score, gender):
    age, income = x_test[:2]
    is_male = gender == 'Male'

    # Phân loại dựa trên giới tính và lứa tuổi
    classification = (
        "Trẻ Em" if age <= 12 else
        "Thanh Thiếu Niên" if age <= 18 else
        "Người Trưởng Thành" if age <= 35 else
        "Người Trung Niên" if age <= 50 else
        "Người Già"
    ) + (
        " Nam" if is_male else " Nữ"
    ) + (
        " Có Thu Nhập Thấp" if income <= 20 else " Có Thu Nhập Trung Bình" if income <= 50 else " Có Thu Nhập Cao"
    ) + (
        " Tiêu Dùng Thấp" if predicted_spending_score <= 50 else " Tiêu Dùng Trung Bình" if predicted_spending_score <= 80 else " Tiêu Dùng Cao"
    )

    return classification
