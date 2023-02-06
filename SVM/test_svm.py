from utils import *

data = load_test_data("test_data")
svm_model = load_svm_model("model.pkl")
prediction = get_prediction(data, svm_model)
save_prediction("result.txt", prediction)
