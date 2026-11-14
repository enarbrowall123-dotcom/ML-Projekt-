import requests

url = 'http://localhost:9696/predict'

student = {
    'school': 'gp', 'sex': 'f', 'age': 16, 'address': 'u',
    'famsize': 'gt3', 'pstatus': 't', 'medu': 4, 'fedu': 3,
    'mjob': 'teacher', 'fjob': 'other', 'reason': 'reputation',
    'guardian': 'mother', 'traveltime': 1, 'studytime': 3,
    'failures': 0, 'schoolsup': 'no', 'famsup': 'yes', 'paid': 'no',
    'activities': 'yes', 'nursery': 'yes', 'higher': 'yes',
    'internet': 'yes', 'romantic': 'no', 'famrel': 4, 'freetime': 3,
    'goout': 2, 'dalc': 1, 'walc': 1, 'health': 5, 'absences': 2,
}

response = requests.post(url, json=student)
print(response.status_code)
print(response.json())

