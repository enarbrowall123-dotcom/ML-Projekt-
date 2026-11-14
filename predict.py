import pickle
from typing import Literal

from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn


YesNo = Literal['yes', 'no']
Job = Literal['teacher', 'health', 'services', 'at_home', 'other']


class Student(BaseModel):
    school: Literal['gp', 'ms']
    sex: Literal['f', 'm']
    address: Literal['u', 'r']
    famsize: Literal['le3', 'gt3']
    pstatus: Literal['t', 'a']
    mjob: Job
    fjob: Job
    reason: Literal['home', 'reputation', 'course', 'other']
    guardian: Literal['mother', 'father', 'other']
    schoolsup: YesNo
    famsup: YesNo
    paid: YesNo
    activities: YesNo
    nursery: YesNo
    higher: YesNo
    internet: YesNo
    romantic: YesNo

    age: int = Field(..., ge=15, le=22)
    medu: int = Field(..., ge=0, le=4)
    fedu: int = Field(..., ge=0, le=4)
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=4)
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    dalc: int = Field(..., ge=1, le=5)
    walc: int = Field(..., ge=1, le=5)
    health: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0, le=93)


class PredictResponse(BaseModel):
    predicted_grade: float
    at_risk: bool


app = FastAPI(title='student-grade-prediction')

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(student):
    result = pipeline.predict([student])[0]
    # grades are on a 0 to 20 scale
    return float(min(max(result, 0), 20))


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(student: Student) -> PredictResponse:
    grade = predict_single(student.model_dump())

    return PredictResponse(
        predicted_grade=round(grade, 2),
        at_risk=grade < 10
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)
    