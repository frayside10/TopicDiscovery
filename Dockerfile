FROM python:3

WORKDIR /home/app

COPY ./requirements.txt . 

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "notebooks/run_pipeline_sample.py"]
