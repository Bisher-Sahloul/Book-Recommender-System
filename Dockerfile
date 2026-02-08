FROM python:3.9-alpine

WORKDIR /app 

COPY . /app 

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python" , "main.py"]