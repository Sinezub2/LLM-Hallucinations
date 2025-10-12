FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
RUN pip3 install pandas
WORKDIR /workspace
COPY . .
ENTRYPOINT ["python3", "solution.py"]
