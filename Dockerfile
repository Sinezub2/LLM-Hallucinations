# Any image is allowed, but this paticular will build significantly faster
# It is a complete copy of pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
RUN pip3 install pandas
WORKDIR /workspace
COPY . .
ENTRYPOINT ["python3", "solution.py"]
