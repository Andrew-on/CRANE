FROM continuumio/anaconda3

WORKDIR /app

COPY . /app

RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/crane_env/bin:$PATH

EXPOSE 8000

CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8123", "CRANE.py"]
