from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "service": "dwpose-ai-server"}


@app.get("/health")
def health():
    return {"status": "healthy"}
