from fastapi import FastAPI

# Create FastAPI app instance
app = FastAPI(title="Simple FastAPI Example", version="1.0.0")

# Basic GET endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello World from FastAPI!"}

# GET endpoint with path parameter
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"}

# GET endpoint with query parameter
@app.get("/items")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit, "message": "This is a simple items endpoint"}

# POST endpoint
@app.post("/items")
async def create_item(item: dict):
    return {"message": "Item created successfully", "item": item}

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
