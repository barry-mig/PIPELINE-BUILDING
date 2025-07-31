# Simple FastAPI Example

This is a basic FastAPI application demonstrating the simplest use cases.

## Features

- Basic GET endpoint (`/`)
- GET endpoint with path parameter (`/hello/{name}`)
- GET endpoint with query parameters (`/items`)
- POST endpoint (`/items`)

## Installation

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Using Python directly
```bash
python main.py
```

### Option 2: Using uvicorn command
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Testing the API

Once the server is running, you can:

1. **Visit the interactive API docs**: http://localhost:8000/docs
2. **Visit the alternative docs**: http://localhost:8000/redoc

### Example API calls:

- **GET** `http://localhost:8000/` - Returns a hello world message
- **GET** `http://localhost:8000/hello/John` - Returns a personalized greeting
- **GET** `http://localhost:8000/items?skip=0&limit=5` - Returns items with pagination
- **POST** `http://localhost:8000/items` - Create a new item (send JSON data in the body)

## What's included

- FastAPI app with multiple endpoint types
- Automatic API documentation (Swagger UI)
- Path parameters and query parameters examples
- Basic POST endpoint for receiving dataE BUILDING

Description of your project.

## Setup
`ash
pip install -r [requirements.txt](http://_vscodecontentref_/1)
pip install -r requirements.txt
`

## Usage
`ash
python main.py
`

