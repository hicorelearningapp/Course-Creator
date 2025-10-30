from fastapi import FastAPI
from routers import courses

app = FastAPI(
    title="Course Creator API",
    description="API for generating and managing educational courses",
    version="1.0.0"
)

# Include the router
app.include_router(courses.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8085, reload=True)