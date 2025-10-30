from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional, Dict
import logging, sys, os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_course import CourseBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/courses",
    tags=["Courses"],
    responses={404: {"description": "Course not found"}},
)

@router.post(
    "/",
    summary="Create a new course",
    description="""
    Create a new AI-generated course for the given topic.

    - **topic**: The main subject or title of the course.  
    - **use_azure**: Whether to use Azure OpenAI for generation (default: False).  
    - **filename**: Optional custom filename for the saved JSON.
    """
)
async def create_course(
    topic: str = Query(..., title="Course Topic", description="Enter the main topic (e.g., Thermodynamics)"),
    use_azure: bool = Query(False, title="Use Azure", description="Set to True to use Azure OpenAI"),
    filename: Optional[str] = Query(None, title="Filename", description="Optional custom filename for saving the course"),
    use_web: bool = Query(False, title="Use Web", description="Set to True to use WebSearchRAG")
):
    try:
        logger.info(f"Creating course for topic: {topic}")
        course_builder = CourseBuilder(use_azure=use_azure, use_web=use_web)
        course_data = course_builder.build_topic(topic)

        # Always save the file, use topic as filename if not provided
        filepath = course_builder.save_course_to_file(
            course_data, 
            filename=filename if filename else f"{topic.lower().replace(' ', '_')}_course.json"
        )

        return {
            "status": "success",
            "message": "Course created successfully",
            "topic": topic,
            "web_search_used":use_web,
            "filepath": filepath,
            "course_data": course_data,
        }

    except Exception as e:
        logger.error(f"Error creating course: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create course: {str(e)}")


@router.get(
    "/{topic}",
    summary="Get course by topic",
    description="""
    Retrieve an automatically generated course structure for a given topic.

    **Path Parameter:**
    - `topic`: The topic name to generate or retrieve course data for.

    **Query Parameter:**
    - `use_azure`: Set to `true` to use Azure OpenAI model.

    **Returns:**  
    JSON with course data for the topic.
    """,
)
async def get_course(
    topic: str = Path(..., description="Topic name to retrieve the course for"),
    use_azure: bool = Query(False, description="Use Azure OpenAI (default: False)"),
    use_web: bool = Query(False, description="Enable live web RAG search (default: False)")
):
    try:
        logger.info(f"Fetching course for topic: {topic}")
        course_builder = CourseBuilder(use_azure=use_azure, use_web=use_web)
        course_data = course_builder.build_topic(topic)

        return {
            "status": "success",
            "topic": topic,
            'course_data': course_data
        }

    except Exception as e:
        logger.error(f"Error fetching course: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch course: {str(e)}")
