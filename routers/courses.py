from fastapi import APIRouter, HTTPException, Query, Path
from typing import Optional, Dict, Union, List
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
    topic: Union[str, List[str]] = Query(..., title="Course Topic", description="Enter the main topic (e.g., Thermodynamics) or a list of topics"),
    use_azure: bool = Query(False, title="Use Azure", description="Set to True to use Azure OpenAI"),
    filename: Optional[str] = Query(None, title="Filename", description="Optional custom filename for saving the course"),
    use_web: bool = Query(False, title="Use Web", description="Set to True to use WebSearchRAG")
):
    try:
        # Handle both single topic and list of topics
        if isinstance(topic, str):
            topics = [topic]
            single_topic = True
        else:
            topics = topic
            single_topic = False
            
        all_course_data = {}
        filepaths = []
        
        for t in topics:
            logger.info(f"Creating course for topic: {t}")
            course_builder = CourseBuilder(use_azure=use_azure, use_web=use_web)
            course_data = course_builder.build_topic(t)
            
            # Generate filename for this topic
            current_filename = None
            if filename:
                if len(topics) > 1:
                    base, ext = os.path.splitext(filename)
                    current_filename = f"{base}_{t.lower().replace(' ', '_')}{ext or '.json'}"
                else:
                    current_filename = filename
            
            # Save the file
            filepath = course_builder.save_course_to_file(
                course_data,
                filename=current_filename or f"{t.lower().replace(' ', '_')}_course.json"
            )
            
            all_course_data[t] = course_data
            filepaths.append(filepath)

        response = {
            "status": "success",
            "message": f"Successfully created course{'s' if not single_topic else ''} for {len(topics)} topic{'s' if len(topics) > 1 else ''}",
            "topics": topics,
            "web_search_used": use_web,
            "filepaths": filepaths[0] if single_topic else filepaths,
        }
        
        # For backward compatibility, include single topic response structure
        if single_topic:
            response.update({
                "topic": topics[0],
                "filepath": filepaths[0],
                "course_data": all_course_data[topics[0]]
            })
        else:
            response["courses_data"] = all_course_data
            
        return response

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
