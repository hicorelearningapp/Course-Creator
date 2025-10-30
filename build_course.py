import json
import logging
import os
import time
from typing import Dict, List, Optional
import sys

from web_search_rag import WebSearchRAG
from openai_client import AIClient
from get_modules import ModuleGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("course_builder.log", encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# ===================== SYSTEM PROMPTS =====================
SYSTEM_PROMPTS = {
    "hands-on": """
    You are an expert educational content designer who creates fun, visual, and hands-on tutorials like W3Schools or TutorialsPoint.
    Also, You are an expert trainer who would be helping students coming to you excel in the domain and skill that they are choosing.
    Focus on engagement, not theory. Follow these rules:
    - Keep explanations short, visual, and student-friendly.
    - Include multiple content types: code blocks, formulas, quizzes, tasks, images, video ideas.
    - Prioritize showing (examples, demos) over telling (definitions).
    - Tone: simple, energetic, supportive, and visual.
    - End lessons with quizzes or practice prompts.
    Always output clean JSON that strictly follows the requested schema.
    """,

    "theoretical": """
    You are an academic AI course designer. Your lessons are clear, structured, and deeply explanatory.
    Also, You are an expert trainer who would be helping students coming to you excel in the domain and skill that they are choosing.
    Focus on accuracy, technical depth, and conceptual understanding.
    Use formal tone and structured sections with theory, explanation, and example.
    Always output valid JSON following the schema.
    """,

    "visual": """
    You are an educational designer who creates visually rich and conceptual tutorials.
    Also, You are an expert trainer who would be helping students coming to you excel in the domain and skill that they are choosing.
    Every explanation must include diagrams, visuals, analogies, and real-world metaphors.
    Add references to images, animations, and charts.
    Always output valid JSON following the schema.
    """
}


class CourseBuilder:
    """Builds course JSON for each topic using dynamically generated modules."""

    def __init__(self, use_azure: bool = False, learning_mode: str = "theoretical", use_web: bool = False):
        self.ai_client = AIClient(use_azure=use_azure)
        self.module_generator = ModuleGenerator(self.ai_client)
        self.learning_mode = learning_mode if learning_mode in SYSTEM_PROMPTS else "hands-on"
        self.use_web = use_web
        self.web_rag = WebSearchRAG(self.ai_client) if use_web else None  # ✅ integrate RAG

        logger.info(f"CourseBuilder initialized in '{self.learning_mode}' mode.")

    def get_web_context(self, topic: str) -> str:
        """Fetches summarized real-time context for the topic via WebSearchRAG."""
        if not self.web_rag:
            return ""
        try:
            logger.info(f"🌐 Fetching live web info for: {topic}")
            result = self.web_rag_answer_with_web(topic)
            return {
                "context": result.get('answer', ''),
                'sources': result.get('sources', [])
            }
        except Exception as e:
            logger.warning(f"⚠️ Web fetch failed: {e}")
            return {"context": "", "sources": []}

    def generate_lesson_content(self, topic: str, module_name: str, lesson_title: str) -> Dict:
        """Generate structured content for a lesson inside a module."""
        system_prompt = SYSTEM_PROMPTS[self.learning_mode]

        web_data = self.get_web_context(f"{topic} {lesson_title}") if self.use_web else {"context": "", "sources": []}
        web_context = web_data["context"]
        sources = web_data["sources"]

        context_section = ""
        if web_context:
            context_section = "\n\nHere is live web-based context to incorporate:\n" + web_context + "\n"
        
        user_prompt = (
            "Generate JSON content for one interactive and engaging lesson.\n"
            f"Topic: {topic}\n"
            f"Module: {module_name}\n"
            f"Lesson: {lesson_title}\n"
            f"{context_section}\n"
            "Follow this schema strictly. Return valid JSON only:\n"
            "{\n"
            '  "title": "' + lesson_title + '",\n'
            '  "path": "<slug>",\n'
            '  "lesson": [\n'
            '    { "type": "heading", "content": "..." },\n'
            '    { "type": "paragraph", "content": "..." },\n'
            '    { "type": "formula", "content": "..." },\n'
            '    { "type": "code", "language": "python", "content": "..." },\n'
            '    { "type": "image", "src": "..." },\n'
            '    { "type": "video", "content": "..." },\n'
            '    { "type": "task", "content": "..." },\n'
            '    { "type": "quiz", "content": "..." }\n'
            "  ],\n"
            '  "notes": [{ "title": "Quick Tips", "items": ["...", "..."] }],\n'
            '  "quickquiz": [{ "question": "...", "options": ["..."], "correctAnswer": "..." }],\n'
            '  "projectideas": [{ "type": "paragraph", "content": "Mini project related to this lesson." }],\n'
            f'  "sources": {json.dumps(sources, indent=2)}\n'
            "}\n\n"
            "Each lesson must include short, visual explanations, practice ideas, and mention cited sources where applicable.\n"
            "Make it look like an interactive learning page:\n"
            "- Use short sentences and examples.\n"
            "- Add 'Try it yourself' coding challenges.\n"
            "- Mention visual or animation ideas.\n"
            "- End with a quiz or fun recap points.\n"
        )

        response = self.ai_client.get_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=2500
        )

        return json.loads(response.choices[0].message.content)

    def build_topic(self, topic: str) -> Dict:
        """Build course structure for a given topic using dynamically generated modules."""
        logger.info(f"Building course for topic: {topic}")
        start_time = time.time()

        # Step 1: Generate modules
        modules = self.module_generator.generate_modules(topic)
        if not modules:
            logger.error(f"Failed to generate modules for topic: {topic}")
            return {}

        course_modules = {topic: {"menu": []}}

        # Step 2: For each module, dynamically generate lessons
        for module in modules:
            module_title = module["section"]
            logger.info(f"🧩 Generating lessons for {module_title}...")
            lessons = self.module_generator.generate_lessons(topic, module_title)

            if not lessons:
                logger.warning(f"No lessons generated for {module_title}. Using fallback.")
                lessons = [
                    f"Introduction to {module_title}",
                    f"Advanced {module_title} Concepts",
                    f"{module_title} Best Practices"
                ]

            module_dict = {"module": module_title, "section": module_title, "items": []}

            for lesson_title in lessons:
                try:
                    lesson_content = self.generate_lesson_content(topic, module_title, lesson_title)
                    module_dict["items"].append(lesson_content)
                    logger.info(f"Lesson '{lesson_title}' added to {module_title}")
                except Exception as e:
                    logger.error(f"Error generating lesson '{lesson_title}': {str(e)}")

            course_modules[topic]["menu"].append(module_dict)

        logger.info(f"✅ Course built for {topic} in {time.time() - start_time:.2f} sec")
        return course_modules

    def save_course_to_file(self, course_data: Dict, filename: str = None) -> str:
        """Save course data to JSON file."""
        if not course_data:
            logger.error("No course data to save")
            return ""

        courses_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'courses_json')
        os.makedirs(courses_dir, exist_ok=True)

        if not filename:
            topic = next(iter(course_data.keys()))
            filename = f"{topic.lower().replace(' ', '_')}_course.json"

        filepath = os.path.join(courses_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(course_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Course data saved to {filepath}")
        return filepath


def main():
    mode = input("Choose learning mode (hands-on / visual / theoretical): ").strip().lower()
    builder = CourseBuilder(use_azure=True, learning_mode=mode if mode in SYSTEM_PROMPTS else "hands-on")
    
    topic = input("Enter the topic for your course: ").strip()
    if not topic:
        print("Error: Topic cannot be empty")
        return
        
    course = builder.build_topic(topic)
    if course:
        filename = builder.save_course_to_file(course)
        print(f"✅ {filename} generated successfully in '{builder.learning_mode}' mode!")
    else:
        print("❌ Failed to generate course. Please check the logs for details.")


if __name__ == "__main__":
    main()
