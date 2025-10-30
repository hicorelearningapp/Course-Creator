import json
from typing import List, Dict
from openai_client import AIClient

class ModuleGenerator:
    """Handles generation of modules and lessons for a topic."""

    def __init__(self, ai_client: AIClient):
        self.ai_client = ai_client

    def generate_modules(self, topic: str) -> List[Dict]:
        """Get list of modules for a topic."""
        system_prompt = """
You are an AI, who is an expert instructional designer who creates visually engaging, hands-on, and practical course modules
similar to W3Schools and TutorialsPoint. You are an expert trainer who would be helping students coming to you excel in the domain and skill that they are choosing.

Each module should:
- Follow a clear learning progression from basics to mastery.
- Encourage learners to practice actively through examples, visuals, and small challenges.
- Include real-world relevance and coding demonstration opportunities.
- Be named in a way that excites curiosity (e.g., “Mastering Loops with Real Projects” instead of “Loops”).
- Avoid overly academic phrasing — keep it simple, approachable, and interactive.
"""

        user_prompt = f"""
        Create a JSON array of 4–6 modules for the topic "{topic}".
        Each module should represent a clear learning phase and have:
        - "module": sequential Module name (Module 1, Module 2, ...)
        - "section": a short, catchy title describing the focus of that module.

        Example:
        {{
          "modules": [
            {{"module": "Module 1", "section": "Introduction & Basics"}},
            {{"module": "Module 2", "section": "Practical Coding"}},
            {{"module": "Module 3", "section": "Advanced Concepts"}},.....
          ]
        }}
        Output ONLY valid JSON.
        """

        response = self.ai_client.get_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.6,
            max_tokens=800
        )

        output = response.choices[0].message.content
        try:
            return json.loads(output)["modules"]
        except Exception:
            print("⚠️ Invalid JSON for modules, raw output:")
            print(output)
            return []

    def generate_lessons(self, topic: str, module_title: str) -> List[str]:
        """Generate engaging, hands-on lesson titles for a given module."""
        system_prompt = """
You are an AI course creator designing short, engaging, hands-on lessons. 
You are an expert trainer who would be helping students coming to you excel in the domain and skill that they are choosing

Each lesson should:
- Be concise, unique, and relevant to the module theme.
- Encourage active learning through examples, visuals, and small challenges.
- Include real-world relevance and coding demonstration opportunities.
- Be named in a way that excites curiosity (e.g., “Build Your First Function” instead of “Introduction to Functions”).
- Avoid overly academic phrasing — keep it simple, approachable, and interactive.
"""

        user_prompt = f"""
        For the module "{module_title}" in the course "{topic}", 
        generate 3–5 interactive lesson titles.

        Each lesson should be:
        - Action-oriented (e.g., “Build Your First Function” instead of “Introduction to Functions”)
        - Clear, practical, and fun to try.
        - Beginner-friendly but informative.

        Example JSON:
        {{
          "lessons": [
            "Getting Started with .....",
            "Writing and Running Your First Script",
            "Using Loops to Automate Tasks",
            ......
          ]
        }}
        Output ONLY valid JSON.
        """

        response = self.ai_client.get_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=700
        )

        output = response.choices[0].message.content
        try:
            return json.loads(output)["lessons"]
        except Exception:
            print(f"⚠️ Invalid JSON for lessons in module '{module_title}', raw output:")
            print(output)
            return []
