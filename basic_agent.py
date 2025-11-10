from typing import Dict


class BasicAgent:
    """
    Minimal fallback agent that accepts either:
    - {"messages": [{"role": "user", "content": "..."}]}
    - {"input": "..."}
    and returns a string response using the provided llm.
    """

    def __init__(self, llm, system_text: str):
        self.llm = llm
        self.system_text = system_text

    def invoke(self, data: Dict):
        # Extract question from supported shapes
        question = None
        msgs = data.get("messages") if isinstance(data, dict) else None
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if isinstance(last, dict):
                question = last.get("content")
        if not question and isinstance(data, dict):
            question = data.get("input")
        if not isinstance(question, str):
            question = str(question) if question is not None else ""
        prompt = f"{self.system_text}\n\nQuestion: {question}\n\nAnswer:"
        try:
            result = self.llm.invoke(prompt)
            # ChatOpenAI may return a message object with .content
            if isinstance(result, str):
                return result
            content = getattr(result, "content", None)
            return content if isinstance(content, str) else str(result)
        except Exception as e2:
            return f"Agent fallback error: {e2}"


