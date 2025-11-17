from typing import Optional
from pydantic import BaseModel, Field


# Pydantic
class ResponseFormat(BaseModel):
    """Respond in a conversational manner."""
    answer: str = Field(description="The answer to the user's question")
    question1: str = Field(description="The first generated follow-up question based on the context.")
    question2: str = Field(description="The second generated follow-up question based on the context.")
    stage: str = Field(description="The current stage of the drift management pipeline.")
    operation: str = Field(description="The next operation in the current drift management pipeline.")
    explanation: str = Field(description="Explain why this operation is recommended.")



