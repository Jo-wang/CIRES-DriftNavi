from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages.base import messages_to_dict
import os
from enum import Enum
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, HumanMessagePromptTemplate, \
    SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import ConfigurableField
from agent.utils import create_pandas_dataframe_agent
from agent.explain_api import generate_response_from_prompt
from db_models.system_log import SystemLogMessage, AssistantLogMessage
from langchain_core.messages import HumanMessage
import re
from langchain.output_parsers import PydanticOutputParser
from db_models.conversation import Conversation
from flask_login import current_user
from agent.parser import ResponseFormat
from UI.constants import PIPELINE_STAGES, DEFAULT_STAGE
import base64
from mimetypes import guess_type
import pandas as pd


class ConversationFormat(str, Enum):
    FULL_JSON = 'Full JSON'
    SIMPLIFIED_JSON = 'Simplified JSON'
    TEXT = 'Text'


class PersistenceType(str, Enum):
    DATABASE = 'database'
    FILE = 'file'


# Module-level LLM configuration without configurable_alternatives
def create_llm_for_model(model_name: str = "gpt-4o-mini"):
    """
    Create a ChatOpenAI instance for the specified model.
    
    Args:
        model_name (str): The model name ('gpt-4o' or 'gpt-4o-mini')
        
    Returns:
        ChatOpenAI: A configured LLM instance
    """
    try:
        if model_name == "gpt-4o":
            return ChatOpenAI(temperature=0.7, model="gpt-4o")
        else:
            return ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    except Exception as e:
        print(f"[AGENT] Error creating LLM for {model_name}: {str(e)}")
        # Fallback to gpt-4o-mini
        return ChatOpenAI(temperature=0.7, model="gpt-4o-mini")


def get_configured_llm():
    """
    Get a default LLM instance.
    Maintained for backward compatibility.
    """
    return create_llm_for_model("gpt-4o-mini")


def reset_llm_configuration():
    """
    Reset function maintained for backward compatibility.
    """
    print("[AGENT] LLM configuration reset (no-op in new implementation)")


def create_safe_llm_for_agent():
    """
    Create a safe LLM instance for DatasetAgent.
    """
    return create_llm_for_model("gpt-4o-mini")


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    # Default to png
    if mime_type is None:
        mime_type = 'image/png'

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


# def pass_argument_next_questions_class(question1, question2):
#     class NextQuestionFormat(BaseModel):
#         response: str = Field(description="Answer to the user's query")
#         suggestion1: str = Field(description=question1)
#         suggestion2: str = Field(description=question2)
#     return NextQuestionFormat

class DatasetAgent:

    def __init__(self, df, conversation_session=None, llm=None, file_name=None, user_id=None):
        self.user_id = user_id
        if llm is None:
            llm = create_llm_for_model("gpt-4o-mini")
        self.llm = llm
        self.model_name = llm.model_name
        self.session_id = conversation_session
        self.elem_queue = []
        self.execution_error: list[Exception] = []
        self.list_commands: list[str] = []
        self.file_name = file_name
        self.current_stage = DEFAULT_STAGE
        self.secondary_df = None  # Initialize secondary dataset as None
        self.secondary_file_name = None  # Initialize secondary file name as None
        self.df = df  # Store the dataframe directly for easier access

        self.parser = PydanticOutputParser(pydantic_object=ResponseFormat)
        multimodal_prompt = self.configure_multimodal_prompt()
        self.prompt = self.configure_chat_prompt()

        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            elem_queue=self.elem_queue,
            execution_error=self.execution_error,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"handle_parsing_errors": True},
            list_commands=self.list_commands,
            prefix=current_user.prefix_prompt,
        )

        self.chain = self.prompt | self.agent
        self.multimodal_chain = multimodal_prompt | self.llm

        self.chat_history = ChatMessageHistory(session_id=self.session_id)

        self.agent_with_chat_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.chat_history,
            history_messages_key="chat_history",
        )

        self.multimodal_chain_with_history = RunnableWithMessageHistory(
            self.multimodal_chain,
            lambda session_id: self.chat_history,
            history_messages_key="chat_history",
        )

        self.agent_with_trimmed_history = (
                RunnablePassthrough.assign(messages_trimmed=self.trim_messages)
                | self.agent_with_chat_history
        )

    def update_agent_prompt(self):
        # invoked when the user changed their prompts or user profile.
        prompt = self.configure_chat_prompt()
        self.chain.first = prompt
        self.chat_history = self.agent_with_chat_history.get_session_history(self.session_id)
        self.agent_with_chat_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.chat_history,
            history_messages_key="chat_history",
        )
        self.agent_with_trimmed_history = (
                RunnablePassthrough.assign(messages_trimmed=self.trim_messages)
                | self.agent_with_chat_history
        )

    def configure_multimodal_prompt(self):
        multimodal_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a data scientist. Describe the image provided in detail and find some insights about distributional drift. "
                 "If there are potential distributional drift between the two datasets, tell the user the ways to mitigate these distributional drift."),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": "{text}",
                        },
                        {
                            "type": "image_url",
                            "image_url": "{encoded_image_url}",
                        },
                    ],
                ),
            ]
        )
        return multimodal_prompt

    def configure_user_msg_prompt(self):
        user_prompt = PromptTemplate(
            template=""""The current stage of distributional drift management is {stage}. Please answer the my question: {input}. 

{dataset_info}

The response should be tailored to my background: {background} and align with the {stage}. Ensure that your 
answer is informative while understandable for me. To make your answer clearer and instructive, 
you can include examples and step-by-step instructions appropriate for my background.

IMPORTANT: When providing code examples, always use 'df1' for the primary dataset and 'df2' for the secondary dataset.
The Python sandbox already has the following libraries pre-loaded:
- pandas (as pd)
- numpy (as np)
- matplotlib.pyplot (as plt)
- seaborn (as sns)
- scipy.stats (as stats)
""",
            input_variables=["input", "stage"],
            partial_variables={
                "background": current_user.persona_prompt,
                "dataset_info": lambda: self.get_both_datasets_info() if hasattr(self, 'secondary_df') and self.secondary_df is not None else self.get_dataset_info()
            },
        )
        user_message_prompt = HumanMessagePromptTemplate(prompt=user_prompt)
        return user_message_prompt

    def configure_system_msg_prompt(self):
        pipeline_prompt = """
            The         
        """
        system_prompt = PromptTemplate(
            template="""          

                    {custom_system_prompt}
                    {format_instructions}
                    {question_prompt}          

                    """,
            partial_variables={"format_instructions": self.parser.get_format_instructions(),
                               "question_prompt": current_user.follow_up_questions_prompt_1,
                               "custom_system_prompt": current_user.system_prompt}
        )
        # system_prompt = PromptTemplate(
        #     template="""
        #     {custom_system_prompt}
        #     """,
        #     partial_variables={"custom_system_prompt": current_user.system_prompt}
        # )
        system_message_prompt = SystemMessagePromptTemplate(prompt=system_prompt)
        return system_message_prompt

    def configure_chat_prompt(self):
        system_message_prompt = self.configure_system_msg_prompt()
        user_message_prompt = self.configure_user_msg_prompt()
        prompt = ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                MessagesPlaceholder(variable_name="chat_history"),
                user_message_prompt
            ]
        )
        return prompt

    def trim_messages(self, trimmed_message):
        # store the most recent 30 messages
        stored_messages = self.chat_history.messages
        if len(stored_messages) <= 30:
            return False
        self.chat_history.clear()
        for message in stored_messages[-30:]:
            self.chat_history.add_message(message)
        return True

    def describe_image(self, query, image_data):
        # image_url = "./UI/assets/cat.jpg"
        # image_data = local_image_to_data_url(image_url)
        result = self.multimodal_chain_with_history.with_config(
            configurable={"session_id": self.session_id}).invoke({"text": query, "encoded_image_url": image_data})
        return result

    def run(self, text, stage):
        self.elem_queue.clear()
        self.execution_error.clear()
        self.list_commands.clear()

        # Direct agent invocation without configurable alternatives
        result = self.agent_with_trimmed_history.with_config(
            configurable={"session_id": self.session_id}).invoke({"input": text, "stage": stage})

        # Parse response 
        suggestions = []
        stage = ""
        try:
            result = self.parser.parse(result['output'])
            suggestions.append(result.question1)
            suggestions.append(result.question2)
            stage = result.stage
            operation = result.operation
            explanation = result.explanation
            result = result.answer
            if stage is not self.current_stage and stage in PIPELINE_STAGES:
                self.current_stage = stage
            else:
                stage = self.current_stage

        except Exception as e:
            # cannot be parsed in the above format, directly return the answer
            self.execution_error.append(e)
            result = result['output']
            return result, self.elem_queue, suggestions, stage, None, None

        # Improve table removal logic
        table_pattern = r'(?s)\|.*?\|\n\|[-:]+\|\n(.*?)\n\n'
        result = re.sub(table_pattern, '', result)

        # Remove any remaining table-like structures
        result = re.sub(r'(?m)^\s*\|.*\|$', '', result)
        # if len(self.list_commands) > 0:
        #     self.persist_commands(json.dumps({"query": self.list_commands[0]}))
        if len(self.execution_error) > 0:
            result = f"""There was an error processing your request. Please provide a clearer query and try again.
                    (Error message: {str(self.execution_error[0])})"""
        if len(self.elem_queue) > 0:
            return result, self.elem_queue, suggestions, stage, None, None
        else:
            return result, None, suggestions, stage, None, None

    def set_llm_model(self, model):
        """
        Set the LLM model for this agent instance.
        
        Args:
            model (str): The model name to use ('gpt-4o' or 'gpt-4o-mini')
        """
        print(f"[AGENT] Switching model from {self.model_name} to {model}")
        
        try:
            # Create a new LLM instance with the selected model
            self.llm = create_llm_for_model(model)
            self.model_name = model
            
            # Update the agent chain with the new LLM
            self.agent = create_pandas_dataframe_agent(
                self.llm,
                self.df,
                verbose=True,
                elem_queue=self.elem_queue,
                execution_error=self.execution_error,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                agent_executor_kwargs={"handle_parsing_errors": True},
                list_commands=self.list_commands,
                prefix=current_user.prefix_prompt,
            )
            
            # Recreate the chain with the new agent
            self.chain = self.prompt | self.agent
            
            # Recreate the agent with chat history
            self.agent_with_chat_history = RunnableWithMessageHistory(
                self.chain,
                lambda session_id: self.chat_history,
                history_messages_key="chat_history",
            )
            
            # Recreate the agent with trimmed history
            self.agent_with_trimmed_history = (
                RunnablePassthrough.assign(messages_trimmed=self.trim_messages)
                | self.agent_with_chat_history
            )
            
            print(f"[AGENT] Model successfully switched to {model}")
            
        except Exception as e:
            print(f"[AGENT] Error switching model: {str(e)}")
            print("[AGENT] Model name updated but LLM instance may not have changed")

    def _get_full_history(self) -> dict:
        return {
            "dataset": self.file_name if self.file_name is not None else 'unknown',
            "messages": messages_to_dict(self.chat_history.messages)
        }

    def _get_agent_type(self, role) -> str:
        if role == 'human':
            return 'user'
        elif role == 'ai':
            return 'assistant'
        elif role == 'system-log':
            return 'system'
        elif role == 'assistant-command':
            return 'assistant-command'
        return role

    def _get_simplified_history(self) -> dict:
        raw_history = self._get_full_history()
        history = []
        for message in raw_history['messages']:
            history.append({
                "role": self._get_agent_type(message['type']),
                "content": message['data']['content']
            })
        return {
            "dataset": raw_history['dataset'],
            "messages": history
        }

    def _get_text_history(self) -> str:
        raw_history = self._get_full_history()
        history = "DATASET: " + raw_history['dataset'] + '\n'
        history += "=" * 100 + '\n'
        for message in raw_history['messages']:
            history += self._get_agent_type(message['type']).upper(
            ) + ": " + message['data']['content'] + '\n'
        return history

    def get_history(self, c_format):
        if not isinstance(c_format, ConversationFormat):
            raise ValueError(
                f'Only {[v.value for v in ConversationFormat]} are allowed.')
        history = None
        extension = None
        if c_format == ConversationFormat.FULL_JSON:
            history = json.dumps(self._get_full_history())
            extension = '.json'
        elif c_format == ConversationFormat.SIMPLIFIED_JSON:
            history = json.dumps(self._get_simplified_history())
            extension = '.json'
        elif c_format == ConversationFormat.TEXT:
            history = self._get_text_history()
            extension = '.txt'
        if history is None:
            raise NotImplementedError("Unsupported format")
        return history, extension

    def persist_history(self, user_id,
                        persistence_type: PersistenceType = PersistenceType.DATABASE,
                        c_format: ConversationFormat = ConversationFormat.SIMPLIFIED_JSON,
                        path: str = 'histories'):
        if not os.path.exists(path):
            os.makedirs(path)
        if persistence_type == PersistenceType.DATABASE and c_format == ConversationFormat.TEXT:
            raise TypeError(
                "Only JSON-like conversations can be written to the database")
        history, extension = self.get_history(c_format=c_format)
        if persistence_type == PersistenceType.FILE:
            with open(os.path.join(path, str(self.session_id) + extension), 'w') as f:
                f.write(history)
        elif persistence_type == PersistenceType.DATABASE:
            Conversation.upsert(user_id, str(
                self.session_id), self.file_name, self.model_name, json.loads(history)['messages'])
        else:
            raise NotImplementedError("Unsupported persistence type")

    def system_log(self,
                   message,
                   persistence_type: PersistenceType = PersistenceType.DATABASE,
                   c_format: ConversationFormat = ConversationFormat.SIMPLIFIED_JSON,
                   path: str = 'histories'):
        self.chat_history.add_message(SystemLogMessage(content=message))
        self.persist_history(persistence_type=persistence_type,
                             c_format=c_format, path=path)

    def add_user_action_to_history(self, message):
        self.chat_history.add_message(HumanMessage(content=message))

    def add_system_message(self, message):
        """
        Add a system message to the chat history that provides context for the model.
        
        Args:
            message (str): The system message to add as context
        """
        self.chat_history.add_message(SystemMessagePromptTemplate.from_template(message).format())

    def persist_commands(self,
                         message,
                         persistent_type: PersistenceType = PersistenceType.DATABASE,
                         c_format: ConversationFormat = ConversationFormat.SIMPLIFIED_JSON,
                         path: str = 'histories'):
        self.chat_history.add_message(AssistantLogMessage(content=message))

        if not os.path.exists(path):
            os.makedirs(path)
        if persistent_type == PersistenceType.DATABASE and c_format == ConversationFormat.TEXT:
            raise TypeError(
                "Only JSON-like conversations can be written to the database")

        history = json.dumps(self._get_simplified_history())
        extension = '.json'

        if persistent_type == PersistenceType.FILE:
            with open(os.path.join(path, str(self.session_id) + extension), 'w') as f:
                f.write(history)
        elif persistent_type == PersistenceType.DATABASE:
            Conversation.upsert(str(current_user.id), str(
                self.session_id), self.file_name, self.model_name, json.loads(history)['messages'])
        else:
            raise NotImplementedError("Unsupported persistence type")

    def add_secondary_dataset(self, secondary_df, secondary_file_name):
        """
        Add a secondary dataset to the agent for comparison with the primary dataset.
        
        Args:
            secondary_df (pandas.DataFrame): The secondary dataset
            secondary_file_name (str): The filename of the secondary dataset
        """
        self.secondary_df = secondary_df
        self.secondary_file_name = secondary_file_name
        self.add_user_action_to_history(f"I have uploaded a secondary dataset: {secondary_file_name}")
        return True

    def compare_datasets(self):
        """
        Compare the primary and secondary datasets and return a summary of differences.
        
        Returns:
            str: A text summary of the differences between datasets
        """
        if self.secondary_df is None:
            return "Secondary dataset not found. Please upload a secondary dataset first."
        
        # Use df directly (it's passed to __init__ and stored in the agent)
        df = self.df
        primary_cols = set(df.columns)
        secondary_cols = set(self.secondary_df.columns)
        
        # Analyze columns
        common_cols = primary_cols.intersection(secondary_cols)
        primary_only_cols = primary_cols - secondary_cols
        secondary_only_cols = secondary_cols - primary_cols
        
        # Compare sizes
        primary_rows = len(df)
        secondary_rows = len(self.secondary_df)
        
        # Generate summary
        summary = []
        summary.append(f"## Dataset Comparison: {self.file_name} vs {self.secondary_file_name}")
        summary.append(f"\n### Size Comparison")
        summary.append(f"* Primary dataset: {primary_rows} rows, {len(primary_cols)} columns")
        summary.append(f"* Secondary dataset: {secondary_rows} rows, {len(secondary_cols)} columns")
        
        summary.append(f"\n### Column Comparison")
        summary.append(f"* Common columns: {len(common_cols)}")
        if common_cols:
            summary.append(f"  * {', '.join(sorted(common_cols))}")
        
        if primary_only_cols:
            summary.append(f"* Columns only in primary dataset: {len(primary_only_cols)}")
            summary.append(f"  * {', '.join(sorted(primary_only_cols))}")
        
        if secondary_only_cols:
            summary.append(f"* Columns only in secondary dataset: {len(secondary_only_cols)}")
            summary.append(f"  * {', '.join(sorted(secondary_only_cols))}")
        
        # Compare data types for common columns
        if common_cols:
            summary.append(f"\n### Data Types for Common Columns")
            type_differences = []
            for col in common_cols:
                primary_type = str(df[col].dtype)
                secondary_type = str(self.secondary_df[col].dtype)
                if primary_type != secondary_type:
                    type_differences.append(f"* {col}: Primary ({primary_type}) vs Secondary ({secondary_type})")
            
            if type_differences:
                summary.append("Columns with different data types:")
                summary.extend(type_differences)
            else:
                summary.append("All common columns have the same data types.")
        
        # Log this action
        self.add_user_action_to_history("I compared the primary and secondary datasets")
        
        return "\n".join(summary)
    
    def get_dataset_info(self, is_primary=True):
        """Return a string containing information about the dataset structure and summary statistics.
        
        Args:
            is_primary (bool): If True, return info about the primary dataset, otherwise the secondary dataset
        
        Returns:
            str: A formatted string with dataset information
        """
        try:
            # Choose the appropriate dataset
            dataset = self.df if is_primary else self.secondary_df
            dataset_name = self.file_name if is_primary else self.secondary_file_name
            variable_name = "df1" if is_primary else "df2"
            
            if dataset is None:
                return f"{'Primary' if is_primary else 'Secondary'} dataset is not available."
            
            # Get basic dataset information
            shape = dataset.shape
            dtypes = dataset.dtypes
            missing_values = dataset.isnull().sum()
            
            # Get summary statistics for numeric columns
            numeric_columns = dataset.select_dtypes(include=['number']).columns
            numeric_summary = dataset[numeric_columns].describe() if len(numeric_columns) > 0 else pd.DataFrame()
            
            # Get summary statistics for categorical columns
            categorical_columns = dataset.select_dtypes(exclude=['number']).columns
            categorical_summary = {}
            for col in categorical_columns:
                if dataset[col].nunique() < 10:  # Only include if small number of unique values
                    categorical_summary[col] = dataset[col].value_counts().to_dict()
            
            # Format the information
            info = [f"{'Primary' if is_primary else 'Secondary'} Dataset: {dataset_name} (Available as variable '{variable_name}' in the Python sandbox)"]
            info.append(f"Shape: {shape[0]} rows, {shape[1]} columns")
            
            info.append("\nColumn Data Types:")
            for col, dtype in dtypes.items():
                info.append(f"- {col}: {dtype}")
            
            if len(numeric_columns) > 0:
                info.append("\nNumeric Column Summary Statistics:")
                info.append(str(numeric_summary.round(2)))
            
            if categorical_summary:
                info.append("\nCategorical Column Frequencies:")
                for col, counts in categorical_summary.items():
                    info.append(f"- {col}: {counts}")
            
            info.append("\nMissing Values:")
            for col, count in missing_values.items():
                if count > 0:
                    percentage = (count / shape[0]) * 100
                    info.append(f"- {col}: {count} ({percentage:.2f}%)")
            
            return "\n".join(info)
        except Exception as e:
            return f"Error retrieving dataset information: {str(e)}"

    def get_both_datasets_info(self):
        """Return information about both primary and secondary datasets
        
        Returns:
            str: A formatted string with information about both datasets
        """
        primary_info = self.get_dataset_info(is_primary=True)
        secondary_info = self.get_dataset_info(is_primary=False)
        
        # Add sandbox usage information
        sandbox_info = "\n\nPython Sandbox Usage Information:\n"
        sandbox_info += "- Primary dataset is available as 'df1' in the Python sandbox\n"
        sandbox_info += "- 'df' is also available as an alias for 'df1' for backward compatibility\n"
        
        if hasattr(self, 'secondary_df') and self.secondary_df is not None:
            sandbox_info += "- Secondary dataset is available as 'df2' in the Python sandbox\n"
        else:
            sandbox_info += "- Secondary dataset (df2) is not currently loaded\n"
        
        sandbox_info += "- Any code examples using the datasets should use these variable names (df1, df2)\n"
        
        comparison = ""
        if hasattr(self, 'secondary_df') and self.secondary_df is not None:
            # Add comparison information if both datasets exist
            comparison = "\n\nDataset Comparison:\n" + self.compare_datasets()
        
        return f"{primary_info}\n\n{secondary_info}{sandbox_info}{comparison}"
