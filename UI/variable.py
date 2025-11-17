from agent import DatasetAgent
import time
from UI.constants import DEFAULT_STAGE


class Variables(object):
    def __init__(self):
        self.df = None
        self.agent: DatasetAgent = None
        self.rag = None
        self.rag_prompt = None
        self.use_rag = False
        self.dialog = []
        self.file_name = None
        self.suggested_questions = None
        self.data_snapshots = []
        self.conversation_session = None
        self.current_stage = DEFAULT_STAGE
        self.secondary_df = None  # secondary dataset
        self.target_attribute = None  # target attribute
        self.column_types = None  # column types



global_vars = Variables()
