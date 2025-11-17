import time
import dash
from UI.app import app
from dash.dependencies import Input, Output, State
import base64
from agent import ConversationFormat, rag
from agent.rag import RAG
import datetime
from dash import callback_context, MATCH, ALL
from dash.exceptions import PreventUpdate
import io
from dash import dcc, html, dash_table
from UI.functions import *
from UI.constants import PIPELINE_STAGES
from UI.components.unified_column_type_modal import (
    create_column_type_datatable,
    create_modal_status_component,
    create_modal_help_component
)
import dash_bootstrap_components as dbc
from flask_login import current_user
from UI.functions import query_llm
import json
@app.callback(
    Output({"type": "chat-type-compare-collapse", "index": ALL}, 'is_open', allow_duplicate=True),
    Input({"type": "chat-type-compare-toggle", "index": ALL}, 'n_clicks'),
    State({"type": "chat-type-compare-collapse", "index": ALL}, 'is_open'),
    prevent_initial_call=True
)
def toggle_chat_type_compare_collapse(n_clicks_list, is_open_list):
    if not n_clicks_list or not is_open_list:
        raise PreventUpdate
    # toggle only the one that changed (last in the list typical)
    toggled = []
    for n, open_state in zip(n_clicks_list, is_open_list):
        if n and n > 0:
            toggled.append(not open_state)
        else:
            toggled.append(open_state)
    return toggled
# Import the dash_to_text function for safely converting Dash components to text
from UI.utils import dash_to_text

# Helper functions for sequence-based chat ordering
# Global sequence counter to ensure chronological ordering
_message_sequence = 0

def create_timestamped_message(content, message_type="llm-msg"):
    """Create a chat message with sequence metadata for chronological ordering"""
    global _message_sequence
    _message_sequence += 1
    
    # Store sequence in component ID (reliably preserved by Dash)
    message_id = {
        "type": "chat-msg",
        "seq": _message_sequence,
        "timestamp": time.time()  # Keep for debugging/logging
    }
    
    print(f"🔢 [CREATE_MESSAGE] Created message with seq={_message_sequence}, timestamp={message_id['timestamp']:.3f}, type={message_type}")
    
    return html.Div(
        content,
        className=message_type,
        id=message_id
    )

def sort_chat_messages(query_records):
    """Sort chat messages by sequence number, maintaining original order for messages without sequence"""
    if not query_records:
        return []
    
    print(f"📋 [SORT_MESSAGES] Starting sort with {len(query_records)} messages")
    
    def get_sequence(item):
        """Extract sequence number from message ID, with robust handling of metadata"""
        try:
            # Check if item has an id attribute with sequence number
            if hasattr(item, 'id') and isinstance(item.id, dict):
                seq = item.id.get('seq')
                original_type = item.id.get('original_type', 'unknown')
                timestamp = item.id.get('timestamp')
                print(f" Debug: item.id={item.id}, seq={seq}, type={type(seq)}")
                print(f" seq is not None: {seq is not None}")
                print(f" seq == None: {seq == None}")
                print(f" bool(seq): {bool(seq)}")
                if seq is not None:
                    timestamp_str = f"{timestamp:.3f}" if timestamp is not None else "None"
                    print(f"  🔢 Found seq={seq}, type={original_type}, timestamp={timestamp_str}")
                    result = int(seq)
                    print(f"  Returning seq={result}")
                    return result
                else:
                    print(f"  seq is None! Full ID: {item.id}")
            
            # Check if it's a dict representation (when serialized)
            if isinstance(item, dict):
                props = item.get('props', {})
                item_id = props.get('id')
                if isinstance(item_id, dict):
                    seq = item_id.get('seq')
                    if seq is not None:
                        return int(seq)
            
            # Handle legacy messages that might have lost sequence info
            # Use timestamp as fallback ordering mechanism
            if hasattr(item, 'id') and isinstance(item.id, dict):
                timestamp = item.id.get('timestamp')
                if timestamp is not None:
                    # Convert timestamp to pseudo-sequence (negative to distinguish)
                    return -int(timestamp * 1000)  # Negative ensures timestamp-ordered items come before new ones
            
            if isinstance(item, dict):
                props = item.get('props', {})
                item_id = props.get('id')
                if isinstance(item_id, dict):
                    timestamp = item_id.get('timestamp')
                    if timestamp is not None:
                        return -int(timestamp * 1000)
                        
        except (ValueError, AttributeError, TypeError) as e:
            print(f"  ❌ Exception in get_sequence: {type(e).__name__}: {e}")
            pass
        print(f"  ⚠️ get_sequence returning 0 - no valid sequence found")
        return 0
    
    # Separate sequenced, timestamp-ordered, and truly non-sequenced messages
    sequenced = []  # Messages with positive sequence numbers (newest system)
    timestamp_ordered = []  # Messages with timestamp-based ordering (fallback)
    non_sequenced = []  # Messages with no ordering info
    
    for i, item in enumerate(query_records):
        sequence = get_sequence(item)
        if sequence > 0:
            sequenced.append((sequence, item))
            print(f"  ✅ Message {i} → sequenced (seq={sequence})")
        elif sequence < 0:
            # Timestamp-based ordering (negative sequence)
            timestamp_ordered.append((sequence, item))
            print(f"  ⏰ Message {i} → timestamp_ordered (ts={sequence})")
        else:
            non_sequenced.append(item)
            print(f"  ❌ Message {i} → non_sequenced (no seq/ts)")
            # Try to identify what this message is
            if hasattr(item, 'id'):
                print(f"     ID: {item.id}")
            elif hasattr(item, 'className'):
                print(f"     ClassName: {item.className}")
    
    # Sort sequenced messages by sequence number (ascending)
    sequenced.sort(key=lambda x: x[0])
    
    # Sort timestamp-ordered messages by timestamp (descending, since they're negative)
    timestamp_ordered.sort(key=lambda x: x[0], reverse=True)
    
    print(f"📊 [SORT_RESULTS] Categorized: sequenced={len(sequenced)}, timestamp_ordered={len(timestamp_ordered)}, non_sequenced={len(non_sequenced)}")
    
    if sequenced:
        print(f"  🔢 Sequenced order: {[seq for seq, _ in sequenced]}")
    if timestamp_ordered:
        print(f"  ⏰ Timestamp order: {[ts for ts, _ in timestamp_ordered]}")
    
    # Combine in chronological order:
    # 1. Timestamp-ordered messages (oldest first) 
    # 2. Sequenced messages (chronological order)
    # 3. Non-sequenced messages last (to preserve existing chat flow)
    result = ([item for _, item in timestamp_ordered] + 
             [item for _, item in sequenced] + 
             non_sequenced)
             
    print(f"🎯 [SORT_FINAL] Final order: timestamp_ordered → sequenced → non_sequenced (total: {len(result)})")
    return result

def handle_check_attribute_types_request(query_records, suggested_questions):
    """
    Handle the 'Check attribute types' request by generating a column type comparison table
    and inserting it into the chat chronologically.
    """
    try:
        from UI.components.unified_column_type_modal import prepare_column_type_data
        
        # Initialize query_records if None
        if query_records is None:
            query_records = []
        
        # First, preserve the original sequence number before removing duplicates
        def _get_node_id(n):
            if hasattr(n, 'id'):
                return n.id
            if isinstance(n, dict):
                props = n.get('props') or {}
                return props.get('id')
            return None

        def _is_type_compare_bubble(node):
            nid = _get_node_id(node)
            # Check both old format and new format with sequence
            return isinstance(nid, dict) and (
                nid.get('type') == 'chat-type-compare-bubble' or 
                nid.get('original_type') == 'chat-type-compare-bubble'
            )

        # CRITICAL FIX: Preserve original sequence number to maintain chronological position
        preserved_sequence = None
        preserved_timestamp = None
        
        for child in query_records:
            if _is_type_compare_bubble(child):
                # Extract sequence and timestamp from existing Column Type Comparison
                if hasattr(child, 'id') and isinstance(child.id, dict):
                    preserved_sequence = child.id.get('seq')
                    preserved_timestamp = child.id.get('timestamp')
                elif isinstance(child, dict):
                    props = child.get('props', {})
                    item_id = props.get('id')
                    if isinstance(item_id, dict):
                        preserved_sequence = item_id.get('seq')
                        preserved_timestamp = item_id.get('timestamp')
                break
        
        # Remove existing Column Type Comparison bubbles
        query_records = [child for child in query_records if not _is_type_compare_bubble(child)]
        
        # Add user message indicating the request with timestamp
        # user_message = create_timestamped_message("Check attribute types", "user-msg")
        # query_records.append(user_message)
        
        # Get column type data
        column_data = prepare_column_type_data()
        if not column_data:
            # Add error message as system response with timestamp
            error_message = create_timestamped_message([
                dcc.Markdown("⚠️ No column data available. Please ensure datasets are loaded.")
            ], "llm-msg")
            query_records.append(error_message)
        else:
            # Build read-only DataTable for chat embedding
            table_id = f"chat-column-type-table-{int(time.time())}"
            columns = [
                {'name': 'Column', 'id': 'column'},
                {'name': 'Classification', 'id': 'classification'},
                {'name': 'Data Type', 'id': 'data_type'},
                {'name': 'Primary', 'id': 'primary_exists'},
                {'name': 'Secondary', 'id': 'secondary_exists'},
                {'name': 'Primary Unique', 'id': 'primary_unique'},
                {'name': 'Secondary Unique', 'id': 'secondary_unique'},
                {'name': 'Sample Values', 'id': 'sample_values'},
            ]

            table = dash_table.DataTable(
                id=table_id,
                columns=columns,
                data=column_data,
                editable=False,
                row_deletable=False,
                style_table={'overflowX': 'auto', 'minWidth': '100%'},
                style_header={
                    'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'
                },
                style_cell={
                    'textAlign': 'left', 'padding': '8px', 'border': '1px solid #dee2e6',
                    'fontFamily': 'Arial, sans-serif', 'fontSize': '14px'
                },
                page_size=15,
                page_action='native',
                sort_action='native',
                sort_mode='multi',
                tooltip_data=[
                    {
                        'column': {
                            'value': f"Column: {row['column']}\nType: {row['classification']}\nData Type: {row['data_type']}",
                            'type': 'markdown'
                        }
                    } for row in column_data
                ],
                tooltip_duration=None,
            )

            # Build collapsible card inside chat bubble (read-only)
            header = dbc.CardHeader(
                html.Div([
                    html.Div(
                        html.Span("Column Type Comparison", className="mb-0"),
                        style={"display": "inline-block", "width": "80%", "fontWeight": "bold"}
                    ),
                    html.Div(
                        html.Button(
                            html.I(className="fa fa-chevron-up"),
                            id={"type": "chat-type-compare-toggle", "index": table_id},
                            n_clicks=0,
                            style={"border": "none", "background": "transparent", "cursor": "pointer"}
                        ),
                        style={"display": "inline-block", "width": "20%", "textAlign": "right"}
                    )
                ], style={"display": "flex", "alignItems": "center"}, className="d-flex justify-content-between")
            )

            collapse = dbc.Collapse(
                dbc.CardBody(table),
                id={"type": "chat-type-compare-collapse", "index": table_id},
                is_open=True
            )

            bubble_content = dbc.Card([header, collapse], className="mb-2")
            
            # CRITICAL FIX: Use preserved sequence if available, otherwise create new sequence
            if preserved_sequence is not None and preserved_timestamp is not None:
                # Recreate bubble with preserved sequence to maintain chronological position
                bubble = html.Div(
                    bubble_content,
                    className="llm-msg chat-table-container",
                    id={
                        "type": "chat-msg",
                        "seq": preserved_sequence,  # Keep original sequence!
                        "timestamp": preserved_timestamp,  # Keep original timestamp!
                        "original_type": "chat-type-compare-bubble"
                    }
                )
                print(f"[DEBUG SEQUENCE FIX] ✅ Recreated Column Type Comparison with preserved seq={preserved_sequence}")
            else:
                # First time creating - use normal sequence
                bubble = create_timestamped_message(
                    bubble_content,
                    "llm-msg chat-table-container"
                )
                # Merge sequence ID with type identification
                if hasattr(bubble, 'id') and isinstance(bubble.id, dict):
                    bubble.id.update({"original_type": "chat-type-compare-bubble"})
                print(f"[DEBUG SEQUENCE FIX] ✅ Created new Column Type Comparison with seq={bubble.id.get('seq')}")
            
            query_records.append(bubble)
        
        # Return updated chat with type comparison table chronologically inserted
        sorted_records = sort_chat_messages(query_records)
        return (
            sorted_records,       # query-area.children
            False,               # error-alert.is_open
            "",                  # error-alert.children 
            time.time(),         # chat-update-trigger.data
            suggested_questions, # next-suggested-questions.children
            "",                  # commands-input.value
            "",                  # query-input.value
            dash.no_update,      # pipeline-alert.is_open
            dash.no_update,      # pipeline-alert-text.children
            dash.no_update,      # pipeline-alert-icon.className
            dash.no_update       # pipeline-alert.color
        )
        
    except Exception as e:
        print(f"[CHAT] Error handling check attribute types request: {str(e)}")
        # Return error message in chat
        if query_records is None:
            query_records = []
        error_message = create_timestamped_message([
            dcc.Markdown(f"❌ Error loading column type comparison: {str(e)}")
        ], "llm-msg")
        query_records.append(error_message)
        sorted_records = sort_chat_messages(query_records)
        return (
            sorted_records,
            False,
            "",
            time.time(),
            suggested_questions,
            "",
            "",
            dash.no_update
        )


# Handle Check Attribute Types Button Click - Aligned with Detect/Explain Pattern
@app.callback(
    [Output("query-area", "children", allow_duplicate=True),
     Output("error-alert", "is_open", allow_duplicate=True),
     Output("error-alert", "children", allow_duplicate=True),
     Output("chat-update-trigger", "data", allow_duplicate=True),
     Output("next-suggested-questions", "children", allow_duplicate=True),
     Output("commands-input", "value", allow_duplicate=True),
     Output("query-input", "value", allow_duplicate=True),
     Output("pipeline-alert", "is_open", allow_duplicate=True)],
    [Input('show-type-compare-btn', 'n_clicks')],
    [State("query-area", "children"),
     State("next-suggested-questions", "children")],
    prevent_initial_call=True
)
def handle_check_attribute_types_button_click(n_clicks, query_records, suggested_questions):
    """Handle check attribute types button click in chat - aligned with detect/explain pattern"""
    if not n_clicks:
        raise PreventUpdate
    
    print(f"[CHAT BUTTON] Check attribute types button clicked (n_clicks={n_clicks})")
    
    # Call the original handler to get the full result
    full_result = handle_check_attribute_types_request(query_records, suggested_questions)
    
    # Return all 8 outputs to match detect/explain signature
    # This ensures the same processing pipeline and prevents sequence ordering issues
    return (
        full_result[0],   # query-area.children (updated with Column Type Comparison)
        full_result[1],   # error-alert.is_open 
        full_result[2],   # error-alert.children
        full_result[3],   # chat-update-trigger.data
        full_result[4],   # next-suggested-questions.children
        full_result[5],   # commands-input.value
        full_result[6],   # query-input.value
        dash.no_update    # pipeline-alert.is_open (preserve current state)
    )


@app.callback(
    [Output("query-area", "children"),
     Output("error-alert", "is_open", allow_duplicate=True),
    Output("error-alert", "children", allow_duplicate=True),
     Output("chat-update-trigger", "data"),
     Output("next-suggested-questions", "children"),
     Output("commands-input", "value"),
     Output('query-input', 'value'),
     Output("pipeline-alert", "is_open"),
     Output("pipeline-alert-text", "children"),
     Output("pipeline-alert-icon", "className"),
     Output("pipeline-alert", "color"),
     # Removed references to non-existent components for cleaner code
    ],
    [Input('send-button', 'n_clicks'),
     Input('query-input', 'n_submit'),
     Input({"type": 'next-suggested-question', "index": ALL}, 'n_clicks')],
    [State('query-input', 'value'),
     State('query-area', 'children'),
     State('next-suggested-questions', 'children'),
     State("chat-context-data", "data")],
    prevent_initial_input=True,
    prevent_initial_call=True
)


def update_messages(n_clicks, n_submit, question_clicked, input_text, query_records, suggested_questions, distribution_context):
    # Note: "Check attribute types" button is now handled by independent callback
    # to avoid conflicts with other chat operations
    
    # ✅ New: Smart trigger source detection to prevent automatic triggering GPT
    trigger_id = callback_context.triggered_id
    
    def should_process_gpt_request():
        """
        Smartly determine whether to call GPT only when the user actively interacts.
        Only call GPT when the user actively interacts to prevent automatic triggering.
        """
        # If there is no trigger source, do not process
        if not trigger_id:
            return False
            
        # Case 1: User clicks send button and has input content
        if trigger_id == 'send-button' and input_text and input_text.strip():
            return True
            
        # Case 2: User presses enter key and has input content  
        if trigger_id == 'query-input' and input_text and input_text.strip():
            return True
            
        # Case 3: User clicks suggested question button
        if (not isinstance(trigger_id, str) and 
            hasattr(trigger_id, 'type') and 
            'next-suggested-question' in trigger_id.type and
            question_clicked and any(question_clicked)):
            return True
            
        # Other cases (including indirect triggers through other callbacks), do not process
        return False
    
    # Use smart decision to determine whether to process GPT request
    if not should_process_gpt_request():
        # Do not process GPT request, return current state
        return (
            query_records,       # query-area.children (unchanged)
            False,               # error-alert.is_open
            "",                  # error-alert.children 
            dash.no_update,      # chat-update-trigger.data (no update)
            suggested_questions, # next-suggested-questions.children (keep existing)
            "",                  # commands-input.value
            "",                  # query-input.value
            False,               # pipeline-alert.is_open
            "Ready for user input",  # pipeline-alert-text.children
            "fas fa-comment me-2",   # pipeline-alert-icon.className
            "secondary"              # pipeline-alert.color
        )
    
    # Existing dataset check logic
    if global_vars.df is None:
        # no dataset loaded
        sorted_records = sort_chat_messages(query_records)
        return sorted_records, True, "Please upload a dataset first.", dash.no_update, suggested_questions, "", "", dash.no_update

    if not isinstance(trigger_id, str) and 'next-suggested-question' in trigger_id.type:
        query = global_vars.suggested_questions[int(trigger_id.index[-1])]
    else:
        query = input_text
    
    # ✅ CRITICAL FIX: Handle None and empty query values
    if query is None:
        query = ""  # Default to empty string if no input provided
    
    # If query is empty and no suggested question was clicked, don't process
    if not query.strip() and not (not isinstance(trigger_id, str) and 'next-suggested-question' in str(trigger_id)):
        # Return current state without processing empty query
        return (
            query_records,       # query-area.children (unchanged)
            False,               # error-alert.is_open
            "",                  # error-alert.children 
            dash.no_update,      # chat-update-trigger.data (no update)
            suggested_questions, # next-suggested-questions.children
            "",                  # commands-input.value
            "",                  # query-input.value
            False,               # pipeline-alert.is_open
            "Workflow stage updated",  # pipeline-alert-text.children
            "fas fa-route me-2",       # pipeline-alert-icon.className
            "primary"                  # pipeline-alert.color
        )
        
    new_user_message = create_timestamped_message(query + '\n', "user-msg")
    global_vars.dialog.append("\nUSER: " + query + '\n')
    suggested_questions = []
    if not query_records:
        query_records = []
    if global_vars.rag and global_vars.use_rag:
        input_text = global_vars.rag.invoke(query)
        global_vars.rag_prompt = query

    # Check if there's any distribution comparison context to include
    distribution_contexts = []
    if distribution_context and isinstance(distribution_context, list) and len(distribution_context) > 0:
        distribution_contexts = distribution_context
    
    # If we have distribution contexts, add them to the agent's context
    if distribution_contexts and len(distribution_contexts) > 0 and hasattr(global_vars, 'agent') and global_vars.agent:
        # ==== DETAILED DEBUG INFO FOR CHAT AI CONTEXT ====
        print(f"\n{'='*70}")
        print(f"[CHAT AI CONTEXT DEBUG] SENDING CONTEXT ITEMS TO AI")
        print(f"{'='*70}")
        print(f"Total Context Items Available: {len(distribution_contexts)}")
        for idx, ctx in enumerate(distribution_contexts):
            print(f"  Context Item {idx+1}:")
            print(f"    - ID: {ctx.get('id', 'Unknown')}")
            print(f"    - Type: {ctx.get('type', 'Unknown')}")
            print(f"    - Timestamp: {ctx.get('timestamp', 'Unknown')}")
            if ctx.get('type') == 'target_distribution':
                print(f"    - Target Attribute: {ctx.get('target_attribute', 'Unknown')}")
            elif ctx.get('type') == 'conditional_distribution':
                print(f"    - Target Attribute: {ctx.get('target_attribute', 'Unknown')}")
                print(f"    - Target Value: {ctx.get('target_value', 'Unknown')}")
                print(f"    - Compare Attribute: {ctx.get('compare_attribute', 'Unknown')}")
            elif ctx.get('type') == 'distribution_comparison':
                print(f"    - Cell Info: {ctx.get('cell_info', 'Unknown')[:50]}...")
            elif ctx.get('type') == 'metric':
                print(f"    - Metric Type: {ctx.get('metric_type', 'Unknown')}")
                print(f"    - Attribute: {ctx.get('attribute_name', 'Unknown')}")
            elif ctx.get('type') == 'drift_analysis':
                print(f"    - Attribute: {ctx.get('attribute_name', 'Unknown')}")
                print(f"    - Metrics Count: {ctx.get('metrics_count', 'Unknown')}")
        print(f"User Query: {query}")
        print(f"{'='*70}\n")
        
        # Create a detailed system message containing all context information
        context_info = "## Data Analysis Context\n\n"
        
        # Add detailed information for each context type
        for idx, ctx in enumerate(distribution_contexts):
            ctx_type = ctx.get('type', 'unknown')
            
            # if ctx_type == 'distribution_comparison':
            # Add basic cell information
            if ctx.get("cell_info"):
                cell_info_text = dash_to_text(ctx.get('cell_info'))
                context_info += f"### Distribution Comparison {idx+1}: {cell_info_text}\n"
            
                # Add complete statistical summary
            if ctx.get("stored_summary"):
                summary_text = dash_to_text(ctx.get('stored_summary'))
                context_info += f"Statistical Summary:\n```\n{summary_text}\n```\n\n"
            elif ctx.get("summary_text"):
                summary_text = dash_to_text(ctx.get('summary_text'))
                context_info += f"Statistical Summary:\n```\n{summary_text}\n```\n\n"
            
            elif ctx_type == 'conditional_distribution':
                # Process conditional distribution context
                target_attr = ctx.get('target_attribute', 'unknown')
                target_val = ctx.get('target_value', 'unknown')
                compare_attr = ctx.get('compare_attribute', 'unknown')
                
                context_info += f"### Conditional Distribution Analysis {idx+1}\n"
                context_info += f"**Analysis Focus**: {target_attr} = {target_val}, grouped by {compare_attr}\n"
                
                if ctx.get("summary_text"):
                    raw_summary = ctx.get('summary_text')
                    print(f"[CONDITIONAL DEBUG] Raw summary_text type: {type(raw_summary)}")
                    print(f"[CONDITIONAL DEBUG] Raw summary_text content: {raw_summary[:300]}...")
                    
                    summary_text = dash_to_text(raw_summary)
                    print(f"[CONDITIONAL DEBUG] Processed summary_text: {summary_text[:300]}...")
                    
                    context_info += f"**Details**:\n```\n{summary_text}\n```\n"
                else:
                    print(f"[CONDITIONAL DEBUG] No summary_text found in context!")
                    print(f"[CONDITIONAL DEBUG] Available keys: {list(ctx.keys())}")
                
                # Add analysis instructions for this specific context
                context_info += f"**Analysis Instructions**: When discussing this conditional distribution:\n"
                context_info += f"- Focus on how '{compare_attr}' values differ when '{target_attr}' equals '{target_val}'\n"
                context_info += f"- Compare the distribution patterns between the two datasets\n"
                context_info += f"- Explain what the differences in '{compare_attr}' distribution might indicate\n"
                context_info += f"- Use the provided statistical data to support your analysis\n\n"
            
            elif ctx_type == 'target_distribution':
                # Process target distribution context
                target_attr = ctx.get('target_attribute', 'unknown')
                context_info += f"### Target Distribution Analysis {idx+1}\n"
                context_info += f"**Target Attribute**: {target_attr}\n"
                
                if ctx.get("summary_text"):
                    summary_text = dash_to_text(ctx.get('summary_text'))
                    context_info += f"**Details**:\n```\n{summary_text}\n```\n\n"
            
            elif ctx_type == 'drift_analysis':
                # Process drift analysis context
                attr_name = ctx.get('attribute_name', 'unknown')
                context_info += f"### Drift Analysis {idx+1}\n"
                context_info += f"**Attribute**: {attr_name}\n"
                
                if ctx.get("summary_text"):
                    summary_text = dash_to_text(ctx.get('summary_text'))
                    context_info += f"**Metrics**:\n```\n{summary_text}\n```\n\n"
            
            elif ctx_type == 'metric':
                # Process individual metric context
                metric_type = ctx.get('metric_type', 'unknown')
                attr_name = ctx.get('attribute_name', 'unknown')
                context_info += f"### Metric Analysis {idx+1}\n"
                context_info += f"**Metric**: {metric_type} for {attr_name}\n"
                
                if ctx.get("summary_text"):
                    summary_text = dash_to_text(ctx.get('summary_text'))
                    context_info += f"**Details**:\n```\n{summary_text}\n```\n\n"
        
        # Add guidance instructions
        context_info += "\n## Guidance Instructions\n"
        context_info += "1. Analyze the data distribution comparison information above\n"
        context_info += "2. When answering user questions, explicitly reference relevant distribution comparison data\n"
        context_info += "3. Provide data-based insights, including statistical differences and their potential implications\n"
        context_info += "4. If the question directly relates to distribution comparisons, analyze this data in detail\n"
        context_info += "5. Consider the potential impact of these data on the question even if the user does not directly ask about distribution differences\n"
        
        # Debug: Print the complete context_info being sent to AI
        print(f"\n[FINAL CONTEXT DEBUG] Complete context being sent to AI:")
        print(f"Context length: {len(context_info)} characters")
        print(f"Context preview: {context_info[:500]}...")
        print(f"Context ending: ...{context_info[-200:]}")
        
        # Add the context as a system message to the conversation history
        global_vars.agent.add_system_message(context_info)
        
        # Record user action in history
        global_vars.agent.add_user_action_to_history("User asked a question based on data distribution comparison context")
        
        # Use the original query
        enhanced_query = query
    else:
        # No context, use the original query
        enhanced_query = query

    # Query the LLM with either the enhanced query or original query
    answer, media, new_suggested_questions, stage, op, expl = query_llm(enhanced_query, global_vars.current_stage, current_user.id)

    print(stage)
    change_stage = False
    stages = PIPELINE_STAGES
    if stage is not None and stage in stages:
        if stage != global_vars.current_stage:
            global_vars.current_stage = stage
            change_stage = True
    if new_suggested_questions is not None:
        for i in range(len(new_suggested_questions)):
            if new_suggested_questions[i]:
                new_suggested_question = html.Div(dbc.CardBody([
                    html.P("Suggested Next Question", style={'fontWeight': 'bold', "marginBottom": "0px"}),
                    html.P(new_suggested_questions[i], style={"marginBottom": "0px"})],
                    style={"padding": 0}), className="next-suggested-question",
                    id={"type": "next-suggested-question", "index": f'next-question-{i}'}, n_clicks=0)
                suggested_questions.append(new_suggested_question)

    answer = format_reply_to_markdown(answer)
    
    response = answer + '\n'
    global_vars.dialog.append("\n" + response)
    
    # Parse code blocks and create a component with formatted blocks
    components = parse_code_blocks(response)
    response_children = []
    
    for i, component in enumerate(components):
        if isinstance(component, tuple):
            # This is a code block
            code, is_python = component
            code_id = f"code-block-{int(time.time())}-{i}"
            
            # Create code block with copy button
            code_block = html.Div([
                html.Div([
                    html.Button("Copy", 
                               id=f"{code_id}-btn",
                               className="copy-button",
                               **{'data-code-id': code_id})
                ], className="code-header"),
                html.Pre([
                    html.Code(code, id=code_id, className="language-python" if is_python else "")
                ])
            ], className="code-block-container")
            
            response_children.append(code_block)
        else:
            # This is regular text
            response_children.append(dcc.Markdown(component))
    
    # Create new message with all components
    new_response_message = create_timestamped_message(response_children, "llm-msg")
    query_records.append(new_user_message)
    query_records.append(new_response_message)
    list_commands = global_vars.agent.list_commands
    
    # Generate intelligent stage transition message if stage changed
    alert_text = "Workflow stage updated"
    alert_icon = "fas fa-route me-2"
    alert_color = "primary"
    
    if change_stage and stage:
        stage_messages = {
            "Detect": {
                "text": f"🔍 Switched to Detection Stage - Ready to analyze distribution shifts",
                "icon": "fas fa-search-plus me-2",
                "color": "info"
            },
            "Explain": {
                "text": f"📊 Switched to Explanation Stage - Analyzing root causes of detected shifts",
                "icon": "fas fa-chart-line me-2",
                "color": "warning"
            },
            "Adapt": {
                "text": f"🔧 Switched to Adaptation Stage - Generating actionable solutions",
                "icon": "fas fa-wrench me-2",
                "color": "success"
            }
        }
        
        if stage in stage_messages:
            alert_text = stage_messages[stage]["text"]
            alert_icon = stage_messages[stage]["icon"]
            alert_color = stage_messages[stage]["color"]
            print(f"[PIPELINE ALERT] Stage transition: {global_vars.current_stage} -> {stage}")
    
    # Return exactly 12 values to match the 12 outputs defined in the callback
    sorted_records = sort_chat_messages(query_records)
    return (
        sorted_records,       # query-area.children
        False,               # error-alert.is_open
        "",                  # error-alert.children 
        time.time(),         # chat-update-trigger.data
        suggested_questions, # next-suggested-questions.children
        ('\n').join(list_commands) if len(list_commands) > 0 else "", # commands-input.value
        "",                  # query-input.value
        change_stage,        # pipeline-alert.is_open
        alert_text,          # pipeline-alert-text.children
        alert_icon,          # pipeline-alert-icon.className
        alert_color          # pipeline-alert.color
    )


@app.callback(
    Output("export-conversation", "data"),
    Input("download-button", "n_clicks"),
    Input("export-format-dropdown", "value"),
    prevent_initial_call=True,
)
def export_conversation(n_clicks, format):
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if (triggered_id != 'download-button'):
        return None
    # return dict(content="".join(global_vars.dialog), filename=f"query-history-{formatted_date_time}.txt")
    if (global_vars.agent is None):
        return None
    history, extension = global_vars.agent.get_history(
        c_format=ConversationFormat(format))
    return dict(content=history, filename=f"query-history-{formatted_date_time}" + extension)




@app.callback(
    Output('rag-switch-output', 'children'),
    Output('rag-card', 'style'),
    Output('RAG-button', 'style'),
    Input('rag-switch', 'value')
)
def rag_switch(value):
    if value:
        global_vars.use_rag = True
        return 'RAG: On', {'display': 'block'}, {'display': 'block'}
    else:
        global_vars.use_rag = False
        return 'RAG: OFF', {'display': 'none'}, {'display': 'none'}


@app.callback(
    Output('RAG-area', 'children'),
    Input('upload-rag', 'contents'),
    State('upload-rag', 'filename'),
    Input('RAG-button', 'n_clicks'),
    Input('send-button', 'n_clicks'),
    [State('RAG-area', 'children')],
    State('query-input', 'value'),
    prevent_initial_call=True
)
def upload_rag_area(list_of_contents, list_of_names, clicks_rag, clicks_send, rag_output, query):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'upload-rag':
        if list_of_contents is not None:
            filename = ''
            output = 'RAG files: '
            # Assuming that only the first file is processed
            contents = list_of_contents[0]
            filename = list_of_names[0]

            # Decode the contents of the file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                # Assume the file is plain text
                if 'pdf' in filename:
                    # Assume that the user uploaded a CSV
                    output += filename + '\n'
                    if global_vars.rag:
                        global_vars.rag.clean()

                    global_vars.rag = RAG(io.BytesIO(decoded))

                    return [html.Div([
                        # placeholder
                        output
                    ])]
                else:
                    return html.Div([
                        "This file format is not supported. Only PDF files are supported."
                    ])

            except Exception as e:
                print(e)
                return html.Div([
                    'There was an error processing this file.'
                ])

    elif triggered_id == 'send-button':
        if not global_vars.rag or not global_vars.use_rag:
            return rag_output
        if not rag_output:
            return html.Div([""])
        if global_vars.rag and global_vars.use_rag:
            message = global_vars.rag.invoke(query)
            output_text, media, new_suggested_questions, stage, op, expl = query_llm(message, global_vars.current_stage, current_user.id)


        rag_output.append(html.Div(["RAG enhanced prompt: " + output_text]))
        global_vars.rag_prompt = None
        return rag_output

    else:
        return rag_output


@app.callback(
    Output({'type': 'llm-media-modal', 'index': MATCH}, 'is_open'),
    Input({'type': 'llm-media-figure', 'index': MATCH}, 'n_clicks'),
    State({'type': 'llm-media-figure', 'index': MATCH}, 'id'),
)
def show_figure_modal(n_clicks, id):
    if (n_clicks is not None and n_clicks > 0):
        return True
    else:
        return False

@app.callback(
    Output({'type': 'llm-media-explanation', 'index': MATCH}, 'children'),
    Output({'type': 'llm-media-explanation', 'index': MATCH}, 'style'),
    Output({'type': 'llm-media-button', 'index': MATCH}, 'children', allow_duplicate=True),
    Input({'type': 'llm-media-button', 'index': MATCH}, 'children'),
    State({'type': 'llm-generated-chart', 'index': MATCH}, 'src'),
    prevent_initial_call=True
)
def explain_llm_figure(_, content):
    if content is not None:
        explanation = global_vars.agent.describe_image('', content)
        return dcc.Markdown(explanation.content,className="llm-text"), {"display": "block", "marginBottom": "20px", "marginTop": "20px"}, "Explain"


# Handle Chat Detect Button Click
@app.callback(
    [Output("query-area", "children", allow_duplicate=True),
     Output("error-alert", "is_open", allow_duplicate=True),
     Output("error-alert", "children", allow_duplicate=True),
     Output("chat-update-trigger", "data", allow_duplicate=True),
     Output("next-suggested-questions", "children", allow_duplicate=True),
     Output("commands-input", "value", allow_duplicate=True),
     Output("query-input", "value", allow_duplicate=True),
     Output("pipeline-alert", "is_open", allow_duplicate=True)],
    [Input("chat-detect-btn", "n_clicks")],
    [State("query-area", "children"),
     State("next-suggested-questions", "children")],
    prevent_initial_call=True
)
def handle_chat_detect_button(n_clicks, query_records, suggested_questions):
    """Handle detect button click in chat - generate complete detect analysis"""
    if not n_clicks:
        raise PreventUpdate
    
    import time
    from UI.functions.global_vars import global_vars
    
    print(f"🎯 [DETECT_BUTTON] === DETECT BUTTON CLICKED === (click #{n_clicks})")
    print(f"🎯 [DETECT_BUTTON] Current query_records length: {len(query_records) if query_records else 0}")
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []
    else:
        # Ensure only one Detect bubble exists to avoid duplicate fixed IDs
        # Remove any previous chat-detect bubbles before inserting a new one
        def _get_node_id(n):
            if hasattr(n, 'id'):
                return n.id
            if isinstance(n, dict):
                props = n.get('props') or {}
                return props.get('id')
            return None

        filtered_records = []
        for node in query_records:
            nid = _get_node_id(node)
            if isinstance(nid, dict) and nid.get('original_type') == 'chat-detect-bubble':
                # Skip old detect bubbles to keep IDs unique per session
                continue
            filtered_records.append(node)
        query_records = filtered_records
    
    # ==========================================================================
    # ON-DEMAND METRICS CALCULATION
    # ==========================================================================
    # Metrics are calculated only when this button is clicked, not beforehand.
    # This ensures optimal performance and user control over computation timing.
    # ==========================================================================
    
    try:
        # -----------------------------------------------------------------------
        # Prerequisite Check 1: Target Attribute
        # -----------------------------------------------------------------------
        has_target = hasattr(global_vars, 'target_attribute') and global_vars.target_attribute is not None
        
        if not has_target:
            print("[DETECT_BTN] No target attribute selected")
            error_bubble = create_timestamped_message([
                dbc.Alert([
                    html.H5("🎯 Target Attribute Required", className="alert-heading"),
                    html.P("Please select a target attribute before calculating distribution shift metrics."),
                    html.Hr(),
                    html.P("Click the target attribute button in the menu to get started.", className="mb-0")
                ], color="warning")
            ], "llm-msg")
            query_records.append(error_bubble)
            sorted_records = sort_chat_messages(query_records)
            return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)
        
        # -----------------------------------------------------------------------
        # Prerequisite Check 2: Datasets
        # -----------------------------------------------------------------------
        has_primary = hasattr(global_vars, 'df') and global_vars.df is not None
        has_secondary = hasattr(global_vars, 'secondary_df') and global_vars.secondary_df is not None
        
        if not (has_primary and has_secondary):
            print("[DETECT_BTN] ❌ Missing datasets")
            missing = []
            if not has_primary:
                missing.append("primary dataset")
            if not has_secondary:
                missing.append("secondary dataset")
            
            error_bubble = create_timestamped_message([
                dbc.Alert([
                    html.H5("📊 Datasets Required", className="alert-heading"),
                    html.P(f"Please upload the {' and '.join(missing)} before calculating metrics."),
                ], color="warning")
            ], "llm-msg")
            query_records.append(error_bubble)
            sorted_records = sort_chat_messages(query_records)
            return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)
        
        print(f"[DETECT_BTN] Prerequisites met - Target: {global_vars.target_attribute}")
        
        # -----------------------------------------------------------------------
        # Check Metrics Status
        # -----------------------------------------------------------------------
        if not hasattr(global_vars, 'metrics_calculation_status'):
            global_vars.metrics_calculation_status = "idle"
        
        # Check if metrics need calculation
        is_cache_valid, cache_reason = global_vars.is_cache_valid()
        needs_calculation = (
            global_vars.metrics_calculation_status in ["idle", "pending", "failed"] or 
            not is_cache_valid
        )
        
        if needs_calculation:
            print(f"[DETECT_BTN] Metrics need calculation - Status: {global_vars.metrics_calculation_status}, Cache: {cache_reason}")
            
            # Show "Calculating..." message
            calculating_bubble = create_timestamped_message([
                dbc.Alert([
                    dbc.Spinner(size="sm", color="primary", spinner_style={"marginRight": "0.5rem"}),
                    html.Span("🔄 Calculating distribution shift metrics... This may take 10-30 seconds depending on dataset size."),
                ], color="info", className="mb-0")
            ], "llm-msg")
            query_records.append(calculating_bubble)
        else:
            print(f"[DETECT_BTN] Using cached metrics (Status: {global_vars.metrics_calculation_status})")
        
        # -----------------------------------------------------------------------
        # Get or Calculate Metrics
        # -----------------------------------------------------------------------
        from UI.utils.detect_utils import get_fresh_metrics_data_for_chat
        metrics_data, data_length = get_fresh_metrics_data_for_chat()
        
        # -----------------------------------------------------------------------
        # Validate Results
        # -----------------------------------------------------------------------
        if not metrics_data or len(metrics_data) == 0:
            print("[DETECT_BTN] ❌ Metrics calculation failed or returned empty results")
            
            # Remove the "Calculating..." message if it was added
            query_records = [r for r in query_records 
                           if not (isinstance(r, dict) and 'Calculating' in str(r))]
            
            # Add error message
            error_bubble = create_timestamped_message([
                dbc.Alert([
                    html.H5("⚠️ Calculation Failed", className="alert-heading"),
                    html.P("Unable to calculate distribution shift metrics. Please check the console for details."),
                    html.Hr(),
                    html.P([
                        "Common causes:",
                        html.Ul([
                            html.Li("Datasets have no common columns"),
                            html.Li("Target attribute not found in datasets"),
                            html.Li("Insufficient data for statistical analysis"),
                        ])
                    ], className="mb-0", style={"fontSize": "0.9em"})
                ], color="danger")
            ], "llm-msg")
            query_records.append(error_bubble)
            sorted_records = sort_chat_messages(query_records)
            return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)
        
        # -----------------------------------------------------------------------
        # Success: Remove "Calculating..." message and proceed
        # -----------------------------------------------------------------------
        if needs_calculation:
            # Remove the "Calculating..." message
            query_records = [r for r in query_records 
                           if not (isinstance(r, dict) and 'Calculating' in str(r))]
            print(f"[DETECT_BTN] ✅ Metrics calculated successfully: {len(metrics_data)} columns")
        else:
            print(f"[DETECT_BTN] ✅ Using cached metrics: {len(metrics_data)} columns")
        
        # If metrics are not ready, block until calculation inside get_fresh_metrics_data_for_chat finishes
        # dcc.Loading in the chat layout will indicate progress; avoid inserting a provisional chat bubble
        
        # ✅ CRITICAL: Set global state to detect BEFORE creating components
        # This ensures the system knows we're in detect stage and maintains consistency with explain button
        if hasattr(global_vars, 'current_stage'):
            old_stage = global_vars.current_stage.lower()
            global_vars.current_stage = "detect"
            print(f"[CHAT DETECT] Changed global stage: {old_stage} -> detect")
        else:
            global_vars.current_stage = "detect"
            print(f"[CHAT DETECT] Set global stage to: detect")
        
        # Generate the COMPLETE detect analysis - copy the exact same logic from home_layout.py
        detect_component = generate_complete_detect_analysis_for_chat(metrics_data, data_length)
        
        # Create unique ID for this chat instance
        table_id = f"chat-detect-{int(time.time() * 1000)}"
        
        # Create chat bubble with the complete detect analysis
        header = dbc.CardHeader([
            html.Div([
                html.Div([
                    html.I(className="fas fa-search-plus me-2"),
                    html.Strong("Detect Stage")
                ], style={"display": "inline-block"}),
                html.Div([
                    html.I(id={"type": "chat-detect-toggle", "index": table_id}, 
                           className="fa fa-chevron-up", 
                           style={"cursor": "pointer", "fontSize": "18px"})
                ], style={"display": "inline-block", "float": "right"})
            ])
        ])
        
        collapse = dbc.Collapse(
            dbc.CardBody(detect_component, style={"padding": "15px"}),
            id={"type": "chat-detect-collapse", "index": table_id},
            is_open=True
        )
        
        bubble_content = dbc.Card([header, collapse], className="mb-2")
        bubble = create_timestamped_message(
            bubble_content,
            "llm-msg chat-table-container"
        )
        # Merge sequence ID with type identification  
        if hasattr(bubble, 'id') and isinstance(bubble.id, dict):
            bubble.id.update({"original_type": "chat-detect-bubble", "index": table_id})
        
        query_records.append(bubble)
        
        print(f"[CHAT DETECT] ✅ Added complete detect analysis to chat")
        
    except Exception as e:
        print(f"[CHAT DETECT] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_bubble = create_timestamped_message([
            dbc.Alert([
                html.H5("❌ Error Generating Analysis", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the console for details.", className="mb-0")
            ], color="danger")
        ], "llm-msg")
        query_records.append(error_bubble)
    
    sorted_records = sort_chat_messages(query_records)
    return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)


# Pipeline Button State Management - Update button classes when clicked
@app.callback(
    [Output("chat-detect-btn", "className"),
     Output("chat-explain-btn", "className"),
     Output("chat-adapt-btn", "className")],
    [Input("chat-detect-btn", "n_clicks"),
     Input("chat-explain-btn", "n_clicks"),
     Input("chat-adapt-btn", "n_clicks")],
    prevent_initial_call=True
)
def update_pipeline_button_states(detect_clicks, explain_clicks, adapt_clicks):
    """Update pipeline button classes based on which button was clicked"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Default state: Detect active, others inactive
        return "pipeline-chat-btn active", "pipeline-chat-btn", "pipeline-chat-btn disabled"
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Base classes for each button
    detect_class = "pipeline-chat-btn"
    explain_class = "pipeline-chat-btn"
    adapt_class = "pipeline-chat-btn"
    
    # Set active class based on which button was clicked
    if triggered_id == "chat-detect-btn":
        detect_class += " active"
    elif triggered_id == "chat-explain-btn":
        explain_class += " active"
    elif triggered_id == "chat-adapt-btn":
        adapt_class += " active"
    else:
        # Default: Detect is active
        detect_class += " active"
    
    return detect_class, explain_class, adapt_class


# Handle Chat Explain Button Click
@app.callback(
    [Output("query-area", "children", allow_duplicate=True),
     Output("error-alert", "is_open", allow_duplicate=True),
     Output("error-alert", "children", allow_duplicate=True),
     Output("chat-update-trigger", "data", allow_duplicate=True),
     Output("next-suggested-questions", "children", allow_duplicate=True),
     Output("commands-input", "value", allow_duplicate=True),
     Output("query-input", "value", allow_duplicate=True),
     Output("pipeline-alert", "is_open", allow_duplicate=True)],
    [Input("chat-explain-btn", "n_clicks")],
    [State("query-area", "children"),
     State("next-suggested-questions", "children")],
    prevent_initial_call=True
)
def handle_chat_explain_button(n_clicks, query_records, suggested_questions):
    """Handle explain button click in chat - generate complete explain analysis"""
    if not n_clicks:
        raise PreventUpdate
    
    import time
    from UI.functions.global_vars import global_vars
    
    print(f"📝 [EXPLAIN_BUTTON] === EXPLAIN BUTTON CLICKED === (click #{n_clicks})")
    print(f"📝 [EXPLAIN_BUTTON] Current query_records length: {len(query_records) if query_records else 0}")
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []

    # Enforce single-instance: remove any existing Explain Step 1 bubbles
    try:
        def _get_node_id(n):
            if hasattr(n, 'id'):
                return n.id
            if isinstance(n, dict):
                props = n.get('props') or {}
                return props.get('id')
            return None

        def _is_context_bubble(node):
            nid = _get_node_id(node)
            return isinstance(nid, dict) and nid.get('original_type') == 'chat-context-bubble'

        if query_records:
            query_records = [child for child in query_records if not _is_context_bubble(child)]
    except Exception:
        # Fail-safe: if filtering fails, continue without removal
        pass
    
    try:
        print(f"[CHAT EXPLAIN] 🚀 Starting explain analysis generation...")
        
        # Get explain context data (same as original explain component)
        from UI.state_connector import get_explain_context
        explain_context = get_explain_context(stage=global_vars.current_stage)
        
        # Check if we have context data to work with
        if not explain_context or not explain_context.get('metrics_data'):
            print(f"[CHAT EXPLAIN] ⚠️ No explain context available")
            
            # Create warning bubble
            warning_bubble = html.Div([
                dbc.Alert([
                    html.H5("⚠️ No Analysis Context Available", className="alert-heading"),
                    html.P("Please run the Detect stage first to generate data for explanation analysis."),
                    html.Hr(),
                    html.P("The Explain stage requires context from the Detect phase to provide meaningful insights.", className="mb-0")
                ], color="warning")
            ], className="llm-msg")
            query_records.append(warning_bubble)
            
            sorted_records = sort_chat_messages(query_records)
            return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)
        
        # ✅ NEW APPROACH: Generate ONLY Context Items component for filtering
        # Users will then click "Start Explain Analysis" to get the full analysis
        
        # Create unique ID for this chat instance
        table_id = f"chat-explain-{int(time.time() * 1000)}"
        
        # Set global stage to explain BEFORE creating components
        if hasattr(global_vars, 'current_stage'):
            old_stage = global_vars.current_stage
            global_vars.current_stage = "explain"
            print(f"[CHAT EXPLAIN] Changed global stage: {old_stage} -> explain")
        else:
            global_vars.current_stage = "explain"
            print(f"[CHAT EXPLAIN] Set global stage to: explain")
        
        # Generate ONLY the Context Items component (first stage)
        context_items_component = create_context_items_chat_component(explain_context, table_id)
        
        # Create chat bubble with Context Items (first stage of explain process)
        header = dbc.CardHeader([
            html.Div([
                html.Div([
                    html.I(className="fas fa-layer-group me-2"),
                    html.Strong("Context Items - Step 1")
                ], style={"display": "inline-block"}),
                html.Div([
                    html.I(id={"type": "chat-context-toggle", "index": table_id}, 
                           className="fa fa-chevron-up", 
                           style={"cursor": "pointer", "fontSize": "18px"})
                ], style={"display": "inline-block", "float": "right"})
            ])
        ])
        
        collapse = dbc.Collapse(
            dbc.CardBody(context_items_component, style={"padding": "15px"}),
            id={"type": "chat-context-collapse", "index": table_id},
            is_open=True
        )
        
        bubble_content = dbc.Card([header, collapse], className="mb-2")
        bubble = create_timestamped_message(
            bubble_content,
            "llm-msg chat-table-container"
        )
        # Merge sequence ID with type identification (single-instance marker)
        if hasattr(bubble, 'id') and isinstance(bubble.id, dict):
            bubble.id.update({"original_type": "chat-context-bubble", "index": table_id})
        
        query_records.append(bubble)
        
        print(f"[CHAT EXPLAIN] ✅ Added complete explain analysis to chat")
        
    except Exception as e:
        print(f"[CHAT EXPLAIN] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_bubble = html.Div([
            dbc.Alert([
                html.H5("❌ Error Generating Explain Analysis", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the console for details.", className="mb-0")
            ], color="danger")
        ], className="llm-msg")
        query_records.append(error_bubble)
    
    sorted_records = sort_chat_messages(query_records)
    return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)


# Handle Chat Adapt Button Click
@app.callback(
    [Output("query-area", "children", allow_duplicate=True),
     Output("error-alert", "is_open", allow_duplicate=True),
     Output("error-alert", "children", allow_duplicate=True),
     Output("chat-update-trigger", "data", allow_duplicate=True),
     Output("next-suggested-questions", "children", allow_duplicate=True),
     Output("commands-input", "value", allow_duplicate=True),
     Output("query-input", "value", allow_duplicate=True),
     Output("pipeline-alert", "is_open", allow_duplicate=True)],
    [Input("chat-adapt-btn", "n_clicks")],
    [State("query-area", "children"),
     State("next-suggested-questions", "children")],
    prevent_initial_call=True
)
def handle_chat_adapt_button(n_clicks, query_records, suggested_questions):
    """Handle adapt button click in chat - show strategy selection and generate adapt analysis"""
    if not n_clicks:
        raise PreventUpdate
    
    import time
    from UI.functions.global_vars import global_vars
    
    print(f"🔧 [ADAPT_BUTTON] === ADAPT BUTTON CLICKED === (click #{n_clicks})")
    print(f"🔧 [ADAPT_BUTTON] Current query_records length: {len(query_records) if query_records else 0}")

  
    try:
        if hasattr(global_vars, 'current_stage'):
            old_stage = global_vars.current_stage
            global_vars.current_stage = "adapt"
            print(f"🔧 [ADAPT_BUTTON] Changed global stage: {old_stage} -> adapt")
    except Exception as _:
      
        pass
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []
    
    # Check if datasets are available
    if global_vars.df is None or global_vars.secondary_df is None:
        error_bubble = create_timestamped_message([
            dbc.Alert([
                html.H5([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "Datasets Required"
                ], className="text-warning"),
                html.P("Please upload both primary and secondary datasets before starting adaptation analysis.", className="mb-2"),
                html.P("Click the target attribute button in the menu to get started.", className="mb-0")
            ], color="warning")
        ], "llm-msg")
        query_records.append(error_bubble)
        sorted_records = sort_chat_messages(query_records)
        return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)
    
    # Check if adaptation strategy has been selected
    if not global_vars.is_adaptation_strategy_selected():
        # Remove any existing adapt selection bubbles to avoid duplicates
        def _extract_node_id(node):
            if hasattr(node, 'id'):
                return node.id
            if isinstance(node, dict):
                props = node.get('props') or {}
                return props.get('id')
            return None

        if query_records:
            filtered_records = []
            for node in query_records:
                nid = _extract_node_id(node)
                if isinstance(nid, dict) and nid.get('original_type') == 'chat-adapt-selection':
                    continue
                filtered_records.append(node)
            query_records = filtered_records

        adapt_card_id = f"chat-adapt-{int(time.time() * 1000)}"

        header = dbc.CardHeader([
            html.Div([
                html.Div([
                    html.I(className="fas fa-wrench me-2"),
                    html.Strong("Adapt Stage")
                ], style={"display": "inline-block"}),
                html.Div([
                    html.I(
                        id={"type": "chat-adapt-toggle", "index": adapt_card_id},
                        className="fa fa-chevron-up",
                        style={"cursor": "pointer", "fontSize": "18px"}
                    )
                ], style={"display": "inline-block", "float": "right"})
            ])
        ])

        card_body = dbc.CardBody([
            html.H5("Select Adaptation Strategy", className="text-primary mb-3"),
            html.P(
                "Choose your approach for adapting the model to handle distribution shifts:",
                className="mb-3"
            ),
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Retrain Model"
                    ],
                    id="chat-retrain-strategy-btn",
                    color="danger",
                    size="lg",
                    className="me-2",
                    n_clicks=0),
                    dbc.Button([
                        html.I(className="fas fa-tools me-2"),
                        "Finetune Model"
                    ],
                    id="chat-finetune-strategy-btn",
                    color="warning",
                    size="lg",
                    n_clicks=0)
                ], className="d-flex justify-content-center")
            ], className="mb-4"),
            html.Div([
                html.P("Retrain Model: Complete rebuild using both datasets", className="small text-muted mb-1"),
                html.P("Finetune Model: Incremental adaptation of existing model", className="small text-muted mb-0")
            ], className="text-center")
        ], style={"padding": "15px"})

        collapse = dbc.Collapse(
            card_body,
            id={"type": "chat-adapt-collapse", "index": adapt_card_id},
            is_open=True
        )

        bubble_content = dbc.Card([header, collapse], className="mb-2")

        strategy_prompt_bubble = create_timestamped_message(
            bubble_content,
            "llm-msg chat-table-container"
        )

        if hasattr(strategy_prompt_bubble, 'id') and isinstance(strategy_prompt_bubble.id, dict):
            strategy_prompt_bubble.id.update({
                "original_type": "chat-adapt-selection",
                "index": adapt_card_id
            })

        query_records.append(strategy_prompt_bubble)

        sorted_records = sort_chat_messages(query_records)
        return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)
    
    # Strategy has been selected - proceed with adaptation analysis
    selected_strategy = global_vars.get_adaptation_strategy()
    print(f"🔧 [ADAPT_BUTTON] Proceeding with strategy: {selected_strategy}")
    
    # Set global stage to adapt
    if hasattr(global_vars, 'current_stage'):
        old_stage = global_vars.current_stage
        global_vars.current_stage = "adapt"
        print(f"🔧 [ADAPT_BUTTON] Changed global stage: {old_stage} -> adapt")
    
    # Remove previous adapt analysis bubbles to keep only the latest view
    def _extract_node_id(node):
        if hasattr(node, 'id'):
            return node.id
        if isinstance(node, dict):
            props = node.get('props') or {}
            return props.get('id')
        return None

    if query_records:
        filtered_records = []
        for node in query_records:
            nid = _extract_node_id(node)
            if isinstance(nid, dict) and nid.get('original_type') == 'chat-adapt-bubble':
                continue
            filtered_records.append(node)
        query_records = filtered_records

    analysis_card_id = f"chat-adapt-analysis-{int(time.time() * 1000)}"

    header = dbc.CardHeader([
        html.Div([
            html.Div([
                html.I(className="fas fa-wrench me-2", style={"color": "#28a745"}),
                html.Strong(f"Adaptation Analysis - {selected_strategy.title()}")
            ], style={"display": "inline-block"}),
            html.Div([
                html.I(
                    id={"type": "chat-adapt-toggle", "index": analysis_card_id},
                    className="fa fa-chevron-up",
                    style={"cursor": "pointer", "fontSize": "18px"}
                )
            ], style={"display": "inline-block", "float": "right"})
        ])
    ])

    body = dbc.CardBody([
        html.P(
            f"Starting adaptation analysis using {selected_strategy} strategy...",
            className="mb-3"
        ),
        html.Div([
            html.P("🔧 Adaptation analysis will be implemented here", className="text-muted mb-2"),
            html.P("This will include:", className="mb-2"),
            html.Ul([
                html.Li("Data preparation based on selected strategy"),
                html.Li("Distribution analysis and alignment"),
                html.Li("Model adaptation recommendations"),
                html.Li("Implementation guidance")
            ], className="small text-muted")
        ], className="p-3 bg-light rounded")
    ], style={"padding": "15px"})

    collapse = dbc.Collapse(
        body,
        id={"type": "chat-adapt-collapse", "index": analysis_card_id},
        is_open=True
    )

    bubble_content = dbc.Card([header, collapse], className="mb-2")

    adapt_bubble = create_timestamped_message(
        bubble_content,
        "llm-msg chat-table-container"
    )

    if hasattr(adapt_bubble, 'id') and isinstance(adapt_bubble.id, dict):
        adapt_bubble.id.update({"original_type": "chat-adapt-bubble", "index": analysis_card_id})

    query_records.append(adapt_bubble)
    
    sorted_records = sort_chat_messages(query_records)
    return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)


# Handle Strategy Selection Button Clicks
@app.callback(
    [Output("query-area", "children", allow_duplicate=True),
     Output("error-alert", "is_open", allow_duplicate=True),
     Output("error-alert", "children", allow_duplicate=True),
     Output("chat-update-trigger", "data", allow_duplicate=True),
     Output("next-suggested-questions", "children", allow_duplicate=True),
     Output("commands-input", "value", allow_duplicate=True),
     Output("query-input", "value", allow_duplicate=True),
     Output("pipeline-alert", "is_open", allow_duplicate=True)],
    [Input("chat-retrain-strategy-btn", "n_clicks"),
     Input("chat-finetune-strategy-btn", "n_clicks")],
    [State("query-area", "children"),
     State("next-suggested-questions", "children")],
    prevent_initial_call=True
)
def handle_strategy_button_click(retrain_clicks, finetune_clicks, query_records, suggested_questions):
    """Handle strategy button clicks and proceed with adaptation analysis"""
    if not retrain_clicks and not finetune_clicks:
        raise PreventUpdate
    
    import time
    from UI.functions.global_vars import global_vars
    
    # Determine which strategy was selected
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if triggered_id == "chat-retrain-strategy-btn":
        selected_strategy = "retrain"
        strategy_name = "Retrain Model"
        strategy_icon = "fas fa-sync-alt"
        strategy_color = "danger"
    elif triggered_id == "chat-finetune-strategy-btn":
        selected_strategy = "finetune"
        strategy_name = "Finetune Model"
        strategy_icon = "fas fa-tools"
        strategy_color = "warning"
    else:
        raise PreventUpdate
    
    print(f"🔧 [STRATEGY_BUTTON] {strategy_name} selected (click #{retrain_clicks + finetune_clicks})")
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []
    
    # Save strategy to global variables
    global_vars.set_adaptation_strategy(selected_strategy)
    
    # Set global stage to adapt
    if hasattr(global_vars, 'current_stage'):
        old_stage = global_vars.current_stage
        global_vars.current_stage = "adapt"
        print(f"🔧 [STRATEGY_BUTTON] Changed global stage: {old_stage} -> adapt")
    
    # Remove existing adapt cards so the latest selection is displayed cleanly
    def _extract_node_id(node):
        if hasattr(node, 'id'):
            return node.id
        if isinstance(node, dict):
            props = node.get('props') or {}
            return props.get('id')
        return None

    if query_records:
        filtered_records = []
        for node in query_records:
            nid = _extract_node_id(node)
            if isinstance(nid, dict) and nid.get('original_type') in {"chat-adapt-selection", "chat-adapt-bubble"}:
                continue
            filtered_records.append(node)
        query_records = filtered_records

    analysis_card_id = f"chat-adapt-analysis-{int(time.time() * 1000)}"

    # Check if finetune strategy - use interactive finetune config bubble
    if selected_strategy == "finetune":
        from UI.pages.components.adapt_finetune import create_finetune_config_bubble
        bubble_id = f"finetune-{int(time.time() * 1000)}"
        bubble_content = create_finetune_config_bubble(bubble_id)
        
        strategy_confirmation_bubble = create_timestamped_message(
            bubble_content,
            "llm-msg"
        )
    elif selected_strategy == "retrain":
        # Retrain strategy - use interactive retrain config bubble
        from UI.pages.components.adapt_retrain import create_retrain_config_bubble
        bubble_id = f"retrain-{int(time.time() * 1000)}"
        bubble_content = create_retrain_config_bubble(bubble_id)
        
        strategy_confirmation_bubble = create_timestamped_message(
            bubble_content,
            "llm-msg"
        )

    if hasattr(strategy_confirmation_bubble, 'id') and isinstance(strategy_confirmation_bubble.id, dict):
        strategy_confirmation_bubble.id.update({
            "original_type": "chat-adapt-bubble",
            "index": f"adapt-{int(time.time() * 1000)}"
        })

    query_records.append(strategy_confirmation_bubble)
    
    sorted_records = sort_chat_messages(query_records)
    return (sorted_records, False, "", time.time(), suggested_questions, "", "", dash.no_update)


def create_context_items_chat_component(explain_context, unique_id):
    """
    Create independent Context Items component for chat interface.
    This component allows users to review and filter context items before analysis.
    """
    from dash import html, dcc
    import dash_bootstrap_components as dbc
    from UI.pages.components.explain_component import create_context_items_panel
    from UI.functions.global_vars import global_vars
    
    try:
        # ✅ FIX: Use global_vars.explain_context_data instead of explain_context parameter
        # This ensures Step 1 displays the same items that Step 2 analyzes
        context_data = getattr(global_vars, 'explain_context_data', [])
        
        # Keep other context info from parameter for backward compatibility
        focus_attribute = explain_context.get('focus_attribute')
        target_attribute = explain_context.get('target_attribute')
        
        print(f"[CONTEXT ITEMS CHAT] Creating component with {len(context_data)} items from global_vars")
        print(f"[CONTEXT ITEMS CHAT] Focus: {focus_attribute}, Target: {target_attribute}")
        
        # Create the component with filtering instructions
        return html.Div([
            # ✅ CRITICAL: Add toast component here so delete callbacks can find it
            dbc.Toast(
                "Context item deleted successfully!",
                id="context-delete-success-toast",
                header="Success",
                is_open=False,
                dismissable=True,
                duration=3000,
                style={"position": "fixed", "top": 20, "right": 20, "width": 300, "zIndex": 9999}
            ),
            
            # Header with instructions
            html.Div([
                html.H4([
                    html.I(className="fas fa-layer-group me-2", style={"color": "#614385"}),
                    "Context Items for Analysis"
                ], className="mb-3", style={"color": "#614385"}),
                
                # Analysis context info
                html.Div([
                    html.P([
                        html.Strong("Analysis Focus: "),
                        html.Span(focus_attribute or "Not specified", className="text-primary"),
                        html.Br(),
                        html.Strong("Target Attribute: "),
                        html.Span(target_attribute or "Not specified", className="text-success")
                    ], className="mb-2"),
                    
                    dbc.Alert([
                        html.I(className="fas fa-info-circle me-2"),
                        "Review the context items below. You can expand each item to see details, and remove any items that are not relevant to your analysis. When ready, click 'Start Explain Analysis' to proceed."
                    ], color="info", className="mb-3")
                ])
            ]),
            
            # Context Items Panel (reuse existing logic with ORIGINAL IDs)
            html.Div([
                # Stage is now managed through proper Dash store synchronization
                
                # Create stores needed for context items functionality (using ORIGINAL IDs)
                dcc.Store(id="explain-context-data", data=context_data),
                
                # Context items panel container (using ORIGINAL IDs for callback compatibility)
                # ✅ This will be populated by the existing callback that monitors explain-context-data
                dcc.Loading(
                    id="explain-context-loading",
                    type="circle",
                    children=html.Div(id="explain-context-items-panel")  # ✅ 使用原始ID，让现有回调更新它
                )
            ]),
            
            # Action Section
            html.Hr(className="my-4"),
            html.Div([
                html.Div([
                    html.P([
                        html.I(className="fas fa-lightbulb me-2", style={"color": "#ffc107"}),
                        html.Strong("Ready to proceed with analysis?")
                    ], className="mb-2", style={"fontSize": "16px"}),
                    html.P("The analysis will use the context items shown above to generate comprehensive insights.", 
                           className="text-muted mb-3")
                ]),
                
                dbc.Button([
                    html.I(className="fas fa-play me-2"),
                    "Start Explain Analysis"
                ], 
                id={"type": "chat-start-explain-btn", "index": unique_id},
                color="primary", 
                size="lg",
                className="w-100",
                style={"padding": "12px 24px", "fontWeight": "600"})
            ], className="text-center", style={
                "backgroundColor": "#ffffff",
                "padding": "20px",
                "borderRadius": "8px",
                "border": "1px solid #dee2e6"
            })
        ], id=f"context-items-chat-component-{unique_id}")
        
    except Exception as e:
        print(f"[CONTEXT ITEMS CHAT] Error creating component: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return html.Div([
            dbc.Alert([
                html.H5("❌ Error Creating Context Items", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the console for details.", className="mb-0")
            ], color="danger")
        ])


def create_simplified_context_items_panel(context_data, unique_id):
    """
    Create a simplified context items panel for chat interface.
    This version doesn't rely on external callbacks and uses inline JavaScript for interactions.
    """
    if not context_data:
        return html.Div([
            html.Div([
                html.I(className="fas fa-info-circle me-2", style={"color": "#6c757d", "fontSize": "2rem"}),
                html.H5("No Context Items", style={"color": "#6c757d"}),
                html.P("Context items will appear here when you add them using 'Add to Explain' buttons", className="text-muted")
            ], className="text-center", style={"padding": "60px 20px"})
        ])
    
    # Type display configuration
    type_config = {
        "target_distribution": {
            "name": "Target Distribution", 
            "icon": "fas fa-chart-bar", 
            "color": "#614385"
        },
        "conditional_distribution": {
            "name": "Conditional Distribution", 
            "icon": "fas fa-project-diagram", 
            "color": "#e67e22"
        },
        "distribution_comparison": {
            "name": "Distribution Comparison", 
            "icon": "fas fa-balance-scale", 
            "color": "#27ae60"
        },
        "metric": {
            "name": "Statistical Metrics", 
            "icon": "fas fa-calculator", 
            "color": "#3498db"
        },
        "drift_analysis": {
            "name": "Drift Analysis", 
            "icon": "fas fa-exclamation-triangle", 
            "color": "#e74c3c"
        }
    }
    
    # Create items list
    items_list = []
    for i, item in enumerate(context_data):
        item_type = item.get("type", "unknown")
        item_id = item.get("id", f"item-{i}")
        config = type_config.get(item_type, {"name": "Unknown", "icon": "fas fa-question", "color": "#6c757d"})
        
        # Create simplified item card
        item_card = dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Div([
                        html.I(className=config["icon"], style={"color": config["color"], "marginRight": "8px"}),
                        html.Strong(config["name"], style={"color": config["color"]})
                    ], className="d-flex align-items-center"),
                    html.Div([
                        dbc.Button(
                            html.I(className="fas fa-eye"),
                            id=f"view-{item_id}-{unique_id}",
                            color="info",
                            size="sm",
                            outline=True,
                            title="View details",
                            className="me-2"
                        ),
                        dbc.Button(
                            html.I(className="fas fa-trash-alt"),
                            id=f"delete-{item_id}-{unique_id}",
                            color="danger",
                            size="sm",
                            outline=True,
                            title="Remove from analysis"
                        )
                    ])
                ], className="d-flex justify-content-between align-items-center")
            ]),
            dbc.CardBody([
                html.P(item.get("summary_text", "No description available")[:100] + "...", 
                       className="text-muted mb-2"),
                html.Small(f"Added: {item.get('timestamp', 'Unknown')}", className="text-muted")
            ])
        ], className="mb-3")
        
        items_list.append(item_card)
    
    return html.Div([
        html.Div([
            html.P(f"Found {len(context_data)} context items for analysis:", className="mb-3 fw-bold"),
            *items_list
        ])
    ])


def generate_complete_explain_analysis_for_chat(explain_context):
    """
    Generate the complete explain analysis for chat - FULLY REUSING existing explain component logic
    This creates a self-contained explain component that works within the chat interface
    """
    from dash import html, dcc
    import dash_bootstrap_components as dbc
    from UI.functions.global_vars import global_vars
    
    # Import all necessary functions from the original explain component
    from UI.pages.components.explain_component import (
        create_context_items_panel,
        create_comprehensive_analysis_panel,
        create_on_demand_analysis_panel
    )
    from UI.state_connector import record_analysis_path
    
    try:
        # Extract context data (same as original explain component)
        focus_attribute = explain_context.get('focus_attribute')
        target_attribute = explain_context.get('target_attribute')
        metrics_data = explain_context.get('metrics_data', [])
        
        # Record transition to explain phase (same as original)
        record_analysis_path('enter_explain_phase', {
            'focus_attribute': focus_attribute,
            'target_attribute': target_attribute
        })
        
        print(f"[CHAT EXPLAIN] Creating explain component with context: focus={focus_attribute}, target={target_attribute}")
        
        # Create unique IDs for this chat instance to avoid conflicts
        import time
        unique_id = f"chat-explain-{int(time.time() * 1000)}"
        
        # ⚠️ ID STRATEGY: Use original IDs for callbacks compatibility, but handle conflicts
        # The existing explain callbacks expect specific IDs like "explain-context-items-panel"
        # We'll create components with original IDs, but wrap them in a unique container
        print(f"[CHAT EXPLAIN] Using unique ID: {unique_id}")
        
        # Create necessary Store components for chat explain (using ORIGINAL IDs for callback compatibility)
        stores = [
            # Original stores (using FIXED IDs for callback compatibility)
            dcc.Store(id="explain-focus-attribute", data=focus_attribute),
            dcc.Store(id="explain-primary-dist", data={}),
            dcc.Store(id="explain-secondary-dist", data={}),
            dcc.Store(id="explain-col-type", data=""),
            dcc.Store(id="explain-top-k-attrs", data=[]),
            dcc.Store(id="explain-conditional-primary-dist", data={}),
            dcc.Store(id="explain-conditional-secondary-dist", data={}),
            
            # Context-based analysis stores (using FIXED IDs)
            dcc.Store(id="explain-selected-context", data=None),
            dcc.Store(id="explain-analysis-type", data=None),
            dcc.Store(id="explain-analysis-results", data={}),
            dcc.Store(id="explain-context-groups", data={}),
            dcc.Store(id="explain-context-expanded-states", data={}),
            
            # UNIFIED ANALYSIS stores (using FIXED IDs) - unified-strategy-analysis now created in comprehensive panel
            dcc.Store(id="unified-selected-strategy", data="monitor"),
            dcc.Store(id="unified-perspective-state", data="technical"),
            dcc.Store(id="unified-analysis-cache", data={}),
            
            # ✅ CRITICAL FIX: Use html.Div instead of dcc.Store for current-stage
            # Explain callbacks listen to current-stage.children, not current-stage.data
            html.Div(id="current-stage", children="explain", style={"display": "none"}),
            
            # Explain context data store (CRITICAL for callbacks to work) - will be set below
            dcc.Store(id="explain-context-data", data=[]),
        ]
        
        # ✅ ENHANCED: Get explain context data with better fallback handling
        existing_context_data = []
        
        # First, try to get from global explain context
        if hasattr(global_vars, 'explain_context_data') and global_vars.explain_context_data:
            existing_context_data = global_vars.explain_context_data
            print(f"[CHAT EXPLAIN] Using existing explain context data: {len(existing_context_data)} items")
        
        # Second, try to create basic context from metrics data if available
        elif explain_context.get('metrics_data'):
            print(f"[CHAT EXPLAIN] No explain context data, creating basic context from metrics")
            # Create a basic context item from the explain_context
            basic_context = {
                'id': f'basic-context-{int(time.time() * 1000)}',
                'type': 'metrics_summary',
                'timestamp': time.time(),
                'summary_text': f"Metrics analysis for {len(explain_context['metrics_data'])} attributes",
                'metrics_data': explain_context['metrics_data'],
                'focus_attribute': focus_attribute,
                'target_attribute': target_attribute
            }
            existing_context_data = [basic_context]
            print(f"[CHAT EXPLAIN] Created basic context with metrics data")
        
        else:
            print(f"[CHAT EXPLAIN] No context data available, will show on-demand panel")
        
        # Use existing context data if available, otherwise use empty list
        final_context_data = existing_context_data if existing_context_data else []
        print(f"[CHAT EXPLAIN] Final context data: {len(final_context_data)} items")
        
        # Set store data to match preserved global state (don't overwrite existing context)
        preserved_context_data = getattr(global_vars, 'explain_context_data', final_context_data)
        for store in stores:
            if store.id == "explain-context-data":
                store.data = preserved_context_data  # Use preserved data, not minimal fallback
                print(f"[CHAT EXPLAIN] Store set with {len(preserved_context_data) if preserved_context_data else 0} preserved context items")
                break
        
        # ✅ CRITICAL: Set global state BEFORE creating components
        # This ensures callbacks have access to the correct data when they're triggered
        
        # Set global stage to explain (this affects other parts of the system)
        if hasattr(global_vars, 'current_stage'):
            old_stage = global_vars.current_stage
            global_vars.current_stage = "explain"
            print(f"[CHAT EXPLAIN] Changed global stage: {old_stage} -> explain")
        else:
            global_vars.current_stage = "explain"
            print(f"[CHAT EXPLAIN] Set global stage to: explain")
        
        # Preserve existing context data instead of overwriting it
        if hasattr(global_vars, 'explain_context_data') and global_vars.explain_context_data:
            # Keep existing accumulated items - don't overwrite them
            existing_count = len(global_vars.explain_context_data)
            print(f"[CHAT EXPLAIN] Preserving existing {existing_count} context items (not overwriting)")
            # Don't overwrite: global_vars.explain_context_data = final_context_data  # ❌ This was the bug!
        else:
            # Only set if there's no existing data
            global_vars.explain_context_data = final_context_data if final_context_data else []
            print(f"[CHAT EXPLAIN] Set initial global explain context data: {len(final_context_data) if final_context_data else 0} items")
        
        # ✅ IMPORTANT: Let existing callbacks handle panel population
        # The callbacks will be triggered when components are rendered with correct current-stage
        
        # Create the complete explain component layout (EXACT SAME STRUCTURE as original)
        explain_component = html.Div([
            # Store components (hidden)
            html.Div(stores, style={"display": "none"}),
            
            # ✅ NOTE: Toast component now in context items component, not here
            
            # Header section (same as original)
            html.Div([
                html.H3([
                    html.I(className="fas fa-search-plus me-2", style={"color": "#614385"}),
                    "Distribution Shift Analysis & Explanation"
                ], style={"color": "#614385", "marginBottom": "10px"}),
                
                html.P([
                    "Understand the root causes and business implications of detected distribution shifts. ",
                    "This analysis provides actionable insights for addressing data drift in your ML pipeline."
                ], className="text-muted mb-3"),
                
                # Context info (same as original)
                dbc.Row([
                    dbc.Col([
                        dbc.Badge([
                            html.I(className="fas fa-crosshairs me-1"),
                            f"Focus: {focus_attribute if focus_attribute else 'Auto-selected'}"
                        ], color="primary", className="me-2")
                    ], width="auto"),
                    dbc.Col([
                        dbc.Badge([
                            html.I(className="fas fa-bullseye me-1"), 
                            f"Target: {target_attribute if target_attribute else 'Not specified'}"
                        ], color="secondary", className="me-2")
                    ], width="auto"),
                    dbc.Col([
                        dbc.Badge([
                            html.I(className="fas fa-chart-bar me-1"),
                            f"Metrics: {len(metrics_data) if metrics_data else 0} attributes"
                        ], color="info")
                    ], width="auto")
                ], className="mb-3")
            ]),
            
            # Main content area: Left panel + Right panel (EXACT SAME as original)
            dbc.Row([
                # Left Panel: Context Items (40% width)
                dbc.Col([
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-layer-group me-2", style={"color": "#614385"}),
                            "Context Items"
                        ], className="mb-3", style={"color": "#614385"}),
                        
                        # Context items panel (will be populated by existing callbacks)
                        dcc.Loading(
                            id="explain-context-loading",
                            type="circle",
                            children=html.Div(id="explain-context-items-panel")
                        )
                    ], 
                    style={
                        "backgroundColor": "#f8f9fa",
                        "padding": "15px",
                        "borderRadius": "8px",
                        "border": "1px solid #dee2e6",
                        "minHeight": "400px",
                        "maxHeight": "80vh",
                        "overflowY": "auto"
                    })
                ], xs=12, sm=12, md=5, lg=4, xl=4),
                
                # Right Panel: Analysis Panel (60% width)
                dbc.Col([
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-line me-2", style={"color": "#516395"}),
                            "Analysis Panel"
                        ], className="mb-3", style={"color": "#516395"}),
                        
                        # Analysis panel (will be populated by existing callbacks)
                        dcc.Loading(
                            id="explain-analysis-panel-loading",
                            type="circle",
                            children=html.Div(id="explain-analysis-panel")
                        )
                    ],
                    className="explain-analysis-panel",
                    style={
                        "backgroundColor": "#ffffff",
                        "padding": "15px", 
                        "borderRadius": "8px",
                        "border": "1px solid #dee2e6",
                        "minHeight": "400px",
                        "maxHeight": "80vh",
                        "overflowY": "auto"
                    })
                ], xs=12, sm=12, md=7, lg=8, xl=8)
            ], className="g-3 explain-main-row")  # Same gap class as original
        ], id=f"explain-component-container-{unique_id}")
        
        # # ✅ SUCCESS: Log component creation details
        # print(f"[CHAT EXPLAIN] Successfully created complete explain component")
        # print(f"[CHAT EXPLAIN] Component details:")
        # print(f"  - Focus attribute: {focus_attribute}")
        # print(f"  - Target attribute: {target_attribute}")
        # print(f"  - Metrics data: {len(metrics_data) if metrics_data else 0} attributes")
        # print(f"  - Context data: {len(final_context_data)} items")
        # print(f"  - Global stage: {global_vars.current_stage}")
        # print(f"  - Stores created: {len([s for s in stores if hasattr(s, 'id')])} components")
        
        return explain_component
        
    except Exception as e:
        print(f"[CHAT EXPLAIN] ❌ Error generating explain component: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Enhanced error reporting
        error_details = []
        error_details.append(f"Error: {str(e)}")
        error_details.append(f"Focus attribute: {focus_attribute if 'focus_attribute' in locals() else 'Not set'}")
        error_details.append(f"Target attribute: {target_attribute if 'target_attribute' in locals() else 'Not set'}")
        error_details.append(f"Has metrics data: {bool(explain_context.get('metrics_data')) if 'explain_context' in locals() else 'Unknown'}")
        error_details.append(f"Global stage: {getattr(global_vars, 'current_stage', 'Not set')}")
        
        # Return detailed error component
        return html.Div([
            dbc.Alert([
                html.H5("❌ Error Creating Explain Component", className="alert-heading"),
                html.P("An error occurred while creating the explain analysis:"),
                html.Ul([html.Li(detail) for detail in error_details]),
                html.Hr(),
                html.P([
                    "Troubleshooting steps:",
                    html.Br(),
                    "1. Try running the Detect stage first",
                    html.Br(),
                    "2. Add some items to explain context using 'Add to Explain' buttons",
                    html.Br(),
                    "3. Check the browser console for detailed error logs"
                ], className="mb-0 small")
            ], color="danger", style={"margin": "20px"})
        ])


def generate_complete_detect_analysis_for_chat(metrics_data, data_length):
    """
    Generate the complete detect analysis - EXACT COPY from home_layout.py
    This includes Statistical Drift Detection + Target Distribution Analysis
    """
    from dash import dash_table
    import dash_bootstrap_components as dbc
    from dash import html, dcc
    from UI.functions.global_vars import global_vars
    from UI.utils.detect_utils import calculate_severity_ranking_and_styles, create_metrics_heatmap, create_dual_add_buttons
    
    # Add action buttons to each row (same as original)
    for row in metrics_data:
        row["AddToChat"] = "💬"  # chat emoji 
        row["AddToExplain"] = "📊"  # chart emoji
        row["ExplainAction"] = "Explain"
    
    # Dynamic column generation - EXACT COPY from original
    def create_dynamic_columns(sample_data):
        """Create dynamic column definitions based on actual data"""
        base_columns = [
            {"name": "Attribute", "id": "Attribute", "selectable": True},
            {"name": "Type", "id": "Type", "selectable": False}
        ]
        
        metric_columns = []
        metric_display_names = {
            'JS_Divergence': 'JS Divergence',
            'PSI': 'PSI', 
            'Wasserstein': 'Wasserstein',
            'Chi_Square': 'Chi-Square',
        }
        
        if sample_data:
            first_row = sample_data[0]
            for key in first_row.keys():
                if key not in ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction']:
                    display_name = metric_display_names.get(key, key.replace('_', ' ').title())
                    metric_columns.append({
                        "name": display_name, 
                        "id": key, 
                        "selectable": False
                    })
        
        action_columns = [
            {"name": "Chat", "id": "AddToChat", "selectable": False},
            {"name": "Explain", "id": "AddToExplain", "selectable": False}
        ]
        
        return base_columns + metric_columns + action_columns
    
    columns = create_dynamic_columns(metrics_data)
    
    # Dynamic style conditions - EXACT COPY from original
    def create_dynamic_style_conditions(columns, metrics_data):
        style_conditions = []
        
        for col in columns:
            if col['id'] not in ['Attribute', 'Type', 'AddToChat', 'AddToExplain', 'ExplainAction']:
                style_conditions.append({
                    'if': {'filter_query': f'{{{col["id"]}}} contains "N/A"', 'column_id': col['id']},
                    'backgroundColor': '#e0e0e0',
                    'color': '#757575',
                    'fontStyle': 'italic'
                })
                if col['id'] not in ['AddToChat', 'AddToExplain']:
                    style_conditions.append({
                        'if': {'column_id': col['id']},
                        'cursor': 'default'
                    })
        
        severity_styles = calculate_severity_ranking_and_styles(metrics_data)
        style_conditions.extend(severity_styles)
        
        return style_conditions
    
    # Create the actual table component - EXACT COPY from original
    table = dash_table.DataTable(
        id='metrics-table',
        active_cell=None,
        columns=columns,
        data=metrics_data,
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell={
            'padding': '10px',
            'textAlign': 'center',
            'cursor': 'pointer'
        },
        style_header={
            'backgroundColor': '#614385',
            'color': 'white',
            'fontWeight': 'bold',
            'padding': '12px'
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f5f5f5'},
            {'if': {'column_id': 'Attribute'}, 'cursor': 'pointer', 'color': '#614385', 'fontWeight': '600', 'textDecoration': 'underline'},
            {'if': {'column_id': 'AddToChat'}, 'cursor': 'pointer', 'color': '#007bff', 'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'center', 'verticalAlign': 'middle', 'backgroundColor': '#f8f9fa', 'border': '1px solid #007bff', 'borderRadius': '4px', 'padding': '4px', 'margin': '0 auto'},
            {'if': {'column_id': 'AddToExplain'}, 'cursor': 'pointer', 'color': '#28a745', 'fontWeight': 'bold', 'fontSize': '18px', 'textAlign': 'center', 'verticalAlign': 'middle', 'backgroundColor': '#f8f9fa', 'border': '1px solid #28a745', 'borderRadius': '4px', 'padding': '4px', 'margin': '0 auto'},
            {'if': {'column_id': ['PrimaryTargetRelevance', 'SecondaryTargetRelevance', 'RelevanceDelta']}, 'backgroundColor': '#f0f8ff', 'borderLeft': '1px solid #d9edf7', 'borderRight': '1px solid #d9edf7', 'textAlign': 'center'},
            {'if': {'column_id': 'TargetRelevance'}, 'textAlign': 'center', 'fontWeight': '500'},
            {'if': {'filter_query': '{TargetRelevance} = "High"', 'column_id': 'TargetRelevance'}, 'color': '#d32f2f', 'fontWeight': 'bold'},
            {'if': {'filter_query': '{TargetRelevance} = "Target"', 'column_id': 'TargetRelevance'}, 'color': '#614385', 'fontWeight': 'bold'}
        ] + create_dynamic_style_conditions(columns, metrics_data),
        cell_selectable=True,
        selected_cells=[]
    )
    
    # Create table content with severity explanation - EXACT COPY from original
    table_content = html.Div([
        html.Div([
            html.P([
                "🚨 ",
                html.Strong("Drift Severity Indicators: "),
                "Red highlighted cells show the ",
                html.Strong("top 3 most severe values for each metric type", style={"color": "#d32f2f"}),
                ". Deeper red = higher rank within each metric."
            ], className="mb-2 small text-muted", style={"fontSize": "0.9rem"})
        ], className="px-2"),
        html.Div([table], style={"maxHeight": "500px", "overflowY": "auto"}, className="p-2")
    ], style={"padding": "15px"})
    
    # Create Statistical Drift Detection card - EXACT COPY from original
    metrics_card = dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.Div(
                    html.H4("Statistical Drift Detection", className="mb-0"),
                    style={"display": "inline-block", "width": "80%"}
                ),
                html.Div(
                    html.I(id="drift-table-toggle", className="fa fa-chevron-up",
                          style={"cursor": "pointer", "fontSize": "20px"}),
                    style={"display": "inline-block", "width": "20%", "textAlign": "right"}
                )
            ], style={"display": "flex", "alignItems": "center"}, className="d-flex justify-content-between")
        ),
        dbc.Collapse(
            dbc.CardBody(table_content),
            id="drift-table-collapse",
            is_open=True
        )
    ], className="mb-3", style={"margin": "10px 0"})
    
    # Create Target Distribution Analysis section - EXACT COPY from original
    components = [metrics_card]
    
    # === Enhanced Target Distribution Analysis ===
    if global_vars.target_attribute:
        # 1. directly generate target distribution chart content
        target_chart_content = generate_target_distribution_chart_content()
        
        # 2. directly generate conditional analysis content  
        conditional_content = generate_conditional_analysis_content(metrics_data)
        
        target_dist_content = html.Div([
            # change to vertical arrangement - Target Attribute Distribution on top
            html.Div([
                html.H4("Target Attribute Distribution", 
                       className="mb-4", 
                       style={"textAlign": "center", "color": "#495057", "fontWeight": "bold", "marginBottom": "25px"}),
                # create same container structure as Home page, ensure callback works
                html.Div(
                    target_chart_content,
                    id="target-distribution-chart-container", 
                    style={"height": "400px", "marginBottom": "40px"}
                )
            ], style={"marginBottom": "50px", "padding": "15px 20px"}),
            
            # separator
            html.Hr(style={"margin": "40px 0", "borderColor": "#dee2e6", "borderWidth": "1px"}),
            
            # Conditional Distribution Analysis below
            html.Div([
                html.H4("Conditional Distribution Analysis", 
                       className="mb-4", 
                       style={"textAlign": "center", "color": "#495057", "fontWeight": "bold", "marginBottom": "25px"}),
                html.Div(
                    conditional_content,  # prerendered content, no callback
                    style={"marginTop": "20px"}
                )
            ], style={"padding": "15px 20px", "paddingTop": "25px"})
        ], style={"padding": "25px"})
        
        # create target distribution card with unique IDs
        unique_id = int(time.time() * 1000)
        target_dist_card = dbc.Card([
            dbc.CardHeader(
                html.Div([
                    html.Div(
                        html.H4("Overall Target Distribution Analysis", className="mb-0"),
                        style={"display": "inline-block", "width": "80%"}
                    ),
                    html.Div(
                        html.I(id={"type": "chat-target-dist-toggle", "index": unique_id}, 
                              className="fa fa-chevron-up", 
                              style={"cursor": "pointer", "fontSize": "20px"}),
                        style={"display": "inline-block", "width": "20%", "textAlign": "right"}
                    )
                ], style={"display": "flex", "alignItems": "center"})
            ),
            dbc.Collapse(
                dbc.CardBody(target_dist_content),
                id={"type": "chat-target-dist-collapse", "index": unique_id},
                is_open=True
            )
        ], className="mb-3", style={"margin": "10px 0"})
        
        # Add store for collapse state
        target_dist_store = dcc.Store(id={"type": "chat-target-dist-expanded", "index": unique_id}, data=True)
        
        components.extend([target_dist_card, target_dist_store])
    
    return html.Div(components)


def generate_target_distribution_chart_content():
    """
    Generate target distribution chart content for chat interface"""
    from UI.shared.components.distribution_analysis_utils_impl import get_distribution_data, create_distribution_chart_component
    
    try:
        # get distribution data
        primary_dist, secondary_dist, column_type, attribute_name = get_distribution_data(stage="detect")
        
        if not primary_dist and not secondary_dist:
            return html.Div(
                "No distribution data available", 
                style={"textAlign": "center", "marginTop": "100px", "color": "#666"}
            )
        
        # directly generate chart component, including buttons
        chart_component = create_distribution_chart_component(
            primary_dist, secondary_dist, column_type, attribute_name,
            include_button=True, stage="chat"  # use new stage identifier
        )
        
        return chart_component
        
    except Exception as e:
        print(f"[CHAT TARGET DIST] Error generating chart: {str(e)}")
        return html.Div(
            f"Error generating chart: {str(e)}", 
            style={"textAlign": "center", "marginTop": "100px", "color": "#d9534f"}
        )


def generate_conditional_analysis_content(metrics_data):
    """
    Generate complete conditional analysis content with interactive dropdowns and chart
    Uses same IDs as Home page to reuse existing callbacks
    """
    try:
        # 1. get target values
        target_values = get_target_values_for_chat()
        
        # 2. get top attributes  
        from UI.pages.components.explain_utils import rank_attributes
        top_attributes = rank_attributes(metrics_data, k=10)
        
        # 3. Import button utility function
        from UI.utils.button_utils import create_dual_add_buttons
        
        # 4. Create dropdowns with proper IDs (matching Home page for callback compatibility)
        target_value_dropdown = dcc.Dropdown(
            id="detect-target-value-dropdown",  # ✅ 使用与Home页面相同的ID
            options=[{"label": str(val), "value": str(val)} for val in target_values],
            placeholder="Select target value...",
            className="mb-2"
        )
        
        compare_attr_dropdown = dcc.Dropdown(
            id="detect-compare-attr-dropdown",  # ✅ 使用与Home页面相同的ID
            options=[{"label": attr, "value": attr} for attr in top_attributes],
            placeholder="Select attribute...",
            className="mb-2"
        )
        
        return html.Div([
            # Hidden components required by callbacks (matching Home page structure)
            # Stage is now managed through proper Dash store synchronization
            dcc.Store(id="detect-top-k-attrs", data=[]),  # ✅ 添加detect-top-k-attrs Store
            
            # Dropdown controls
            dbc.Row([
                dbc.Col([
                    html.Label("Target Value", className="mb-1"),
                    target_value_dropdown
                ], width=6),
                
                dbc.Col([
                    html.Label("Compare With", className="mb-1"),
                    compare_attr_dropdown
                ], width=6),
            ], className="mb-3"),
            
            # Help text
            html.Div(
                html.I(
                    "Shows distribution of an attribute when conditioned on a specific target value",
                    className="text-muted small",
                    style={"fontSize": "0.85rem"}
                ),
                className="mb-3"
            ),
            
            # Chart container (matching Home page structure)
            dcc.Loading(
                id="detect-conditional-chart-loading",
                type="circle",
                children=html.Div(
                    id="detect-conditional-chart-container",  # ✅ 与Home页面相同的ID
                    children=[],
                    style={"height": "280px", "marginBottom": "15px"}
                )
            ),
            
            # Add to Chat/Explain buttons (using Home page's button IDs)
            html.Div([
                create_dual_add_buttons(
                    feature_name="conditional distribution",
                    chat_button_id="add-cond-dist-to-chat",     # ✅ 与Home页面相同
                    explain_button_id="add-cond-dist-to-explain", # ✅ 与Home页面相同
                    chat_disabled=True,      # 初始禁用，选择后启用
                    explain_disabled=True,
                    chat_aria_disabled="true",
                    explain_aria_disabled="true"
                )
            ], style={"marginTop": "15px"})
        ])
        
    except Exception as e:
        print(f"[CHAT CONDITIONAL] Error generating content: {str(e)}")
        return html.Div(
            f"Error generating conditional analysis: {str(e)}", 
            style={"textAlign": "center", "marginTop": "40px", "color": "#d9534f"}
        )


def get_target_values_for_chat():
    """
    get unique values of target attribute for dropdown
    """
    try:
        if not hasattr(global_vars, 'target_attribute') or not global_vars.target_attribute:
            return []
        
        if hasattr(global_vars, 'df') and global_vars.df is not None:
            target_col = global_vars.target_attribute
            if target_col in global_vars.df.columns:
                unique_values = global_vars.df[target_col].dropna().unique()
                return sorted([str(val) for val in unique_values])
        
        return []
        
    except Exception as e:
        print(f"[CHAT TARGET VALUES] Error: {str(e)}")
        return []


# Chat collapse/expand callbacks for interactive cards

# Chat Detect Stage collapse/expand toggle
@app.callback(
    [Output({"type": "chat-detect-collapse", "index": MATCH}, "is_open"),
     Output({"type": "chat-detect-toggle", "index": MATCH}, "className")],
    [Input({"type": "chat-detect-toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "chat-detect-collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_chat_detect_collapse(n_clicks, is_open):
    """Toggle the collapse/expand state of chat detect stage card"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fa fa-chevron-up" if new_state else "fa fa-chevron-down"
        return new_state, icon_class
    return dash.no_update, dash.no_update


# Chat Target Distribution Analysis collapse/expand toggle
@app.callback(
    [Output({"type": "chat-target-dist-collapse", "index": MATCH}, "is_open"),
     Output({"type": "chat-target-dist-toggle", "index": MATCH}, "className")],
    [Input({"type": "chat-target-dist-toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "chat-target-dist-collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_chat_target_dist_collapse(n_clicks, is_open):
    """Toggle the collapse/expand state of chat target distribution analysis card"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fa fa-chevron-up" if new_state else "fa fa-chevron-down"
        return new_state, icon_class
    return dash.no_update, dash.no_update


@app.callback(
    [Output({"type": "chat-explain-collapse", "index": MATCH}, "is_open"),
     Output({"type": "chat-explain-toggle", "index": MATCH}, "className")],
    [Input({"type": "chat-explain-toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "chat-explain-collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_chat_explain_collapse(n_clicks, is_open):
    """Toggle the collapse/expand state of chat explain stage card"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fa fa-chevron-up" if new_state else "fa fa-chevron-down"
        return new_state, icon_class
    return dash.no_update, dash.no_update


# ===== START EXPLAIN ANALYSIS CALLBACK =====

@app.callback(
    [Output("query-area", "children", allow_duplicate=True),
     Output("analysis-modal", "is_open", allow_duplicate=True),
     Output("analysis-modal-body", "children", allow_duplicate=True)],
    [Input({"type": "chat-start-explain-btn", "index": ALL}, "n_clicks")],
    [State("query-area", "children"),
     State("explain-context-data", "data")],  # single-instance: one store exists
    prevent_initial_call=True
)
def handle_start_explain_analysis(n_clicks_list, query_records, context_data):
    """
    Handle Start Explain Analysis button click - generate Analysis Panel.
    This creates the comprehensive analysis based on filtered context items.
    """
    # Check if any button was clicked
    print(f"🔬 [STEP2_BUTTON] === STEP 2 ANALYSIS BUTTON CLICKED ===")
    print(f"🔬 [STEP2_BUTTON] Button clicks received: {n_clicks_list}")
    print(f"🔬 [STEP2_BUTTON] Current query_records length: {len(query_records) if query_records else 0}")
    
    if not n_clicks_list or not any(n_clicks_list):
        print(f"[START EXPLAIN ANALYSIS] No button clicks, preventing update")
        raise PreventUpdate
    
    import time
    from UI.functions.global_vars import global_vars
    
    # ✅ CRITICAL: Check if we're in explain stage - prevent accidental triggering
    current_stage = getattr(global_vars, 'current_stage', 'detect')
    if current_stage != "explain":
        print(f"🚫 [STEP2_BUTTON] Not in explain stage (current: {current_stage}), preventing analysis")
        raise PreventUpdate
    
    # Initialize query_records if None
    if query_records is None:
        query_records = []
    
    try:
        print(f"[START EXPLAIN ANALYSIS] 🚀 Starting comprehensive analysis generation...")
        print(f"[START EXPLAIN ANALYSIS] Context data length: {len(context_data) if context_data else 0}")
        
        # Check if we have context data
        if not context_data:
            print(f"[START EXPLAIN ANALYSIS] ⚠️ No context data available")
            
            # Create warning bubble
            warning_bubble = html.Div([
                dbc.Alert([
                    html.H5("⚠️ No Context Data Available", className="alert-heading"),
                    html.P("Unable to find context data for analysis. Please try refreshing the context items."),
                    html.Hr(),
                    html.P("The analysis requires valid context data to proceed.", className="mb-0")
                ], color="warning")
            ], className="llm-msg")
            query_records.append(warning_bubble)
            
            sorted_records = sort_chat_messages(query_records)
            return (sorted_records, False, "")
        
        print(f"[START EXPLAIN ANALYSIS] Found {len(context_data)} context items for analysis")
        
        # Get explain context for strategy analysis
        from UI.state_connector import get_explain_context
        explain_context = get_explain_context(stage=global_vars.current_stage)
        
        # Generate comprehensive analysis using existing logic
        from UI.pages.components.gpt_severity_analyzer import gpt_severity_analyzer
        from UI.pages.components.explain_component import create_comprehensive_analysis_panel
        
        print("[START EXPLAIN ANALYSIS] Calling GPT for comprehensive analysis...")
        strategy_analysis = gpt_severity_analyzer.analyze_context_severity(context_data)
        
        # Create comprehensive analysis panel
        analysis_component = create_comprehensive_analysis_panel(context_data, strategy_analysis)
        
        # Create unique ID for this analysis instance
        analysis_id = f"chat-analysis-{int(time.time() * 1000)}"
        
        # Option 1: Add to chat as a new bubble
        header = dbc.CardHeader([
            html.Div([
                html.Div([
                    html.I(className="fas fa-chart-line me-2"),
                    html.Strong("Analysis Results - Step 2")
                ], style={"display": "inline-block"}),
                html.Div([
                    html.I(id={"type": "chat-analysis-toggle", "index": analysis_id}, 
                           className="fa fa-chevron-up", 
                           style={"cursor": "pointer", "fontSize": "18px"})
                ], style={"display": "inline-block", "float": "right"})
            ])
        ])
        
        collapse = dbc.Collapse(
            dbc.CardBody(analysis_component, style={"padding": "15px"}),
            id={"type": "chat-analysis-collapse", "index": analysis_id},
            is_open=True
        )
        
        bubble_content = dbc.Card([header, collapse], className="mb-2")
        bubble = create_timestamped_message(
            bubble_content,
            "llm-msg chat-table-container"  
        )
        # Assign unique ID to bubble for future reference
        if hasattr(bubble, 'id') and isinstance(bubble.id, dict):
            old_id = bubble.id.copy()
            bubble.id.update({"original_type": "chat-analysis-bubble", "index": analysis_id})
            print(f"🔬 [STEP2_BUTTON] Updated bubble ID: {old_id} → {bubble.id}")
        else:
            print(f"🔬 [STEP2_BUTTON] ⚠️ Bubble has no ID or wrong format!")

        query_records.append(bubble)
        
        print(f"[START EXPLAIN ANALYSIS] ✅ Added comprehensive analysis to chat")
        
        # Option 2: Also show in modal for better visibility
        sorted_records = sort_chat_messages(query_records)
        return (sorted_records, False, "")
        
    except Exception as e:
        print(f"[START EXPLAIN ANALYSIS] ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_bubble = create_timestamped_message([
            dbc.Alert([
                html.H5("❌ Error Generating Analysis", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.Hr(),
                html.P("Please try again or check the console for details.", className="mb-0")
            ], color="danger")
        ], "llm-msg")
        query_records.append(error_bubble)
        
        sorted_records = sort_chat_messages(query_records)
        return (sorted_records, False, "")


# Modal close callback
@app.callback(
    Output("analysis-modal", "is_open", allow_duplicate=True),
    [Input("analysis-modal-close", "n_clicks")],
    [State("analysis-modal", "is_open")],
    prevent_initial_call=True
)
def close_analysis_modal(n_clicks, is_open):
    """Close the analysis modal when close button is clicked."""
    if n_clicks:
        return False
    return is_open


# Context Items collapse/expand toggle
@app.callback(
    [Output({"type": "chat-context-collapse", "index": MATCH}, "is_open"),
     Output({"type": "chat-context-toggle", "index": MATCH}, "className")],
    [Input({"type": "chat-context-toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "chat-context-collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_chat_context_collapse(n_clicks, is_open):
    """Toggle the collapse/expand state of chat context items card"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fa fa-chevron-up" if new_state else "fa fa-chevron-down"
        return new_state, icon_class
    return dash.no_update, dash.no_update


# Analysis Results collapse/expand toggle
@app.callback(
    [Output({"type": "chat-analysis-collapse", "index": MATCH}, "is_open"),
     Output({"type": "chat-analysis-toggle", "index": MATCH}, "className")],
    [Input({"type": "chat-analysis-toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "chat-analysis-collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_chat_analysis_collapse(n_clicks, is_open):
    """Toggle the collapse/expand state of chat analysis results card"""
    if n_clicks:
        new_state = not is_open
        icon_class = "fa fa-chevron-up" if new_state else "fa fa-chevron-down"
        return new_state, icon_class
    return dash.no_update, dash.no_update


# Adapt cards collapse/expand toggle
@app.callback(
    [Output({"type": "chat-adapt-collapse", "index": MATCH}, "is_open"),
     Output({"type": "chat-adapt-toggle", "index": MATCH}, "className")],
    [Input({"type": "chat-adapt-toggle", "index": MATCH}, "n_clicks")],
    [State({"type": "chat-adapt-collapse", "index": MATCH}, "is_open")],
    prevent_initial_call=True
)
def toggle_chat_adapt_collapse(n_clicks, is_open):
    """Toggle the collapse/expand state of adapt stage cards."""
    if n_clicks:
        new_state = not is_open
        icon_class = "fa fa-chevron-up" if new_state else "fa fa-chevron-down"
        return new_state, icon_class
    return dash.no_update, dash.no_update


# Simple stage synchronization
@app.callback(
    Output('current-stage', 'children', allow_duplicate=True),
    [Input('chat-update-trigger', 'data')],
    prevent_initial_call=True
)
def sync_stage_to_store(trigger):
    """Keep hidden Div children in sync with global_vars.current_stage (single-instance mirror)."""
    if trigger and hasattr(global_vars, 'current_stage'):
        # Sync agent stage too
        if hasattr(global_vars, 'agent') and global_vars.agent:
            global_vars.agent.current_stage = global_vars.current_stage
        return global_vars.current_stage
    return dash.no_update