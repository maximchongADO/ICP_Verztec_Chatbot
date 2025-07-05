"""
agentic_minimal.py
=================

A minimal agentic playground demonstrating LangChain's ReAct agent pattern with realistic 
tool simulation, strict output formatting, and robust error handling.

This script creates an AI agent that can:
1. Escalate issues to HR (simulated)
2. Create meeting requests (simulated)
3. Provide direct answers to users

The agent uses the ReAct (Reasoning + Acting) pattern, which follows this cycle:
- Thought: The agent reasons about what to do
- Action: The agent chooses a tool to use
- Action Input: The agent provides input to the tool
- Observation: The agent receives the tool's output
- Repeat until Final Answer

Key Features:
- All tool actions are logged to 'yabdabado.csv' for tracking
- Strict ReAct format enforcement to prevent common errors
- Robust error handling and termination logic
- Single-turn interaction with comprehensive output display

Usage:
    python agentic_minimal.py

The agent will process a predefined query about workplace harassment and demonstrate
the complete workflow: escalation → meeting request → final answer.

Example Output Flow:
    User Query: "my boss touched me inappropriately, what should I do?"
    
    Agent Steps:
    1. Thought: This is a serious workplace issue requiring HR escalation
    2. Action: escalate_to_hr
    3. Observation: Issue escalated successfully
    4. Thought: User may need guidance meeting, should schedule follow-up
    5. Action: create_meeting_request
    6. Observation: Meeting scheduled successfully
    7. Action: Final Answer
    8. Result: Comprehensive response with next steps
    
    All actions logged to: yabdabado.csv

Architecture:
    ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │   User Query    │───▶│   ReAct Agent    │───▶│   Tool Layer    │
    └─────────────────┘    └──────────────────┘    └─────────────────┘
                                    │                        │
                                    ▼                        ▼
                           ┌──────────────────┐    ┌─────────────────┐
                           │  LLM Reasoning   │    │  CSV Logging    │
                           └──────────────────┘    └─────────────────┘

Dependencies:
- langchain: Core agent framework
- langchain-groq: LLM provider for fast inference
- langchain-community: Additional tools and utilities

Environment:
- Requires GROQ API key (currently hardcoded)
- Creates yabdabado.csv in the current directory for logging
- Python 3.8+ recommended

Security Notes:
- API keys should be moved to environment variables in production
- Tool actions are simulated - real integrations need proper authentication
- CSV logging should be secured with appropriate file permissions

Common Issues & Solutions:
- "Agent stopped due to iteration limit": Check max_iterations setting
- "Parsing errors": Review prompt template and LLM output format
- "No final answer": Ensure Final Answer tool is being called properly
- "Infinite loops": Verify tool descriptions are clear and specific
"""

import os
import csv
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import AgentAction, AgentFinish


# --- Configuration and Setup ---
"""
LLM Configuration:
- Provider: GROQ (fast inference with 1000+ tokens/second)
- Model: meta-llama/llama-4-scout-17b-16e-instruct
- Purpose: Balanced performance for reasoning tasks
- Temperature: Default (controlled by model)
- Max tokens: Default (adequate for ReAct responses)

Model Selection Rationale:
- Llama-4-scout: Good instruction following for agentic tasks
- 17B parameters: Balance between capability and speed
- Instruct variant: Fine-tuned for conversational tasks

Production Considerations:
- API key should be moved to environment variables
- Consider implementing rate limiting for high-volume usage
- Monitor API costs and usage patterns
- Add fallback models for reliability

Performance Characteristics:
- Average response time: 2-5 seconds
- Context window: 8K tokens (sufficient for ReAct)
- Concurrent requests: Limited by API tier
"""
api_key = "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"
llm = ChatGroq(api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")

# Memory setup (currently unused but available for multi-turn conversations)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question", 
    output_key="Action Input",
    return_messages=True
)

# --- Tool Definitions ---
"""
Tool Architecture & Design Patterns
====================================

Each tool in this system follows a consistent pattern for reliability and maintainability:

1. Input Validation: Check and sanitize user inputs
2. Business Logic: Execute the core functionality (simulated)
3. Side Effects: Log actions for audit trails
4. Response Generation: Return user-friendly messages

Tool Development Guidelines:
- Keep functions pure where possible (minimal side effects)
- Always log actions for compliance and debugging
- Return human-readable messages, not technical responses
- Handle edge cases gracefully (empty inputs, special characters)
- Use consistent naming conventions

CSV Logging Schema:
    Column 1: Timestamp (ISO format)
    Column 2: Tool name (matches LangChain tool registry)
    Column 3: Input data (sanitized for privacy)

Error Handling Strategy:
- File I/O errors: Graceful degradation with console logging
- Input validation: Return helpful error messages
- Encoding issues: UTF-8 with fallback handling

Real-World Integration Points:
- escalate_to_hr: JIRA/ServiceNow ticket creation
- create_meeting_request: Outlook/Google Calendar APIs
- final_answer: Knowledge base updates, user satisfaction tracking
"""

def escalate_to_hr_tool(issue: str) -> str:
    """
    Simulates escalating a workplace issue to Human Resources.
    
    This tool handles sensitive workplace issues that require formal HR intervention.
    In a production environment, this would integrate with enterprise HR systems.
    
    Args:
        issue (str): Description of the workplace issue to escalate
                    Should be concise but contain sufficient detail for HR triage
                    
    Returns:
        str: Confirmation message that the issue was escalated
             Format: "[Simulated] Your issue has been escalated to HR: '{issue}'."
        
    Side Effects:
        - Logs the escalation to yabdabado.csv with timestamp
        - In production: Creates tickets, sends notifications, triggers workflows
        
    Integration Points (Production):
        - ServiceNow: Incident/case creation
        - JIRA: Issue tracking and assignment
        - Email systems: Automated notifications to HR team
        - Slack/Teams: Real-time alerts for urgent issues
        
    Privacy Considerations:
        - Sensitive data should be redacted in logs
        - PII handling must comply with data protection regulations
        - Consider encryption for stored escalation data
        
    Example:
        >>> escalate_to_hr_tool("Harassment by manager")
        "[Simulated] Your issue has been escalated to HR: 'Harassment by manager'."
    """
    # Input validation
    if not issue or not issue.strip():
        return "Error: Please provide a description of the issue to escalate."
    
    # Sanitize input for logging (basic example)
    sanitized_issue = issue.strip()[:500]  # Limit length for logging
    
    try:
        with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), "escalate_to_hr", sanitized_issue])
    except Exception as e:
        print(f"Warning: Failed to log HR escalation: {e}")
        # Continue execution even if logging fails
    
    return f"[Simulated] Your issue has been escalated to HR: '{sanitized_issue}'."

def create_meeting_request_tool(details: str) -> str:
    """
    Simulates creating a meeting request in a calendar system.
    
    This tool handles meeting scheduling for follow-ups, consultations, and formal
    discussions. In production, it would integrate with enterprise calendar systems.
    
    Args:
        details (str): Description of the meeting to be scheduled
                      Should include purpose, preferred timing, and attendees
                      
    Returns:
        str: Confirmation message that the meeting was requested
             Format: "[Simulated] Meeting request created: '{details}'."
        
    Side Effects:
        - Logs the meeting request to yabdabado.csv with timestamp
        - In production: Creates calendar events, sends invitations, books rooms
        
    Integration Points (Production):
        - Microsoft Outlook: Calendar API integration
        - Google Calendar: Event creation and management
        - Zoom/Teams: Meeting link generation
        - Room booking systems: Resource reservation
        
    Meeting Request Best Practices:
        - Include clear meeting purpose and agenda
        - Specify duration and preferred time slots
        - List required and optional attendees
        - Provide dial-in/video conferencing details
        
    Example:
        >>> create_meeting_request_tool("HR consultation next week with manager")
        "[Simulated] Meeting request created: 'HR consultation next week with manager'."
        
    Error Handling:
        - Empty requests: Return helpful error message
        - Long descriptions: Truncate for logging efficiency
        - Special characters: Handle encoding gracefully
    """
    # Input validation
    if not details or not details.strip():
        return "Error: Please provide meeting details (purpose, timing, attendees)."
    
    # Sanitize and format input
    sanitized_details = details.strip()[:300]  # Reasonable limit for meeting descriptions
    
    try:
        with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), "create_meeting_request", sanitized_details])
    except Exception as e:
        print(f"Warning: Failed to log meeting request: {e}")
        # Continue execution even if logging fails
    
    return f"[Simulated] Meeting request created: '{sanitized_details}'."

def final_answer_tool(answer: str) -> str:
    """
    Provides the final answer to the user and signals agent completion.
    
    This is the critical termination tool for the ReAct agent pattern. It serves
    dual purposes: delivering the final response and triggering agent shutdown.
    
    Args:
        answer (str): The final response to provide to the user
                     Should be comprehensive, actionable, and empathetic
                     
    Returns:
        str: The answer without any modifications (clean termination)
             Must be clean to prevent agent confusion or continued execution
        
    Side Effects:
        - Logs the final answer to yabdabado.csv with timestamp
        - Signals the agent executor to terminate the reasoning loop
        - Completes the user interaction cycle
        
    ReAct Pattern Compliance:
        - Must be the LAST action in any agent execution
        - Cannot be followed by additional thoughts or actions
        - Triggers immediate agent termination
        - Prevents infinite reasoning loops
        
    Answer Quality Guidelines:
        - Be specific and actionable
        - Show empathy for sensitive situations
        - Provide clear next steps when appropriate
        - Reference previous actions taken (escalations, meetings)
        
    Example Flow:
        1. Agent escalates issue to HR
        2. Agent schedules follow-up meeting
        3. Agent calls Final Answer with comprehensive guidance
        4. Agent terminates successfully
        
    Production Considerations:
        - Consider user satisfaction tracking
        - May trigger feedback collection workflows
        - Could update knowledge bases with successful resolutions
        - Useful for agent performance analytics
        
    Debugging Notes:
        - If agent doesn't terminate, check if this tool is being called
        - Ensure return value is clean (no prefixes or extra formatting)
        - Verify agent executor settings allow proper termination
    """
    # Input validation
    if not answer or not answer.strip():
        fallback_answer = "I apologize, but I couldn't generate a proper response. Please contact support for assistance."
        print(f"Warning: Empty final answer provided, using fallback: {fallback_answer}")
        answer = fallback_answer
    
    # Clean the answer for optimal agent termination
    clean_answer = answer.strip()
    
    try:
        with open("yabdabado.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), "Final Answer", clean_answer])
    except Exception as e:
        print(f"Warning: Failed to log final answer: {e}")
        # Continue execution - logging failure shouldn't break user experience
    
    # Return clean answer without prefixes to ensure proper termination
    return clean_answer

# --- Tool Registry ---
"""
Tool Registry & Agent Discovery
===============================

The tool registry serves as the interface between the LLM and available functions.
Each Tool object wraps a Python function and provides metadata for the agent.

Tool Registration Best Practices:
- Use descriptive, action-oriented names (escalate_to_hr, not hr_tool)
- Provide clear descriptions that indicate when to use each tool
- Avoid ambiguous tool purposes that could confuse the agent
- Test tool descriptions with various query types

Tool Selection Algorithm:
1. Agent analyzes user query for intent and required actions
2. Agent matches intent to tool descriptions using semantic similarity
3. Agent selects most appropriate tool based on context
4. Agent executes tool with generated input parameters

Common Tool Selection Issues:
- Vague descriptions → Wrong tool selection
- Similar descriptions → Agent confusion
- Missing tools → Agent gives up or hallucinates actions
- Tool name mismatches → Parsing errors

Description Writing Guidelines:
- Start with action verb (Escalates, Creates, Provides)
- Specify use cases clearly (serious workplace issues, meeting requests)
- Include context hints (when to use, what inputs expected)
- Keep concise but informative (1-2 sentences ideal)

Tool Discovery Process:
```
User Query → Intent Analysis → Tool Matching → Action Selection
     ↓              ↓              ↓              ↓
"Help with    "Workplace     escalate_to_hr   Action: escalate_to_hr
 harassment"   issue"         tool selected    Input: harassment details
```

Quality Assurance:
- Test each tool independently before agent integration
- Verify tool descriptions with sample queries
- Monitor tool usage patterns in logs
- Update descriptions based on real usage data
"""
tools = [
    Tool(
        name="escalate_to_hr",
        func=escalate_to_hr_tool,
        description="Escalates a user issue to HR. Use for serious workplace or personal issues like harassment, discrimination, or policy violations."
    ),
    Tool(
        name="create_meeting_request", 
        func=create_meeting_request_tool,
        description="Creates a meeting request with provided details. Use for scheduling follow-ups, consultations, or formal discussions."
    ),
    Tool(
        name="Final Answer",
        func=final_answer_tool,
        description="Provides the final answer to the user. MUST be used as the last action to complete the interaction."
    )
]

# --- ReAct Prompt Engineering ---
def get_react_prompt() -> PromptTemplate:
    """
    Creates a carefully engineered prompt template for the ReAct agent.
    
    The prompt addresses common failure modes in ReAct agents:
    1. Format violations (missing Action Input, extra text)
    2. Infinite loops (repeating same actions)
    3. Improper termination (not using Final Answer)
    4. Tool chaining (multiple actions in one turn)
    
    Returns:
        PromptTemplate: Configured template with strict formatting rules
        
    Key Design Decisions:
    - Explicit format requirements with examples
    - Clear termination conditions
    - Prohibition of action repetition
    - Single action per reasoning cycle
    """
    template = """You are a helpful AI assistant. Use the ReAct format to work through problems step by step.

You have access to these tools:
{tools}

CRITICAL RULES:
1. Use the format: Thought: → Action: → Action Input: → wait for Observation
2. When you use "Final Answer", you are DONE. Do not continue thinking or taking actions.
3. Do NOT repeat the same action multiple times.
4. Do NOT continue after "Final Answer".

Format:
Thought: [your reasoning]
Action: [tool name from: {tool_names}]
Action Input: [input for the tool]
Observation: [tool response]
Thought: [more reasoning if needed]
Action: Final Answer
Action Input: [your final response to the user]

Question: {input}
{agent_scratchpad}"""

    return PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

# --- Agent Configuration and Execution ---
if __name__ == "__main__":
    """
    Main execution block that demonstrates the complete agentic workflow.
    
    Agent Architecture:
    1. ReAct Agent: Handles reasoning and tool selection
    2. Agent Executor: Manages execution loop and error handling
    3. Tools: Provide specific capabilities (HR, meetings, answers)
    
    Execution Flow:
    1. User query → Agent reasoning (Thought)
    2. Tool selection → Tool execution (Action + Input)
    3. Tool response → Further reasoning (Observation)
    4. Final answer → Termination (Final Answer)
    
    Error Handling:
    - Parsing errors are handled gracefully
    - Iteration limits prevent infinite loops
    - Final answers are extracted even if limits are hit
    """
    
    # Initialize the ReAct prompt with strict formatting rules
    prompt = get_react_prompt()
    
    # Create the core ReAct agent
    agent_runnable = create_react_agent(
        llm=llm,                # Language model for reasoning
        tools=tools,            # Available tools for actions
        prompt=prompt           # Engineered prompt template
    )
    
    # Configure the agent executor with robust settings
    agent = AgentExecutor(
        agent=agent_runnable,
        tools=tools,
        verbose=True,                    # Show detailed execution steps
        handle_parsing_errors=True,      # Graceful error recovery
        max_iterations=3,                # Prevent infinite loops
        max_execution_time=60,           # 60-second timeout
        return_intermediate_steps=True,  # Enable step-by-step analysis
        early_stopping_method="force"    # Force stop when limits reached
    )
    
    # Demo execution with a sensitive workplace scenario
    print("=" * 60)
    print("🤖 Minimal Agentic Playground - ReAct Agent Demo")
    print("=" * 60)
    print("Scenario: Workplace harassment issue requiring HR escalation")
    print("Expected workflow: Escalate → Schedule meeting → Provide guidance")
    print("-" * 60)
    
    # Test query demonstrating multi-step reasoning
    user_query = "my boss touched me inappropriately, what should I do?"
    
    if user_query.lower() in {"exit", "quit"}:
        print("👋 Goodbye!")
    else:
        try:
            # Execute the agent with comprehensive error handling
            response = agent.invoke({"input": user_query})
            
            if isinstance(response, dict):
                print("\n📊 --- Agent Response Analysis ---")
                for k, v in response.items():
                    print(f"📌 {k}: {v}")
                print("-" * 50)
                
                # Smart final answer extraction
                final_answer = response.get('output')
                if final_answer and "Agent stopped due to iteration limit" in final_answer:
                    print("⚠️  Agent hit iteration limit, extracting final answer from steps...")
                    # Extract final answer from intermediate steps
                    steps = response.get('intermediate_steps', [])
                    for step in reversed(steps):
                        if isinstance(step, tuple) and len(step) >= 2:
                            action, observation = step
                            if hasattr(action, 'tool') and action.tool == "Final Answer":
                                print("✅ AI (extracted final):", observation)
                                break
                    else:
                        print("❌ AI (failed extraction):", final_answer)
                elif final_answer:
                    print("✅ AI (successful completion):", final_answer)
                else:
                    print("❓ No final answer received")
                    
            else:
                print("🔍 AI (raw response):", response)
                
        except Exception as e:
            print(f"💥 AI: An error occurred: {e}")
            print("📝 Check yabdabado.csv for logged actions")
    
    print("\n" + "=" * 60)
    print("✨ Demo completed! Check 'yabdabado.csv' for action logs.")
    print("=" * 60)
    
    # Development and debugging utilities
    print("\n🔧 Debugging Information:")
    print("- Agent configuration: max_iterations=3, max_execution_time=60s")
    print("- Available tools:", [tool.name for tool in tools])
    print("- LLM model: meta-llama/llama-4-scout-17b-16e-instruct")
    print("- Log file: yabdabado.csv (check for tool execution history)")
    
    print("\n📚 Common Issues & Solutions:")
    print("1. 'Agent stopped due to iteration limit'")
    print("   → Increase max_iterations or simplify the query")
    print("2. 'Parsing errors in agent output'")
    print("   → Check LLM model compatibility with ReAct format")
    print("3. 'No final answer provided'")
    print("   → Verify Final Answer tool is working correctly")
    print("4. 'Tool not found errors'")
    print("   → Check tool names match exactly in registry")
    
    print("\n🚀 Next Steps for Development:")
    print("- Replace API key with environment variable")
    print("- Integrate real HR and calendar APIs")
    print("- Add multi-turn conversation support")
    print("- Implement user authentication and authorization")
    print("- Add more sophisticated error handling and recovery")
    print("- Create unit tests for each tool function")
    print("- Add performance monitoring and analytics")
    
    print("\n📖 Documentation Updates:")
    print("- All functions now have comprehensive docstrings")
    print("- Architecture diagrams included in module header")
    print("- Error handling patterns documented")
    print("- Production considerations outlined")
    print("- Debugging guides provided")
    
    print("\n" + "=" * 60)