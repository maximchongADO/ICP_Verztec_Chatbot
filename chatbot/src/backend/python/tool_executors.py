
"""
Tool Execution Functions for Verztec Chatbot

This module contains all the tool execution functions that were previously 
defined in the main chatbot.py file. This separation improves code organization
and maintainability.

Functions:
- execute_confirmed_tool: Main dispatcher for tool execution
- execute_hr_escalation_tool: Handles HR escalation requests with email notifications
- execute_meeting_scheduling_tool: Handles meeting scheduling requests
- send_hr_escalation_email: Sends email notifications to HR representatives

Email Configuration:
- HR escalations automatically send email notifications to jwwl6424@gmail.com
- For production use, configure proper SMTP credentials in send_hr_escalation_email function
- Currently using simulated email sending (logs email content instead of actual sending)

Note: To enable actual email sending in production:
1. Set up proper SMTP credentials (Gmail, Outlook, or corporate email server)
2. Update sender_email and sender_password in send_hr_escalation_email function
3. Uncomment the actual SMTP sending code
4. Use environment variables for email credentials (never hardcode passwords)
"""

import csv
import json
import logging
import os
import re
import smtplib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables from .env file in the root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env'))

# Set up logging
logger = logging.getLogger(__name__)


def send_hr_escalation_email(escalation_id: str, user_id: str, chat_id: str, 
                           user_query: str, user_description: str = None,
                           hr_email: str = None) -> bool:
    """
    Send an email notification to HR about the escalation.
    
    Args:
        escalation_id (str): Unique escalation ID
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        user_query (str): Original user query
        user_description (str, optional): Additional description from user
        hr_email (str): HR representative email address
        
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    try:
        # Email configuration - Load from environment variables
        # IMPORTANT: Configure these in your .env file!
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_APP_PASSWORD')  # Gmail App Password from .env
        
        # Validate required environment variables
        if not sender_email or not sender_password:
            logger.error("Missing email configuration: SENDER_EMAIL or SENDER_APP_PASSWORD not set in environment")
            logger.error("Please configure these in your .env file:")
            logger.error("- SENDER_EMAIL=your_email@gmail.com")
            logger.error("- SENDER_APP_PASSWORD=your_16_character_app_password")
            return False
        
        # Use HR email from environment or fallback to parameter
        if hr_email is None:
            hr_email = os.getenv('HR_EMAIL', 'jwwl6424@gmail.com')
        
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = hr_email
        message["Subject"] = f"HR Escalation Alert - {escalation_id}"
        
        # Create email body
        issue_summary = user_description if user_description else user_query
        
        email_body = f"""
HR ESCALATION ALERT

A new HR escalation has been submitted and requires your attention.

ESCALATION DETAILS:
• Reference ID: {escalation_id}
• Date & Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
• User ID: {user_id}
• Chat ID: {chat_id}

ORIGINAL QUERY:
{user_query}

{f'''
DETAILED DESCRIPTION:
{user_description}
''' if user_description else ''}

ISSUE SUMMARY:
{issue_summary[:500]}{'...' if len(issue_summary) > 500 else ''}

PRIORITY: NORMAL
STATUS: PENDING

Please log into the HR system to review this escalation and take appropriate action.
The user has been informed that HR will contact them within 24 hours.

For urgent matters, please contact the user directly using the provided User ID.

Best regards,
Verztec AI Assistant
        """
        
        message.attach(MIMEText(email_body, "plain"))
        
        # Actually send the email
        try:
            logger.info(f"Attempting to send email to {hr_email} using SMTP server {smtp_server}:{smtp_port}")
            server = smtplib.SMTP(smtp_server, smtp_port)
            logger.info("SMTP connection established")
            
            server.starttls()
            logger.info("TLS encryption started")
            
            server.login(sender_email, sender_password)
            logger.info(f"Successfully logged in as {sender_email}")
            
            text = message.as_string()
            server.sendmail(sender_email, hr_email, text)
            logger.info("Email sent successfully")
            
            server.quit()
            logger.info("SMTP connection closed")
            
            logger.info(f"Email notification successfully sent to HR ({hr_email}) for escalation {escalation_id}")
            return True
            
        except Exception as smtp_error:
            logger.error(f"SMTP error sending email for {escalation_id}: {str(smtp_error)}")
            logger.error(f"Error type: {type(smtp_error).__name__}")
            logger.error(f"Full error details: {repr(smtp_error)}")
            # Log the email content for debugging
            logger.info(f"Failed to send - Email Content for {escalation_id}:")
            logger.info(f"To: {hr_email}")
            logger.info(f"Subject: HR Escalation Alert - {escalation_id}")
            logger.info(f"Body: {email_body}")
            return False
        
    except Exception as e:
        logger.error(f"Failed to send HR escalation email for {escalation_id}: {str(e)}")
        return False
    
    
    
    
    
    
# Meeting request email sender (modeled after send_hr_escalation_email)
def send_meeting_request_email(
    meeting_request_id: str,
    user_id: str,
    chat_id: str,
    meeting_details: dict,
    user_query: str = None,
    user_description: str = None,
    meeting_email: str = None
) -> bool:
    """
    Send an email notification for a meeting request.
    """
    try:
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        sender_email = os.getenv('SENDER_EMAIL')
        sender_password = os.getenv('SENDER_APP_PASSWORD')
        if not sender_email or not sender_password:
            logger.error("Missing email configuration: SENDER_EMAIL or SENDER_APP_PASSWORD not set in environment")
            return False
        if meeting_email is None:
            meeting_email = os.getenv('MEETING_CONFIRM_EMAIL', 'meetings@verztec.com')
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = meeting_email
        message["Subject"] = f"Meeting Request - {meeting_request_id}"
        email_body = f"""
MEETING REQUEST INITIATED

Meeting ID: {meeting_request_id}
Date & Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
User ID: {user_id}
Chat ID: {chat_id}

Subject: {meeting_details.get('subject', 'Not specified')}
Date/Time: {meeting_details.get('date_time', 'Not specified')}
Duration: {meeting_details.get('duration', 'Not specified')}
Participants: {', '.join(meeting_details.get('participants', []))}
Meeting Type: {meeting_details.get('meeting_type', 'Not specified')}
Location: {meeting_details.get('location', 'Not specified')}
Priority: {meeting_details.get('priority', 'normal')}

Original User Query: {user_query or ''}
Additional Description: {user_description or ''}

This meeting request has been initiated and is awaiting user confirmation.
"""
        message.attach(MIMEText(email_body, "plain"))
        try:
            logger.info(f"Attempting to send meeting email to {meeting_email} using SMTP server {smtp_server}:{smtp_port}")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, meeting_email, message.as_string())
            server.quit()
            logger.info(f"Meeting request email sent to {meeting_email} for meeting {meeting_request_id}")
            return True
        except Exception as smtp_error:
            logger.error(f"SMTP error sending meeting email for {meeting_request_id}: {str(smtp_error)}")
            logger.info(f"Failed to send - Email Content for {meeting_request_id}:")
            logger.info(f"To: {meeting_email}")
            logger.info(f"Subject: Meeting Request - {meeting_request_id}")
            logger.info(f"Body: {email_body}")
            return False
    except Exception as e:
        logger.error(f"Failed to send meeting request email for {meeting_request_id}: {str(e)}")
        return False





def extract_meeting_details(user_query: str) -> Dict[str, Any]:
    """
    Extract meeting details from natural language user query using LLM.
    
    Args:
        user_query (str): User's natural language meeting request
        
    Returns:
        dict: Extracted meeting details with confidence scores
    """
    try:
        # Import the LLM models from chatbot.py when needed
        from langchain_groq import ChatGroq
        from langchain_core.prompts import ChatPromptTemplate
        import json
        import re
        
        # Initialize the LLM (using same config as in chatbot.py)
        api_key = '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'
        extraction_model = ChatGroq(
            api_key=api_key, 
            model="qwen/qwen3-32b",
            temperature=0,
            model_kwargs={
                "top_p": 0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        )
        
        # Get current date and time context
        from datetime import datetime
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%A, %B %d, %Y")
        current_time = current_datetime.strftime("%I:%M %p")
        
        # Create extraction prompt with current date/time context
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meeting detail extraction assistant. Extract meeting details from user input and return ONLY a valid JSON object.

CURRENT CONTEXT:
- Today is: {current_date}
- Current time is: {current_time}

Required JSON format:
{{
    "subject": "string or null",
    "date_time": "string or null", 
    "duration": "string or null",
    "participants": ["array", "of", "strings"],
    "meeting_type": "virtual|in-person|hybrid or null",
    "location": "string or null",
    "priority": "high|normal|low",
    "extraction_confidence": "high|medium|low"
}}

Extraction rules:
- subject: Main topic/purpose of the meeting
- date_time: Convert relative dates to specific dates/times. For example:
  * "tomorrow" = the next day from today
  * "next Monday" = the next Monday from today
  * "this Friday" = the upcoming Friday this week
  * Include both date and time when available (e.g., "Tuesday, July 16, 2025 at 3:00 PM")
- duration: How long the meeting should be
- participants: Names, emails, departments, titles mentioned
- meeting_type: virtual (zoom/teams/online), in-person (conference room/office), or hybrid
- location: Physical location, room names, or virtual platform
- priority: high (urgent/asap/emergency), normal (default), low (when possible/eventually)
- extraction_confidence: high (4+ fields), medium (2-3 fields), low (0-1 fields)

Return ONLY the JSON object. No explanations."""),
            ("human", "Extract meeting details from: {query}")
        ])
        
        # Create and run the extraction chain
        extraction_chain = extraction_prompt | extraction_model
        
        # Get the response
        response = extraction_chain.invoke({
            "query": user_query,
            "current_date": current_date,
            "current_time": current_time
        })
        raw_response = response.content.strip()
        
        logger.info(f"Raw LLM response: {raw_response}")
        
        # Clean up the response to extract JSON
        # Remove any markdown formatting or extra text
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = raw_response[json_start:json_end]
            logger.info(f"Extracted JSON string: {json_str}")
            
            # Additional cleanup for common issues
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            json_str = re.sub(r'\s+', ' ', json_str)  # Replace multiple spaces with single space
            
            extracted_data = json.loads(json_str)
            
            # Add raw query for reference
            extracted_data['raw_query'] = user_query.strip()
            
            # Validate and clean the extracted data
            details = {
                'subject': extracted_data.get('subject'),
                'date_time': extracted_data.get('date_time'),
                'duration': extracted_data.get('duration'),
                'participants': extracted_data.get('participants', []) if isinstance(extracted_data.get('participants'), list) else [],
                'meeting_type': extracted_data.get('meeting_type'),
                'location': extracted_data.get('location'),
                'priority': extracted_data.get('priority', 'normal'),
                'raw_query': user_query.strip(),
                'extraction_confidence': extracted_data.get('extraction_confidence', 'medium')
            }
            
            logger.info(f"LLM extraction successful: {details}")
            return details
            
        else:
            raise ValueError("No valid JSON found in LLM response")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed at position {e.pos}: {str(e)}")
        logger.error(f"Problematic JSON string: {json_str if 'json_str' in locals() else 'Not extracted'}")
        logger.error(f"Full LLM response: {raw_response if 'raw_response' in locals() else 'Not available'}")
        
        # Continue to fallback logic...
    except Exception as e:
        logger.error(f"LLM extraction failed: {str(e)}, falling back to regex")
        
        # Fallback to basic regex extraction if LLM fails
        details = {
            'subject': None,
            'date_time': None,
            'duration': None,
            'participants': [],
            'meeting_type': None,
            'location': None,
            'priority': 'normal',
            'raw_query': user_query.strip(),
            'extraction_confidence': 'low'
        }
        
        query_lower = user_query.lower().strip()
        
        # Basic regex fallbacks for critical fields
        # Extract time mentions
        time_patterns = [
            r'\d{1,2}(?::\d{2})?\s*(?:am|pm)',
            r'(?:at\s+)?\d{1,2}(?::\d{2})?',
            r'tomorrow|today|next week|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                details['date_time'] = match.group(0)
                break
        
        # Extract participant names (simple pattern)
        participant_match = re.search(r'with\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', user_query)
        if participant_match:
            details['participants'] = [participant_match.group(1)]
        
        # Extract room/location
        room_match = re.search(r'(?:room|conference room)\s+([A-Za-z0-9]+)', query_lower)
        if room_match:
            details['location'] = room_match.group(1).upper()
        
        # Basic subject extraction (everything before time/participant mentions)
        subject_match = re.search(r'^(.+?)(?:\s+(?:with|at|tomorrow|today|next|room))', query_lower)
        if subject_match:
            subject = subject_match.group(1).strip()
            if len(subject) > 3:
                details['subject'] = subject.replace('meeting', '').strip().title()
        
        return details






def execute_confirmed_tool(
    tool_identified: str, 
    user_query: str, 
    user_id: str, 
    chat_id: str, 
    user_description: Optional[str] = None,
    global_tools: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the confirmed tool based on the tool identification.
    
    Args:
        tool_identified (str): The identified tool name from the decision layer
        user_query (str): The original user query
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        user_description (str, optional): Additional description provided by user (for HR escalations)
        global_tools (dict, optional): Dictionary of available tools
        
    Returns:
        dict: Response containing the tool execution result with text, images, sources, etc.
    """
    logger.info(f"Executing confirmed tool: {tool_identified} for user {user_id}")
    if user_description:
        logger.info(f"User description provided: {user_description[:100]}...")
    
    try:
        # Default global_tools if not provided
        if global_tools is None:
            global_tools = {
                "raise_to_hr": {
                    "description": "Raise the query to HR — for serious complaints, legal issues, or sensitive workplace matters.",
                    "prompt_style": "You are a compassionate and professional HR liaison assistant.",
                    "response_tone": "supportive_professional"
                },
                "schedule_meeting": {
                    "description": "Set up a meeting — for coordination involving multiple stakeholders or recurring issues.",
                    "prompt_style": "You are an efficient scheduling and coordination assistant.",
                    "response_tone": "organized_efficient"
                }
            }
        
        # Check if the tool exists in our global tools dictionary
        if tool_identified not in global_tools:
            logger.error(f"Tool '{tool_identified}' not found in available tools: {list(global_tools.keys())}")
            return {
                'text': f"Sorry, the requested tool '{tool_identified}' is not available. Please try again or contact support.",
                'images': [],
                'sources': [],
                'tool_used': False,
                'tool_identified': tool_identified,
                'tool_confidence': 'error - tool not found'
            }
        
        # Execute the appropriate tool based on identification
        if tool_identified == "raise_to_hr":
            # Import the dependency functions when needed
            from chatbot import store_hr_escalation, store_chat_log_updated
            return execute_hr_escalation_tool(
                user_query, 
                user_id, 
                chat_id, 
                user_description,
                store_hr_escalation,
                store_chat_log_updated
            )
            
        elif tool_identified == "schedule_meeting":
            # Import the dependency function when needed
            from chatbot import store_chat_log_updated
            return execute_meeting_scheduling_tool(
                user_query, 
                user_id, 
                chat_id,
                user_description,
                store_chat_log_updated
            )
        elif tool_identified == "vacation_check":
            # Import the dependency function when needed
            from chatbot import store_chat_log_updated
            return execute_vacation_check_tool(
                user_query, 
                user_id, 
                chat_id,  # Now pass the additional user input
                store_chat_log_updated
            )
            
        else:
            logger.warning(f"Tool '{tool_identified}' exists but no execution handler implemented")
            return {
                'text': f"The tool '{tool_identified}' is available but not yet implemented. This feature is coming soon!",
                'images': [],
                'sources': [],
                'tool_used': False,
                'tool_identified': tool_identified,
                'tool_confidence': 'error - handler not implemented'
            }
            
    except Exception as e:
        logger.error(f"Error executing tool '{tool_identified}': {str(e)}", exc_info=True)
        return {
            'text': f"An error occurred while executing the {tool_identified} tool. Please try again or contact support.",
            'images': [],
            'sources': [],
            'tool_used': False,
            'tool_identified': tool_identified,
            'tool_confidence': f'error - {str(e)}'
        }





def execute_hr_escalation_tool(
    user_query: str, 
    user_id: str, 
    chat_id: str, 
    user_description: Optional[str] = None,
    store_hr_escalation_func: Optional[callable] = None,
    store_chat_log_updated_func: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Execute HR escalation tool with enhanced logging and response formatting.
    
    Args:
        user_query (str): The original user query that triggered HR escalation
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        user_description (str, optional): Additional description provided by the user
        store_hr_escalation_func (callable, optional): Function to store HR escalation in database
        store_chat_log_updated_func (callable, optional): Function to store chat log
        
    Returns:
        dict: Formatted response for HR escalation
    """
    try:
        # Use user description if provided, otherwise fall back to original query
        primary_issue = user_description if user_description and user_description.strip() else user_query
        
        # Sanitize and structure the input for logging
        sanitized_issue = primary_issue.strip()[:800]  # Limit for better context
        escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{user_id[:8]}"
        
        
        # Enhanced logging with structured data
        logger.info(f"HR Escalation initiated - ID: {escalation_id}, User: {user_id}, Chat: {chat_id}")
        if user_description:
            logger.info(f"User provided detailed description: {user_description[:200]}...")
        
        # Read mailing list from database
        try:
            import mysql.connector
            from mysql.connector import Error
            
            # Database configuration - using same config as userController.js
            db_config = {
                'host': 'localhost',
                'user': 'chatbot_user',
                'password': 'strong_password',
                'database': 'chatbot_db'
            }
            
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            
            # Query mailing list table
            cursor.execute("SELECT id, name, email FROM mailing_list")
            mailing_list_records = cursor.fetchall()
            
            # Extract emails and log all found entries
            email_list = []
            logger.info(f"Found {len(mailing_list_records)} entries in mailing_list table:")
            for record in mailing_list_records:
                record_id, name, email = record
                if email and email.strip():
                    email_list.append(email.strip())
                    logger.info(f"Mailing list entry - ID: {record_id}, Name: {name}, Email: {email}")
                else:
                    logger.warning(f"Mailing list entry with empty email - ID: {record_id}, Name: {name}")
            
            connection.close()
            
            # Filter out duplicates
            unique_emails = list(set(email_list))
            logger.info(f"Total unique emails for HR escalation: {len(unique_emails)} - {unique_emails}")
            
        except Error as e:
            logger.error(f"Database error while fetching mailing list: {e}")
            # Fallback to environment variables if database fails
            logger.warning("Falling back to environment variable email configuration")
            email_list = [
                os.getenv('HR_EMAIL', 'jwwl6424@gmail.com'),
                os.getenv('HR_EMAIL2', ''),
                os.getenv('HR_EMAIL3', ''),
                os.getenv('HR_EMAIL4', '')
            ]
            unique_emails = list(set([email.strip() for email in email_list if email.strip()]))
            logger.info(f"Fallback emails: {unique_emails}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching mailing list: {e}")
            # Fallback to environment variables
            unique_emails = [os.getenv('HR_EMAIL', 'jwwl6424@gmail.com')]
            logger.info(f"Emergency fallback to single email: {unique_emails}")

        # Send email notification to each HR contact
        
        for email in unique_emails:
            logger.info(f"Sending HR escalation email to: {email}")
            email_sent = send_hr_escalation_email(
                escalation_id=escalation_id,
                user_id=user_id,
                chat_id=chat_id,
                user_query=user_query,
                user_description=user_description,
                hr_email=email
            )
            if email_sent:
                logger.info(f"Email notification sent to HR for escalation {escalation_id}")
            else:
                logger.warning(f"Failed to send email notification for escalation {escalation_id}")
            
        # Store HR escalation in dedicated database table if function is provided
        db_success = False
        if store_hr_escalation_func:
            try:
                db_success = store_hr_escalation_func(
                    escalation_id=escalation_id,
                    user_id=user_id,
                    chat_id=chat_id,
                    user_message=user_query,
                    issue_summary=sanitized_issue,
                    status="PENDING",
                    priority="NORMAL",
                    user_description=user_description
                )
            except Exception as db_error:
                logger.error(f"Failed to store HR escalation in database: {db_error}")
        
        if not db_success:
            logger.warning(f"Failed to store HR escalation {escalation_id} in database, but continuing with process")
        
        # Also maintain CSV backup logging
        try:
            with open("hr_escalations.csv", "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header if file is empty
                if f.tell() == 0:
                    writer.writerow(["escalation_id", "timestamp", "user_id", "chat_id", "original_query", "user_description", "issue_summary", "status"])
                
                writer.writerow([
                    escalation_id,
                    datetime.now().isoformat(),
                    user_id,
                    chat_id,
                    user_query[:200] + "..." if len(user_query) > 200 else user_query,
                    user_description[:300] + "..." if user_description and len(user_description) > 300 else (user_description or ""),
                    sanitized_issue,
                    "PENDING"
                ])
                
        except Exception as log_error:
            logger.error(f"Failed to log HR escalation to CSV: {log_error}")
        
        # Store in chat logs as well for tracking if function is provided
        if store_chat_log_updated_func:
            try:
                escalation_summary = f"HR_ESCALATION_INITIATED: {escalation_id}"
                if user_description:
                    escalation_summary += f" | User Description: {user_description[:100]}..."
                
                store_chat_log_updated_func(
                    user_message=user_query, 
                    bot_response=escalation_summary, 
                    query_score=0.0, 
                    relevance_score=2.0, 
                    user_id=user_id, 
                    chat_id=chat_id
                )
            except Exception as db_error:
                logger.error(f"Failed to store HR escalation in chat logs: {db_error}")
        
        # Build the response message with conditional acknowledgment of user description
        description_acknowledgment = ""
        if user_description and user_description.strip():
            description_acknowledgment = f"\n\nIssue Description Recorded:\n\"{user_description[:200]}{'...' if len(user_description) > 200 else ''}\"\n\n"
        
        hr_response = f"""HR Escalation Successfully Initiated

Your concern has been escalated to our Human Resources department with the highest priority.{description_acknowledgment}
Escalation Details:
• Reference ID: {escalation_id}
• Date & Time: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
• Status: Active and Under Review
• HR Notification: Email sent to HR representative

What Happens Next:
1. Immediate: Your case has been logged in our HR system and HR has been notified via email
2. Within 4 hours: HR will acknowledge receipt via email
3. Within 24 hours: An HR representative will contact you directly
4. Ongoing: You will receive regular updates until resolution

Important Information:
• All escalations are handled with strict confidentiality
• You may be contacted for additional details if needed
• For urgent safety matters, also contact your immediate supervisor
• Keep your reference ID for tracking purposes

Contact Information:
• HR Emergency Line: Available 24/7 for urgent matters
• HR Email: hr@verztec.com
• HR Direct Line: [Your HR Phone Number]

Your wellbeing and concerns are our top priority. HR is equipped to handle sensitive matters with the care and attention they deserve.
Please inform me if you would like to escalate this matter further or if you have any additional questions."""

        return {
            'text': hr_response,
            'images': [],
            'sources': [],
            'tool_used': True,
            'tool_identified': 'raise_to_hr',
            'tool_confidence': 'executed_successfully'
        }
        
    except Exception as e:
        logger.error(f"Error in HR escalation tool: {str(e)}", exc_info=True)
        return {
            'text': "Sorry, there was an error processing your HR escalation. Please contact HR directly or try again later.",
            'images': [],
            'sources': [],
            'tool_used': False,
            'tool_identified': 'raise_to_hr',
            'tool_confidence': f'error - {str(e)}'
        }






def execute_meeting_scheduling_tool(
        # Send meeting request email using the new function
       
    user_query: str, 
    user_id: str, 
    chat_id: str,
    user_description: Optional[str] = None,
    store_chat_log_updated_func: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Execute meeting scheduling tool with intelligent detail extraction and enhanced response formatting.
    
    Args:
        user_query (str): The original user query that triggered meeting scheduling
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        user_description (str, optional): Additional description provided by the user for meeting details
        store_chat_log_updated_func (callable, optional): Function to store chat log
        
    Returns:
        dict: Formatted response for meeting scheduling with extracted details
    """
    try:
        # Combine user query with additional description for better extraction
        combined_input = user_query
        if user_description and user_description.strip():
            combined_input = f"{user_query}. Additional details: {user_description}"
            logger.info(f"User provided additional meeting details: {user_description[:200]}...")
        
        # Extract meeting details from combined input
        meeting_details = extract_meeting_details(combined_input)
        if meeting_details is None:
            logger.warning("Meeting details extraction failed; using empty details.")
            meeting_details = {}

        # Generate meeting request ID
        meeting_request_id = f"MTG-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{user_id[:8]}"

        # Enhanced logging with extracted details
        logger.info(f"Meeting request initiated - ID: {meeting_request_id}, User: {user_id}, Chat: {chat_id}")
        logger.info(f"Extracted details - Subject: {meeting_details.get('subject', 'Not specified')}")
        logger.info(f"Extracted details - DateTime: {meeting_details.get('date_time', 'Not specified')}")
        logger.info(f"Extracted details - Participants: {meeting_details.get('participants', [])}")
        logger.info(f"Extraction confidence: {meeting_details.get('extraction_confidence', 'unknown')}")
        
        
        
        email_sent = send_meeting_request_email(
            meeting_request_id=meeting_request_id,
            user_id=user_id,
            chat_id=chat_id,
            meeting_details=meeting_details,
            user_query=user_query,
            user_description=user_description,
            meeting_email=os.getenv('HR_EMAIL', 'jwwl6424@gmail.com') 
        )
        if email_sent:
            logger.info(f"Meeting request email sent to {os.getenv('HR_EMAIL', 'jwwl6424@gmail.com') } for meeting {meeting_request_id}")
        else:
            logger.warning(f"Failed to send meeting request email for meeting {meeting_request_id}")
        
        # Store enhanced meeting details in CSV
        try:
            with open("meeting_requests.csv", "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header if file is empty
                if f.tell() == 0:
                    writer.writerow([
                        "request_id", "timestamp", "user_id", "chat_id", "original_query",
                        "user_description", "extracted_subject", "extracted_datetime", "extracted_duration", 
                        "extracted_participants", "meeting_type", "location", "priority",
                        "extraction_confidence", "status"
                    ])
                
                writer.writerow([
                    meeting_request_id,
                    datetime.now().isoformat(),
                    user_id,
                    chat_id,
                    user_query[:300] + "..." if len(user_query) > 300 else user_query,
                    user_description[:300] + "..." if user_description and len(user_description) > 300 else (user_description or ""),
                    meeting_details.get('subject', ''),
                    meeting_details.get('date_time', ''),
                    meeting_details.get('duration', ''),
                    '; '.join(meeting_details.get('participants', [])),
                    meeting_details.get('meeting_type', ''),
                    meeting_details.get('location', ''),
                    meeting_details.get('priority', 'normal'),
                    meeting_details.get('extraction_confidence', 'unknown'),
                    "AWAITING_USER_CONFIRMATION"
                ])
                
        except Exception as log_error:
            logger.error(f"Failed to log meeting request: {log_error}")
        
        # Store in database if function is provided
        if store_chat_log_updated_func:
            try:
                meeting_summary = f"MEETING_REQUEST_PREPARED: {meeting_request_id} | Status: AWAITING_USER_CONFIRMATION"
                if meeting_details.get('subject'):
                    meeting_summary += f" | Subject: {meeting_details['subject'][:50]}..."
                if meeting_details.get('date_time'):
                    meeting_summary += f" | DateTime: {meeting_details['date_time']}"
                
                store_chat_log_updated_func(
                    user_message=user_query, 
                    bot_response=meeting_summary, 
                    query_score=0.0, 
                    relevance_score=2.0, 
                    user_id=user_id, 
                    chat_id=chat_id
                )
            except Exception as db_error:
                logger.error(f"Failed to store meeting request in database: {db_error}")
        
        # Build additional details acknowledgment
        additional_details_acknowledgment = ""
        if user_description and user_description.strip():
            additional_details_acknowledgment = f"\n**Additional Details Provided:**\n\"{user_description[:200]}{'...' if len(user_description) > 200 else ''}\"\n"
        
        # Build extracted details section for user review
        extracted_details_section = "\n**📋 Extracted Meeting Details:**\n"
        
        if meeting_details.get('subject'):
            extracted_details_section += f"• **Subject:** {meeting_details['subject']}\n"
        else:
            extracted_details_section += f"• **Subject:** Not specified\n"
        
        if meeting_details.get('date_time'):
            extracted_details_section += f"• **Date & Time:** {meeting_details['date_time']}\n"
        else:
            extracted_details_section += f"• **Date & Time:** Not specified\n"
        
        if meeting_details.get('duration'):
            extracted_details_section += f"• **Duration:** {meeting_details['duration']}\n"
        else:
            extracted_details_section += f"• **Duration:** Not specified\n"
        
        if meeting_details.get('participants'):
            participants_list = meeting_details['participants'][:5]  # Limit display
            if len(meeting_details['participants']) > 5:
                participants_list.append(f"... and {len(meeting_details['participants']) - 5} more")
            extracted_details_section += f"• **Participants:** {', '.join(participants_list)}\n"
        else:
            extracted_details_section += f"• **Participants:** Not specified\n"
        
        if meeting_details.get('meeting_type'):
            extracted_details_section += f"• **Meeting Type:** {meeting_details['meeting_type'].title()}\n"
        
        if meeting_details.get('location'):
            extracted_details_section += f"• **Location:** {meeting_details['location']}\n"
        
        if meeting_details.get('priority') != 'normal':
            extracted_details_section += f"• **Priority:** {meeting_details['priority'].title()}\n"
        
        extracted_details_section += f"\n*AI Extraction Confidence: {meeting_details['extraction_confidence'].title()}*\n"
        
        # Determine response timeline based on priority
        response_timeline = "Within 2 hours"
        if meeting_details.get('priority') == 'high':
            response_timeline = "Within 1 hour"
        elif meeting_details.get('priority') == 'low':
            response_timeline = "Within 4 hours"
        
        # Build the comprehensive response with confirmation options
        meeting_response = f"""📅 **Your Meeting Request Is Ready!**

Here’s what I’ve gathered from your request:
{extracted_details_section}
{additional_details_acknowledgment}
**Your Original Request:**
"{user_query[:200]}{'...' if len(user_query) > 200 else ''}"

---
**What happens next:**
• Our team will review your request and send you a confirmation within {response_timeline.lower()}.
• You’ll receive an email update as soon as your meeting is scheduled.

If you need to make changes or have questions, just reply here or email **meetings@verztec.com** (Ref: {meeting_request_id}).

Thank you for letting us help coordinate your meeting!"""

        return {
            'text': meeting_response,
            'images': [],
            'sources': [],
            'tool_used': True,
            'tool_identified': 'schedule_meeting',
            'tool_confidence': 'executed_successfully',
            'extracted_details': meeting_details,  # Include for potential API consumers
            'meeting_confirmation': {
                'meeting_request_id': meeting_request_id,
                'buttons': [
                    {
                        'id': 'confirm_meeting',
                        'text': '✅ Confirm Meeting',
                        'action': 'confirm meeting',
                        'style': 'primary'
                    },
                    {
                        'id': 'modify_meeting',
                        'text': '✏️ Modify Details',
                        'action': 'modify meeting',
                        'style': 'secondary'
                    },
                    {
                        'id': 'cancel_meeting',
                        'text': '❌ Cancel Request',
                        'action': 'cancel meeting',
                        'style': 'danger'
                    }
                ],
                'original_details': {
                    'subject': meeting_details.get('subject', 'Not specified'),
                    'date_time': meeting_details.get('date_time', 'Not specified'),
                    'participants': meeting_details.get('participants', []),
                    'meeting_type': meeting_details.get('meeting_type'),
                    'location': meeting_details.get('location'),
                    'duration': meeting_details.get('duration'),
                    'priority': meeting_details.get('priority', 'normal')
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in meeting scheduling tool: {str(e)}", exc_info=True)
        return {
            'text': "Sorry, there was an error processing your meeting request. Please contact the coordination team directly or try again later.",
            'images': [],
            'sources': [],
            'tool_used': False,
            'tool_identified': 'schedule_meeting',
            'tool_confidence': f'error - {str(e)}'
        }


def handle_meeting_confirmation_response_2(
    user_response: str,
    meeting_request_id: str,
    user_id: str,
    chat_id: str,
    original_details: Dict[str, Any],
    store_chat_log_updated_func: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Handle user response to meeting confirmation (confirm, modify, or cancel).
    
    Args:
        user_response (str): User's response to the meeting confirmation
        meeting_request_id (str): The meeting request ID from the original request
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        original_details (dict): Original extracted meeting details
        store_chat_log_updated_func (callable, optional): Function to store chat log
        
    Returns:
        dict: Response indicating the action taken
    """
    try:
        response_lower = user_response.lower().strip()
        
        # Check for confirmation
        if any(phrase in response_lower for phrase in ['confirm meeting', 'confirm', 'yes', 'proceed', 'submit']):
            # Update meeting status to confirmed and submit
            try:
                # Update CSV with confirmed status
                import pandas as pd
                
                # Read existing CSV
                csv_file = "meeting_requests.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    # Update status for this request
                    df.loc[df['request_id'] == meeting_request_id, 'status'] = 'CONFIRMED_SUBMITTED'
                    df.to_csv(csv_file, index=False)
                
            except Exception as csv_error:
                logger.warning(f"Failed to update CSV status: {csv_error}")
            
            # Store confirmation in database
            if store_chat_log_updated_func:
                try:
                    confirmation_summary = f"MEETING_CONFIRMED: {meeting_request_id} | Status: SUBMITTED_TO_COORDINATION_TEAM"
                    store_chat_log_updated_func(
                        user_message=user_response,
                        bot_response=confirmation_summary,
                        query_score=1.0,
                        relevance_score=0.0,
                        user_id=user_id,
                        chat_id=chat_id
                    )
                except Exception as db_error:
                    logger.error(f"Failed to store confirmation in database: {db_error}")
            
            logger.info(f"Meeting {meeting_request_id} confirmed and submitted by user {user_id}")
            
            response_text = f"""✅ **Your Meeting Is Confirmed!**

Your meeting request (**{meeting_request_id}**) has been sent to our coordination team.

**What’s next:**
• We’ll review your request and send you a confirmation email soon.
• Calendar invites will be sent to all participants once scheduled.

**Meeting Summary:**
• **Subject:** {original_details.get('subject', 'Not specified')}
• **Date & Time:** {original_details.get('date_time', 'Not specified')}
• **Participants:** {', '.join(original_details.get('participants', ['Not specified']))}

If you have any questions or need to make changes, just reply here or email **meetings@verztec.com** (Ref: {meeting_request_id}).

Thank you for using our meeting service! 📅"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_confirmed',
                'tool_confidence': 'confirmed_and_submitted'
            }
            
        # Check for cancellation
        elif any(phrase in response_lower for phrase in ['cancel meeting', 'cancel', 'nevermind', 'discard']):
            # Update meeting status to cancelled
            try:
                import pandas as pd
                csv_file = "meeting_requests.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df.loc[df['request_id'] == meeting_request_id, 'status'] = 'CANCELLED_BY_USER'
                    df.to_csv(csv_file, index=False)
            except Exception as csv_error:
                logger.warning(f"Failed to update CSV status: {csv_error}")
            
            logger.info(f"Meeting {meeting_request_id} cancelled by user {user_id}")
            
            response_text = f"""❌ **Meeting Request Cancelled**

Your meeting request (**{meeting_request_id}**) has been cancelled.

No further action will be taken. If you’d like to schedule a new meeting, just let me know!

Is there anything else I can help you with today?"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_cancelled',
                'tool_confidence': 'cancelled_successfully'
            }
            
        # Check for modification request
        elif any(phrase in response_lower for phrase in ['modify meeting', 'modify', 'change', 'update']):
            # Extract what needs to be modified and re-run extraction
            modification_text = user_response
            
            # Remove the command part to get the actual changes
            for phrase in ['modify meeting', 'modify', 'change', 'update']:
                if phrase in response_lower:
                    modification_text = user_response[response_lower.find(phrase) + len(phrase):].strip()
                    if modification_text.startswith('-'):
                        modification_text = modification_text[1:].strip()
                    break
            
            logger.info(f"Meeting modification requested for {meeting_request_id}: {modification_text}")
            
            response_text = f"""✏️ **You’d Like to Update Your Meeting**

You requested changes to meeting **{meeting_request_id}**.

**Your requested changes:** {modification_text}

To help me update your meeting, please provide the full new details (date, time, participants, etc.).
For example: “Schedule a meeting tomorrow at 3pm with John and Sarah to discuss budget review in conference room C.”

You can also:
• **Cancel** this request and start over
• **Confirm** the original request as-is
• Or contact meetings@verztec.com for complex changes

What would you like to do next?"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_modification_requested',
                'tool_confidence': 'awaiting_new_details'
            }
            
        else:
            # Unclear response - ask for clarification
            response_text = f"""❓ **Could You Clarify Your Response?**

I wasn’t sure what you’d like to do with meeting request **{meeting_request_id}**.

Please reply with one of these options:
• **"confirm meeting"** – to submit as shown
• **"modify meeting"** – and tell me what to change
• **"cancel meeting"** – to discard this request

For example:
• “confirm meeting”
• “modify meeting – change time to 4pm”
• “cancel meeting”

**Current meeting details:**
• **Subject:** {original_details.get('subject', 'Not specified')}
• **Date & Time:** {original_details.get('date_time', 'Not specified')}
• **Participants:** {', '.join(original_details.get('participants', ['Not specified']))}

What would you like to do?"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_clarification_needed',
                'tool_confidence': 'awaiting_clear_response'
            }
            
    except Exception as e:
        logger.error(f"Error handling meeting confirmation response: {str(e)}", exc_info=True)
        return {
            'text': f"Sorry, there was an error processing your response to meeting request {meeting_request_id}. Please contact meetings@verztec.com for assistance.",
            'images': [],
            'sources': [],
            'tool_used': False,
            'tool_identified': 'meeting_confirmation_error',
            'tool_confidence': f'error - {str(e)}'
        }


# ...existing code...
def handle_meeting_confirmation_response(
    user_response: str,
    meeting_request_id: str,
    user_id: str,
    chat_id: str,
    original_details: Dict[str, Any],
    store_chat_log_updated_func: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Handle user response to meeting confirmation (confirm, modify, or cancel).
    
    Args:
        user_response (str): User's response to the meeting confirmation
        meeting_request_id (str): The meeting request ID from the original request
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        original_details (dict): Original extracted meeting details
        store_chat_log_updated_func (callable, optional): Function to store chat log
        
    Returns:
        dict: Response indicating the action taken
    """
    try:
        response_lower = user_response.lower().strip()
        
        # Check for confirmation
        if any(phrase in response_lower for phrase in ['confirm meeting', 'confirm', 'yes', 'proceed', 'submit']):
            # Update meeting status to confirmed and submit
            try:
                # Update CSV with confirmed status
                import pandas as pd
                
                # Read existing CSV
                csv_file = "meeting_requests.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    # Update status for this request
                    df.loc[df['request_id'] == meeting_request_id, 'status'] = 'CONFIRMED_SUBMITTED'
                    df.to_csv(csv_file, index=False)
                
            except Exception as csv_error:
                logger.warning(f"Failed to update CSV status: {csv_error}")
            
            # Store confirmation in database
            if store_chat_log_updated_func:
                try:
                    confirmation_summary = f"MEETING_CONFIRMED: {meeting_request_id} | Status: SUBMITTED_TO_COORDINATION_TEAM"
                    store_chat_log_updated_func(
                        user_message=user_response,
                        bot_response=confirmation_summary,
                        query_score=1.0,
                        relevance_score=0.0,
                        user_id=user_id,
                        chat_id=chat_id
                    )
                except Exception as db_error:
                    logger.error(f"Failed to store confirmation in database: {db_error}")
            
            logger.info(f"Meeting {meeting_request_id} confirmed and submitted by user {user_id}")
            
            response_text = f"""✅ **Meeting Confirmed & Submitted!**

Your meeting request **{meeting_request_id}** has been successfully submitted to our coordination team.

**What happens next:**
• **Within 2 hours:** Coordination team will review your request
• **Within 1 business day:** You'll receive scheduling confirmation via email  
• **Upon scheduling:** Calendar invitations will be sent to all participants
• **24 hours before meeting:** Automatic reminder notifications

**Meeting Summary:**
• **Subject:** {original_details.get('subject', 'Not specified')}
• **Date & Time:** {original_details.get('date_time', 'Not specified')}
• **Participants:** {', '.join(original_details.get('participants', ['Not specified']))}

**Contact Info:** For any changes or questions, email meetings@verztec.com with reference ID: **{meeting_request_id}**

Thank you for using our meeting coordination service! 📅"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_confirmed',
                'tool_confidence': 'confirmed_and_submitted'
            }
            
        # Check for cancellation
        elif any(phrase in response_lower for phrase in ['cancel meeting', 'cancel', 'nevermind', 'discard']):
            # Update meeting status to cancelled
            try:
                import pandas as pd
                csv_file = "meeting_requests.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    df.loc[df['request_id'] == meeting_request_id, 'status'] = 'CANCELLED_BY_USER'
                    df.to_csv(csv_file, index=False)
            except Exception as csv_error:
                logger.warning(f"Failed to update CSV status: {csv_error}")
            
            logger.info(f"Meeting {meeting_request_id} cancelled by user {user_id}")
            
            response_text = f"""❌ **Meeting Request Cancelled**

Your meeting request **{meeting_request_id}** has been cancelled and removed from the system.

No further action will be taken on this request. If you need to schedule a meeting in the future, just let me know!

Is there anything else I can help you with today?"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_cancelled',
                'tool_confidence': 'cancelled_successfully'
            }
            
        # Check for modification request
        elif any(phrase in response_lower for phrase in ['modify meeting', 'modify', 'change', 'update']):
            # Extract what needs to be modified and re-run extraction
            modification_text = user_response
            
            # Remove the command part to get the actual changes
            for phrase in ['modify meeting', 'modify', 'change', 'update']:
                if phrase in response_lower:
                    modification_text = user_response[response_lower.find(phrase) + len(phrase):].strip()
                    if modification_text.startswith('-'):
                        modification_text = modification_text[1:].strip()
                    break
            
            logger.info(f"Meeting modification requested for {meeting_request_id}: {modification_text}")
            
            response_text = f"""✏️ **Meeting Modification Requested**

I understand you want to modify your meeting request **{meeting_request_id}**.

**Requested Changes:** {modification_text}

To process your modification, I'll need to create a new meeting request with the updated details. 

Please provide your complete updated meeting request, for example:
"Schedule a meeting tomorrow at 3pm with John and Sarah to discuss budget review in conference room C"

I'll then extract the new details and ask for your confirmation again.

**Alternative:** You can also:
• **Cancel** this request and start fresh
• **Confirm** the original request as-is
• Contact meetings@verztec.com directly for complex changes

What would you like to do?"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_modification_requested',
                'tool_confidence': 'awaiting_new_details'
            }
            
        else:
            # Unclear response - ask for clarification
            response_text = f"""❓ **Please Clarify Your Response**

I didn't understand your response regarding meeting request **{meeting_request_id}**.

Please respond with one of the following:

• **"confirm meeting"** - to submit the request as shown
• **"modify meeting"** - followed by what you want to change  
• **"cancel meeting"** - to discard this request

For example:
• "confirm meeting"
• "modify meeting - change time to 4pm"  
• "cancel meeting"

**Current meeting details are:**
• **Subject:** {original_details.get('subject', 'Not specified')}
• **Date & Time:** {original_details.get('date_time', 'Not specified')}
• **Participants:** {', '.join(original_details.get('participants', ['Not specified']))}

What would you like to do?"""

            return {
                'text': response_text,
                'images': [],
                'sources': [],
                'tool_used': True,
                'tool_identified': 'meeting_clarification_needed',
                'tool_confidence': 'awaiting_clear_response'
            }
            
    except Exception as e:
        logger.error(f"Error handling meeting confirmation response: {str(e)}", exc_info=True)
        return {
            'text': f"Sorry, there was an error processing your response to meeting request {meeting_request_id}. Please contact meetings@verztec.com for assistance.",
            'images': [],
            'sources': [],
            'tool_used': False,
            'tool_identified': 'meeting_confirmation_error',
            'tool_confidence': f'error - {str(e)}'
        }

from pathlib import Path

def get_vacation_days(user_id: str, filename: str = r"leave.csv") -> int:
    """
    Reads leave.csv and returns the number of vacation days for the given user_id.
    Assumes columns: user_id, vacation_days
    """
    try:
        logger.info(f"Fetching vacation days for user_id: {user_id} from {filename}")
        path = Path(filename)
        if not path.exists():
            logger.info(f"File does not exist: {filename}")
            return 0

        with path.open(mode='r', encoding='utf-8-sig') as csvfile:
            logger.info(f"Reading vacation days from {filename}")
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('user_id') == str(user_id):  # Ensure string match
                    return int(row.get('vacation_days', 0))
        return 0  # User not found
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        return 0

def execute_vacation_check_tool(user_query, user_id, chat_id, store_chat_log_updated_func):
    """
    Responds to a vacation check query using consistent tool return format.
    """
    vacation_days = get_vacation_days(user_id)
    response_text = f"You have {vacation_days} days of vacation remaining."

    # Logging in same structure as meeting scheduling tool
    if store_chat_log_updated_func:
        try:
            vacation_summary = f"VACATION_CHECK_EXECUTED | Days Remaining: {vacation_days}"

            store_chat_log_updated_func(
                user_message=user_query, 
                bot_response=vacation_summary, 
                query_score=0.0, 
                relevance_score=2.0, 
                user_id=user_id, 
                chat_id=chat_id
            )
        except Exception as db_error:
            logger.error(f"Failed to store vacation check log in database: {db_error}")

    return {
        'text': response_text,
        'images': [],
        'sources': [],
        'tool_used': True,
        'tool_identified': 'vacation_check',
        'tool_confidence': 'executed_successfully',
        'extracted_details': {
            'user_id': user_id,
            'vacation_days': vacation_days,
            'query': user_query
        },
        'vacation_summary': {
            'days_remaining': vacation_days,
            'last_updated': 'N/A'
        }
    }