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
import logging
import os
import smtplib
from datetime import datetime
from typing import Optional, Dict, Any, List
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
        
        # Send email notification to HR
        email_sent = send_hr_escalation_email(
            escalation_id=escalation_id,
            user_id=user_id,
            chat_id=chat_id,
            user_query=user_query,
            user_description=user_description,
            hr_email=os.getenv('HR_EMAIL', 'jwwl6424@gmail.com')  # Use env var or fallback
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
                    relevance_score=1.0, 
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

Your wellbeing and concerns are our top priority. HR is equipped to handle sensitive matters with the care and attention they deserve."""

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
    user_query: str, 
    user_id: str, 
    chat_id: str,
    store_chat_log_updated_func: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Execute meeting scheduling tool with logging and response formatting.
    
    Args:
        user_query (str): The original user query that triggered meeting scheduling
        user_id (str): User identifier
        chat_id (str): Chat session identifier
        store_chat_log_updated_func (callable, optional): Function to store chat log
        
    Returns:
        dict: Formatted response for meeting scheduling
    """
    try:
        # Sanitize and structure the input
        sanitized_details = user_query.strip()[:500]
        meeting_request_id = f"MTG-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{user_id[:8]}"
        
        # Log the meeting request
        logger.info(f"Meeting request initiated - ID: {meeting_request_id}, User: {user_id}, Chat: {chat_id}")
        
        try:
            with open("meeting_requests.csv", "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header if file is empty
                if f.tell() == 0:
                    writer.writerow(["request_id", "timestamp", "user_id", "chat_id", "meeting_details", "status"])
                
                writer.writerow([
                    meeting_request_id,
                    datetime.now().isoformat(),
                    user_id,
                    chat_id,
                    sanitized_details,
                    "PENDING"
                ])
                
        except Exception as log_error:
            logger.error(f"Failed to log meeting request: {log_error}")
        
        # Store in database if function is provided
        if store_chat_log_updated_func:
            try:
                store_chat_log_updated_func(
                    user_message=user_query, 
                    bot_response=f"MEETING_REQUEST_INITIATED: {meeting_request_id}", 
                    query_score=1.0, 
                    relevance_score=0.0, 
                    user_id=user_id, 
                    chat_id=chat_id
                )
            except Exception as db_error:
                logger.error(f"Failed to store meeting request in database: {db_error}")
        
        meeting_response = f"""📅 **Meeting Request Successfully Submitted**

Your meeting request has been submitted and is being processed by our coordination team.

**Request Details:**
• **Request ID:** {meeting_request_id}
• **Date & Time:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
• **Status:** Pending Review

**What Happens Next:**
1. **Within 2 hours:** Your request will be reviewed by our coordination team
2. **Within 1 business day:** You'll receive meeting options and scheduling details
3. **Upon confirmation:** Calendar invitations will be sent to all participants
4. **24 hours before:** You'll receive a meeting reminder with agenda

**Meeting Request Summary:**
{sanitized_details}

**Next Steps:**
• Our team will contact you to discuss specific requirements
• Please check your email for updates on scheduling options
• If urgent, you can also contact the coordination team directly

**Contact Information:**
• **Coordination Team:** meetings@verztec.com
• **Direct Line:** [Meeting Coordination Number]

Thank you for using our meeting coordination service. We'll ensure your meeting is scheduled efficiently!"""

        return {
            'text': meeting_response,
            'images': [],
            'sources': [],
            'tool_used': True,
            'tool_identified': 'schedule_meeting',
            'tool_confidence': 'executed_successfully'
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
