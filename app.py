__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dotenv import load_dotenv
import json

# Telegram Bot Library
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# CrewAI
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Gemini API
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Available Gemini models
DEFAULT_MODEL = "gemini-pro"

# Custom class for wrapping Gemini with LLM interface compatible with CrewAI
class GeminiLLM:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = genai.GenerativeModel(name=model_name)
        self.supports_functions = True
        self.supports_stop_words = False
    
    def __call__(self, prompt: str, **kwargs):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            return f"Error generating content: {str(e)}"

    # Add the necessary methods expected by CrewAI
    def function_calling_llm(self):
        return self

# Custom tools for CrewAI agents - with proper type annotations
class InternetSearchTool(BaseTool):
    name: str = "Internet Search"
    description: str = "Search for information on the internet."
    
    def _run(self, query: str) -> str:
        """Simulate searching the internet for information."""
        # In a real implementation, this would use a search API
        llm = GeminiLLM()
        prompt = f"Search the internet for: {query}\nProvide a comprehensive summary of the results."
        result = llm(prompt)
        return result

class DataAnalysisTool(BaseTool):
    name: str = "Data Analysis"
    description: str = "Analyze data and provide insights."
    
    def _run(self, data: str) -> str:
        """Simulate data analysis."""
        llm = GeminiLLM()
        prompt = f"Analyze the following data and provide key insights:\n{data}"
        result = llm(prompt)
        return result

class ContentCreationTool(BaseTool):
    name: str = "Content Creation"
    description: str = "Create high-quality content based on a topic."
    
    def _run(self, topic: str) -> str:
        """Simulate content creation."""
        llm = GeminiLLM()
        prompt = f"Create high-quality content about: {topic}"
        result = llm(prompt)
        return result

# Define the CrewAI agents
def create_researcher_agent() -> Agent:
    """Create a researcher agent."""
    return Agent(
        role="Research Specialist",
        goal="Find accurate and detailed information on any given topic",
        backstory="You are an expert researcher with years of experience in finding reliable information quickly.",
        llm=GeminiLLM(),
        tools=[InternetSearchTool()],
        verbose=True,
        allow_delegation=True,
    )

def create_analyst_agent() -> Agent:
    """Create an analyst agent."""
    return Agent(
        role="Data Analyst",
        goal="Analyze data and extract meaningful insights",
        backstory="You are a data analyst with expertise in interpreting complex information and finding patterns.",
        llm=GeminiLLM(),
        tools=[DataAnalysisTool()],
        verbose=True,
        allow_delegation=True,
    )

def create_writer_agent() -> Agent:
    """Create a content writer agent."""
    return Agent(
        role="Content Writer",
        goal="Create engaging and informative content",
        backstory="You are a skilled writer capable of creating compelling content on any topic.",
        llm=GeminiLLM(),
        tools=[ContentCreationTool()],
        verbose=True,
        allow_delegation=True,
    )

# Task types and their corresponding agent creators
TASK_TYPES = {
    "research": create_researcher_agent,
    "analysis": create_analyst_agent,
    "writing": create_writer_agent,
}

# Store active tasks and crews for each user
user_tasks: Dict[int, Dict[str, Any]] = {}

# Helper function to run synchronous tasks in a thread-safe way
def run_sync_task(func, *args, **kwargs):
    """Run a synchronous function in a thread-safe way."""
    result = None
    error = None
    
    def thread_target():
        nonlocal result, error
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = e
    
    thread = threading.Thread(target=thread_target)
    thread.start()
    thread.join()
    
    if error:
        raise error
    return result

# Helper functions
def ask_gemini_directly_sync(query: str, model_name: str = DEFAULT_MODEL) -> str:
    """Ask Gemini directly without using CrewAI (synchronous version)."""
    try:
        model = genai.GenerativeModel(name=model_name)
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        return f"Error generating content: {str(e)}"

async def ask_gemini_directly(query: str, model_name: str = DEFAULT_MODEL) -> str:
    """Ask Gemini directly without using CrewAI."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, ask_gemini_directly_sync, query, model_name)

def create_and_run_crew_sync(user_id: int, task_type: str, task_description: str) -> str:
    """Create and run a CrewAI crew for a specific task (synchronous version)."""
    try:
        # Create the primary agent based on task type
        primary_agent = TASK_TYPES[task_type]()
        
        # Add additional agents based on task complexity
        agents = [primary_agent]
        if task_type == "research":
            # For research tasks, also add writer agent to format the findings
            agents.append(create_writer_agent())
        elif task_type == "writing":
            # For writing tasks, also add researcher for gathering facts
            agents.append(create_researcher_agent())
        
        # Create the task
        task = Task(
            description=task_description,
            agent=primary_agent,
            expected_output="A comprehensive response addressing all aspects of the task."
        )
        
        # Create the crew
        crew = Crew(
            agents=agents,
            tasks=[task],
            verbose=True,
            process=Process.sequential,  # Use sequential process for deterministic results
        )
        
        # Store the crew and task info for the user
        user_tasks[user_id] = {
            "crew": crew,
            "task_description": task_description,
            "task_type": task_type,
            "agents": agents,
            "status": "running"
        }
        
        # Run the crew
        result = crew.kickoff()
        
        # Update task status
        user_tasks[user_id]["status"] = "completed"
        user_tasks[user_id]["result"] = result
        
        return result
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        return f"Error running crew: {str(e)}"

async def create_and_run_crew(user_id: int, task_type: str, task_description: str) -> str:
    """Create and run a CrewAI crew for a specific task."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, create_and_run_crew_sync, user_id, task_type, task_description)

# Telegram bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    welcome_message = (
        f"👋 Hello {user.first_name}! I'm a CrewAI powered Telegram bot that can help you with various tasks.\n\n"
        "I use multiple AI agents working together to tackle complex problems. Here's what I can do:\n"
        "• 🔍 Research topics in depth\n"
        "• 📊 Analyze data and provide insights\n"
        "• ✍️ Create high-quality content\n\n"
        "Use /help to see all available commands."
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *CrewAI Bot Commands* 🤖\n\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/research [topic] - Research a topic in depth\n"
        "/analyze [data] - Analyze data and provide insights\n"
        "/write [topic] - Create content on a topic\n"
        "/ask [question] - Ask a direct question (uses Gemini without CrewAI)\n"
        "/status - Check the status of your current task\n"
        "/cancel - Cancel your current task\n\n"
        "You can also simply send me a message, and I'll determine the best agent team to handle it!"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def research_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /research command."""
    user_id = update.effective_user.id
    
    # Check if any arguments were provided
    if not context.args:
        await update.message.reply_text("Please specify a research topic. Example: /research artificial intelligence trends")
        return
    
    # Get the research topic from arguments
    topic = " ".join(context.args)
    
    # Inform the user that research is starting
    await update.message.reply_text(f"🔍 Starting research on: '{topic}'\nThis may take a few minutes...")
    
    # Run the research task
    try:
        result = await create_and_run_crew(user_id, "research", f"Research the following topic and provide a comprehensive summary: {topic}")
    except Exception as e:
        logger.error(f"Error in research command: {e}")
        result = f"Sorry, an error occurred: {str(e)}"
    
    # Send the result
    if not result:
        result = "Sorry, I couldn't complete the research task. There might be an issue with the CrewAI system."
    
    # Send the result in chunks due to Telegram message size limitations
    MAX_MESSAGE_LENGTH = 4096
    if len(result) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(f"Research results on '{topic}':\n\n{result}")
    else:
        # Split the result into chunks
        chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(
                f"Research results on '{topic}' (Part {i+1}/{len(chunks)}):\n\n{chunk}"
            )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /analyze command."""
    user_id = update.effective_user.id
    
    # Check if any arguments were provided
    if not context.args:
        await update.message.reply_text("Please provide data to analyze. Example: /analyze sales data for Q1 and Q2")
        return
    
    # Get the data from arguments
    data = " ".join(context.args)
    
    # Inform the user that analysis is starting
    await update.message.reply_text(f"📊 Starting analysis of the provided data...\nThis may take a few minutes...")
    
    # Run the analysis task
    try:
        result = await create_and_run_crew(user_id, "analysis", f"Analyze the following data and provide insights: {data}")
    except Exception as e:
        logger.error(f"Error in analysis command: {e}")
        result = f"Sorry, an error occurred: {str(e)}"
    
    # Send the result
    if not result:
        result = "Sorry, I couldn't complete the analysis task. There might be an issue with the CrewAI system."
    
    # Send the result in chunks due to Telegram message size limitations
    MAX_MESSAGE_LENGTH = 4096
    if len(result) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(f"Analysis results:\n\n{result}")
    else:
        # Split the result into chunks
        chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(
                f"Analysis results (Part {i+1}/{len(chunks)}):\n\n{chunk}"
            )

async def write_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /write command."""
    user_id = update.effective_user.id
    
    # Check if any arguments were provided
    if not context.args:
        await update.message.reply_text("Please specify a writing topic. Example: /write benefits of exercise")
        return
    
    # Get the topic from arguments
    topic = " ".join(context.args)
    
    # Inform the user that content creation is starting
    await update.message.reply_text(f"✍️ Starting to create content on: '{topic}'\nThis may take a few minutes...")
    
    # Run the writing task
    try:
        result = await create_and_run_crew(user_id, "writing", f"Create high-quality content about the following topic: {topic}")
    except Exception as e:
        logger.error(f"Error in write command: {e}")
        result = f"Sorry, an error occurred: {str(e)}"
    
    # Send the result
    if not result:
        result = "Sorry, I couldn't complete the writing task. There might be an issue with the CrewAI system."
    
    # Send the result in chunks due to Telegram message size limitations
    MAX_MESSAGE_LENGTH = 4096
    if len(result) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(f"Content on '{topic}':\n\n{result}")
    else:
        # Split the result into chunks
        chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(
                f"Content on '{topic}' (Part {i+1}/{len(chunks)}):\n\n{chunk}"
            )

async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /ask command (direct to Gemini)."""
    # Check if any arguments were provided
    if not context.args:
        await update.message.reply_text("Please ask a question. Example: /ask What is the capital of France?")
        return
    
    # Get the question from arguments
    question = " ".join(context.args)
    
    # Inform the user that the question is being processed
    await update.message.reply_text("🤔 Processing your question...")
    
    # Ask Gemini directly
    try:
        result = await ask_gemini_directly(question)
    except Exception as e:
        logger.error(f"Error in ask command: {e}")
        result = f"Sorry, an error occurred: {str(e)}"
    
    # Send the result
    if not result:
        result = "Sorry, I couldn't process your question. There might be an issue with the Gemini API."
    
    # Send the result in chunks due to Telegram message size limitations
    MAX_MESSAGE_LENGTH = 4096
    if len(result) <= MAX_MESSAGE_LENGTH:
        await update.message.reply_text(f"Answer:\n\n{result}")
    else:
        # Split the result into chunks
        chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
        for i, chunk in enumerate(chunks):
            await update.message.reply_text(
                f"Answer (Part {i+1}/{len(chunks)}):\n\n{chunk}"
            )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check the status of the user's current task."""
    user_id = update.effective_user.id
    
    if user_id not in user_tasks:
        await update.message.reply_text("You don't have any active tasks.")
        return
    
    task_info = user_tasks[user_id]
    status = task_info.get("status", "unknown")
    task_type = task_info.get("task_type", "unknown")
    task_description = task_info.get("task_description", "unknown")
    
    status_message = (
        f"*Task Status:* {status}\n"
        f"*Task Type:* {task_type}\n"
        f"*Description:* {task_description}\n"
    )
    
    if status == "completed" and "result" in task_info:
        result_summary = task_info['result'][:200] if task_info['result'] else "No result available."
        status_message += f"\n*Result Summary:*\n{result_summary}...\n\n"
        status_message += "Use /result to view the full result."
    
    await update.message.reply_text(status_message, parse_mode="Markdown")

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancel the user's current task."""
    user_id = update.effective_user.id
    
    if user_id not in user_tasks:
        await update.message.reply_text("You don't have any active tasks to cancel.")
        return
    
    # Remove the user's task
    del user_tasks[user_id]
    await update.message.reply_text("✅ Your current task has been cancelled.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages and determine the appropriate action."""
    user_id = update.effective_user.id
    message_text = update.message.text
    
    # Create keyboard with task type options
    keyboard = [
        [
            InlineKeyboardButton("Research", callback_data=f"task_research_{message_text}"),
            InlineKeyboardButton("Analysis", callback_data=f"task_analysis_{message_text}"),
            InlineKeyboardButton("Writing", callback_data=f"task_writing_{message_text}"),
        ],
        [
            InlineKeyboardButton("Direct Question", callback_data=f"direct_{message_text}"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "How would you like me to process your request?",
        reply_markup=reply_markup,
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from inline keyboards."""
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    callback_data = query.data
    
    if callback_data.startswith("task_"):
        # Extract task type and message
        parts = callback_data.split("_", 2)
        if len(parts) < 3:
            await query.edit_message_text("Invalid request format.")
            return
        
        task_type = parts[1]
        message = parts[2]
        
        if task_type not in TASK_TYPES:
            await query.edit_message_text(f"Unknown task type: {task_type}")
            return
        
        # Inform the user that the task is starting
        await query.edit_message_text(f"Starting {task_type} task for: '{message}'\nThis may take a few minutes...")
        
        # Run the task
        try:
            result = await create_and_run_crew(user_id, task_type, message)
        except Exception as e:
            logger.error(f"Error in handle_callback (task): {e}")
            result = f"Sorry, an error occurred: {str(e)}"
        
        # Handle empty results
        if not result:
            result = "Sorry, I couldn't complete the task. There might be an issue with the CrewAI system."
        
        # Send the result in chunks due to Telegram message size limitations
        MAX_MESSAGE_LENGTH = 4096
        if len(result) <= MAX_MESSAGE_LENGTH:
            await context.bot.send_message(chat_id=user_id, text=f"Results:\n\n{result}")
        else:
            # Split the result into chunks
            chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
            for i, chunk in enumerate(chunks):
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"Results (Part {i+1}/{len(chunks)}):\n\n{chunk}"
                )
    
    elif callback_data.startswith("direct_"):
        # Extract the question
        question = callback_data[7:]  # Remove "direct_" prefix
        
        # Inform the user that the question is being processed
        await query.edit_message_text("Processing your question directly...")
        
        # Ask Gemini directly
        try:
            result = await ask_gemini_directly(question)
        except Exception as e:
            logger.error(f"Error in handle_callback (direct): {e}")
            result = f"Sorry, an error occurred: {str(e)}"
        
        # Handle empty results
        if not result:
            result = "Sorry, I couldn't process your question. There might be an issue with the Gemini API."
        
        # Send the result in chunks due to Telegram message size limitations
        MAX_MESSAGE_LENGTH = 4096
        if len(result) <= MAX_MESSAGE_LENGTH:
            await context.bot.send_message(chat_id=user_id, text=f"Answer:\n\n{result}")
        else:
            # Split the result into chunks
            chunks = [result[i:i + MAX_MESSAGE_LENGTH] for i in range(0, len(result), MAX_MESSAGE_LENGTH)]
            for i, chunk in enumerate(chunks):
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"Answer (Part {i+1}/{len(chunks)}):\n\n{chunk}"
                )

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("research", research_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("write", write_command))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Add message handler for general messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    application.run_polling()

if __name__ == "__main__":
    main()
