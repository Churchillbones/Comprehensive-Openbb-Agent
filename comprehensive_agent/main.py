from typing import AsyncGenerator
import json
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sse_starlette.sse import EventSourceResponse
import logging

from openbb_ai.models import (
    MessageChunkSSE, 
    QueryRequest,
    SingleFileReference,
    SingleDataContent,
    PdfDataFormat,
    DataContent,
    DataFileReferences,
    WidgetRequest,
)
from openbb_ai import get_widget_data, message_chunk, reasoning_step

from .config import settings
from .prompts import SYSTEM_PROMPT, REASONING_PROMPTS, ERROR_MESSAGES
from .processors.widgets import process_widget_data
from .processors.widget_intelligence import widget_intelligence
from .processors.pdf import process_pdf_data
from .processors.spreadsheet import process_spreadsheet_data
from .processors.api_data import process_api_data
from .processors.citations import generate_citations
from .processors.web_search import process_web_search, detect_web_search_request
from .visualizations.charts import generate_charts
from .visualizations.tables import generate_tables
from .processors.financial_web_search import FinancialWebSearcher
from .core.ml_widget_bridge import ml_widget_bridge
from .utils.data_correlator import correlate_sentiment_with_prices  # Assume utility for fusion
from .utils.alerting import send_alert

app = FastAPI(title="Comprehensive OpenBB Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pro.openbb.co", "http://localhost:7777", "http://127.0.0.1:7777", "https://your-agent-domain.ngrok-free.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Validation error for {request.url}: {exc.errors()}")
    
    # Check if this is a file upload validation error
    request_body = await request.body()
    try:
        import json as json_module
        body_data = json_module.loads(request_body)
        
        # Handle file upload widget missing description
        if ("widgets" in body_data and 
            "primary" in body_data["widgets"] and 
            body_data["widgets"]["primary"]):
            
            for i, widget in enumerate(body_data["widgets"]["primary"]):
                if (widget.get("widget_id") == "file-undefined" and 
                    "description" not in widget):
                    widget["description"] = f"Uploaded file: {widget.get('name', 'unknown')}"
                    logger.info(f"Fixed widget {i}: added description for {widget.get('name')}")
            
            # Try to re-validate with the corrected data
            try:
                from openbb_ai.models import QueryRequest
                corrected_request = QueryRequest.model_validate(body_data)
                # If validation succeeds, call the actual endpoint
                return await query(corrected_request)
            except Exception as e:
                logger.error(f"Still failed after correction: {e}")
    
    except Exception as e:
        logger.error(f"Error processing validation error: {e}")
    
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "errors": str(exc.errors())}
    )

@app.get("/agents.json")
def get_agent_description():
    return JSONResponse(
        content={
            "comprehensive_agent": {
                "name": settings.openbb_agent_name,
                "description": settings.openbb_agent_description,
                "image": settings.openbb_agent_image,
                "endpoints": {"query": f"http://localhost:{settings.server_port}/v1/query"},
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": True,
                    "file-upload": True,
                    "web-search": True,
                },
                "tools": [
                    {
                        "name": "web",
                        "description": "Search the internet for current information. Use @web followed by your search query.",
                        "usage": "@web [search query]",
                        "examples": [
                            "@web latest Apple earnings report",
                            "@web current interest rates",
                            "@web Tesla stock news today"
                        ]
                    }
                ]
            }
        }
    )

@app.post("/v1/query")
async def query(request: QueryRequest) -> EventSourceResponse:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"=== QUERY ENDPOINT REACHED ===")
    logger.info(f"Received query request with {len(request.messages)} messages")
    
    # Debug widgets and messages
    if request.widgets:
        logger.info(f"=== WIDGETS FOUND ===")
        logger.info(f"Primary widgets: {len(request.widgets.primary) if request.widgets.primary else 0}")
        if request.widgets.primary:
            for i, widget in enumerate(request.widgets.primary):
                logger.info(f"Widget {i}: {widget.widget_id}, name: {widget.name}")
    
    # Debug file upload
    for i, message in enumerate(request.messages):
        logger.info(f"=== MESSAGE {i} ===")
        logger.info(f"Role: {message.role}")
        if hasattr(message, 'files'):
            logger.info(f"Message {i} has files attribute: {message.files}")
        if message.role == "human":
            logger.info(f"Human message {i}: {message.content[:100]}...")
            if hasattr(message, '__dict__'):
                logger.info(f"Message attributes: {list(message.__dict__.keys())}")
        elif message.role == "tool":
            logger.info(f"=== TOOL MESSAGE {i} DETAILS ===")
            logger.info(f"Tool message data type: {type(message.data)}")
            if hasattr(message, 'data'):
                logger.info(f"Tool message data: {str(message.data)[:1000]}...")
            if hasattr(message, '__dict__'):
                logger.info(f"Tool message attributes: {list(message.__dict__.keys())}")
    if (
        request.messages[-1].role == "human"
        and request.widgets
        and request.widgets.primary
    ):
        # Check if this contains file uploads (file-undefined widget)
        has_file_uploads = any(widget.widget_id == "file-undefined" for widget in request.widgets.primary)
        non_file_widgets = [w for w in request.widgets.primary if w.widget_id != "file-undefined"]
        
        logger.info(f"Has file uploads: {has_file_uploads}")
        logger.info(f"Non-file widgets: {len(non_file_widgets)}")
        
        # Process ALL widgets (including file uploads) through get_widget_data
        widget_requests = []
        for widget in request.widgets.primary:
            logger.info(f"Creating widget request for: {widget.widget_id} - {widget.name}")
            widget_requests.append(
                WidgetRequest(
                    widget=widget,
                    input_arguments={
                        param.name: param.current_value for param in widget.params
                    },
                )
            )

        async def retrieve_widget_data():
            logger.info(f"Getting widget data for {len(widget_requests)} widgets")
            result = get_widget_data(widget_requests)
            logger.info(f"Widget data result type: {type(result)}")
            
            # Log the actual widget data content for debugging
            if hasattr(result, 'model_dump'):
                data_dump = result.model_dump()
                logger.info(f"Widget data dump: {str(data_dump)[:500]}...")
            elif hasattr(result, '__dict__'):
                logger.info(f"Widget data dict: {str(result.__dict__)[:500]}...")
            
            yield result.model_dump()

        return EventSourceResponse(
            content=retrieve_widget_data(),
            media_type="text/event-stream",
        )

    async def execution_loop() -> AsyncGenerator[MessageChunkSSE, None]:
        try:
            yield reasoning_step(REASONING_PROMPTS["starting"]).model_dump()
            
            ollama_messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]

            context_str = ""
            widget_data = None
            pdf_data = None
            spreadsheet_data = None
            api_data = None
            uploaded_files = []
            web_search_results = None
            
            # Check for file uploads in widgets
            if request.widgets and request.widgets.primary:
                for widget in request.widgets.primary:
                    if widget.widget_id == "file-undefined":
                        logger.info(f"Processing file upload widget: {widget.name}")
                        logger.info(f"Widget metadata: {widget.metadata}")
                        logger.info(f"Widget params: {widget.params}")
                        logger.info(f"Widget attributes: {dir(widget)}")
                        
                        # Try to find file content in multiple places
                        file_content = None
                        
                        # 1. Check widget metadata
                        if hasattr(widget, 'metadata') and widget.metadata:
                            file_content = widget.metadata.get('content') or widget.metadata.get('data') or widget.metadata.get('base64_content')
                            logger.info(f"Checked metadata for content: {file_content is not None}")
                        
                        # 2. Check widget params
                        if not file_content and hasattr(widget, 'params'):
                            for param in widget.params:
                                if hasattr(param, 'current_value') and param.current_value:
                                    if isinstance(param.current_value, str) and len(param.current_value) > 100:
                                        file_content = param.current_value
                                        logger.info(f"Found content in param: {param.name}")
                                        break
                        
                        # 3. Check if widget has data attribute
                        if not file_content and hasattr(widget, 'data'):
                            widget_data = widget.data
                            if isinstance(widget_data, dict):
                                file_content = widget_data.get('content') or widget_data.get('base64_content')
                                logger.info(f"Found content in widget.data: {file_content is not None}")
                        
                        # 4. Check widget attributes directly
                        if not file_content:
                            for attr in ['content', 'file_content', 'base64_content', 'raw_content']:
                                if hasattr(widget, attr):
                                    potential_content = getattr(widget, attr)
                                    if potential_content:
                                        file_content = potential_content
                                        logger.info(f"Found content in widget.{attr}")
                                        break
                        
                        # Create a file object from the widget
                        file_obj = type('FileUpload', (), {
                            'filename': widget.name,
                            'content': file_content,
                            'widget': widget  # Keep reference to original widget
                        })()
                        uploaded_files.append(file_obj)
                        logger.info(f"Found uploaded file widget: {widget.name}, has content: {file_content is not None}")
            
            for index, message in enumerate(request.messages):
                if message.role == "human":
                    content = message.content
                    
                    # Check for web search requests
                    web_search_request = detect_web_search_request(content)
                    if web_search_request and not web_search_results:  # Only do one search per request
                        yield reasoning_step("Searching the web for current information...").model_dump()
                        
                        searcher = FinancialWebSearcher()
                        search_results = searcher.search_and_analyze(content)
                        logging.info(f'Query: {content}, Results: {json.dumps(search_results, default=str)}')
                        yield reasoning_step(f'Interpreting sentiment: Average score {sum(r["sentiment_score"] for r in search_results) / max(1, len(search_results)):.2f}')
                        correlated_insights = correlate_sentiment_with_prices(search_results, openbb_data)
                        send_alert(search_results)
                        web_search_results = {
                            "results": search_results,
                            "correlated_insights": correlated_insights
                        }
                        
                        if web_search_results and not web_search_results.get("error"):
                            results_summary = f"Found {len(web_search_results.get('results', []))} web search results"
                            yield reasoning_step(f"Web search complete: {results_summary}").model_dump()
                            
                            # Add web search results to context
                            web_context = _format_web_search_results(web_search_results)
                            context_str += f"\n\nWeb Search Results:\n{web_context}"
                        else:
                            error_msg = web_search_results.get("error", "Web search failed") if web_search_results else "Web search failed"
                            yield reasoning_step(f"Web search error: {error_msg}").model_dump()
                    
                    if hasattr(message, 'files') and message.files:
                        uploaded_files.extend(message.files)
                        file_names = [f.filename for f in message.files if hasattr(f, 'filename')]
                        if file_names:
                            content += f"\n\nUploaded files: {', '.join(file_names)}"
                    
                    ollama_messages.append(
                        {"role": "user", "content": content}
                    )
                elif message.role == "ai":
                    if isinstance(message.content, str):
                        ollama_messages.append(
                            {"role": "assistant", "content": message.content}
                        )
                elif message.role == "tool" and index == len(request.messages) - 1:
                    yield reasoning_step(REASONING_PROMPTS["processing_widgets"]).model_dump()

                    widget_data = message.data
                    logger.info(f"=== PROCESSING TOOL MESSAGE ===")
                    logger.info(f"Tool message data type: {type(widget_data)}")
                    logger.info(f"Raw tool message data: {str(widget_data)[:1000]}...")

                    # Apply widget intelligence if enabled
                    if settings.enable_widget_intelligence and request.widgets and request.widgets.primary:
                        yield reasoning_step(REASONING_PROMPTS["analyzing_widgets"]).model_dump()
                        try:
                            # Analyze dashboard context
                            dashboard_profile = await widget_intelligence.analyze_dashboard(request.widgets.primary)
                            logger.info(f"Dashboard profile: {dashboard_profile}")

                            # Add dashboard context to ollama messages
                            dashboard_context = f"\n\n**Dashboard Context:**\n"
                            dashboard_context += f"- Primary focus: {dashboard_profile.get('primary_focus', 'general')}\n"
                            dashboard_context += f"- Analysis style: {dashboard_profile.get('analysis_style', 'balanced')}\n"
                            if dashboard_profile.get('top_symbols'):
                                dashboard_context += f"- Tracking: {', '.join(dashboard_profile['top_symbols'][:5])}\n"
                            dashboard_context += f"- Widgets: {dashboard_profile.get('widget_count', 0)}\n"

                            context_str += dashboard_context
                        except Exception as e:
                            logger.warning(f"Widget intelligence analysis failed: {e}")
                    
                    # Use the official OpenBB approach for handling widget data (including PDFs)
                    try:
                        if isinstance(widget_data, list):
                            # Check if we have any PDF files
                            has_pdfs = any(
                                any(isinstance(item.data_format, PdfDataFormat) for item in result.items)
                                for result in widget_data
                                if hasattr(result, 'items')
                            )
                            
                            if has_pdfs:
                                yield reasoning_step(REASONING_PROMPTS["analyzing_pdf"]).model_dump()
                            
                            # Process all widget data including PDFs
                            processed_data = await handle_widget_data_with_files(widget_data)
                            context_str += f"\n\n{processed_data}"

                            # Apply ML analysis if enabled
                            if settings.enable_ml_predictions and request.widgets and request.widgets.primary:
                                yield reasoning_step(REASONING_PROMPTS["applying_ml"]).model_dump()
                                try:
                                    # Analyze each widget with ML
                                    ml_insights = []
                                    for widget in request.widgets.primary:
                                        widget_context = await widget_intelligence.analyze_widget(widget, widget_data)
                                        ml_result = await ml_widget_bridge.auto_apply_ml(
                                            widget_data,
                                            widget_context.detected_type,
                                            widget_context.__dict__
                                        )

                                        if ml_result.get("status") == "success" and ml_result.get("insights"):
                                            ml_insights.extend(ml_result["insights"])
                                            logger.info(f"ML insights for {widget.name}: {ml_result['insights']}")

                                    # Add ML insights to context
                                    if ml_insights:
                                        context_str += f"\n\n**ML Insights:**\n"
                                        for insight in ml_insights[:5]:  # Limit to top 5
                                            context_str += f"- {insight}\n"
                                except Exception as e:
                                    logger.warning(f"ML analysis failed: {e}")
                        else:
                            # Handle single DataContent object
                            processed_data = await handle_widget_data_with_files([widget_data])
                            context_str += f"\n\n{processed_data}"
                    
                    except Exception as e:
                        logger.error(f"Error processing widget data: {str(e)}")
                        # Fall back to old method
                        file_upload_error = _check_for_file_upload_error(widget_data)
                        if file_upload_error:
                            yield reasoning_step(f"File upload issue detected: {file_upload_error}").model_dump()
                            context_str += f"\n\nFile Upload Status: {file_upload_error}"
                            
                            # Add troubleshooting information
                            troubleshooting_info = """
                            
Troubleshooting Steps:
1. Ensure the PDF file is not corrupted
2. Try uploading a smaller PDF file (under 10MB)
3. Make sure the file is a valid PDF format
4. The file upload should now work better with the updated processing method
                            """
                            context_str += troubleshooting_info

            if uploaded_files:
                yield reasoning_step(REASONING_PROMPTS["processing_files"]).model_dump()
                file_context = await _process_uploaded_files(uploaded_files)
                if file_context:
                    context_str += f"\n\nUploaded Files Content:\n{file_context}"
                    yield reasoning_step(f"Successfully processed {len(uploaded_files)} file(s)").model_dump()
                else:
                    # If we can't get file content, at least acknowledge the files
                    file_names = [f.filename for f in uploaded_files if hasattr(f, 'filename')]
                    if file_names:
                        context_str += f"\n\nUploaded Files: {', '.join(file_names)} (content extraction in progress...)"
                        yield reasoning_step(f"Found {len(file_names)} file(s) but content extraction needs improvement").model_dump()

            if context_str:
                ollama_messages[-1]["content"] += f"\n\n{context_str}"

            yield reasoning_step(REASONING_PROMPTS["finalizing"]).model_dump()

            # Check Ollama availability before processing
            try:
                async with httpx.AsyncClient(timeout=settings.connection_timeout) as client:
                    response = await client.get(f"{settings.ollama_base_url}/api/version")
                    if response.status_code != 200:
                        raise httpx.HTTPStatusError(f"Ollama not available", request=response.request, response=response)
            except Exception as e:
                error_msg = f"Ollama service not available at {settings.ollama_base_url}. Please start Ollama with: 'ollama serve'"
                yield reasoning_step(error_msg, event_type="ERROR").model_dump()
                return

            # Use separate timeouts for connection and streaming
            timeout_config = httpx.Timeout(
                connect=settings.connection_timeout,
                read=settings.stream_timeout,
                write=settings.connection_timeout,
                pool=settings.connection_timeout
            )
            
            yield reasoning_step("Generating response using local AI model...").model_dump()

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                async with client.stream(
                    "POST",
                    f"{settings.ollama_base_url}/api/chat",
                    json={
                        "model": settings.ollama_model,
                        "messages": ollama_messages,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_predict": 2048,  # Limit response length
                            "num_ctx": 4096,      # Context window
                            "stop": ["<|im_end|>", "<|endoftext|>"]
                        }
                    },
                    headers={"Content-Type": "application/json"},
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if content := data.get("message", {}).get("content"):
                                    yield message_chunk(content).model_dump()
                            except json.JSONDecodeError:
                                continue

            if widget_data:
                chart_data = await generate_charts(widget_data)
                if chart_data:
                    yield reasoning_step(REASONING_PROMPTS["generating_charts"]).model_dump()
                    for chart in chart_data:
                        yield chart.model_dump()

                table_data = await generate_tables(widget_data)
                if table_data:
                    yield reasoning_step(REASONING_PROMPTS["creating_tables"]).model_dump()
                    for table in table_data:
                        yield table.model_dump()

                citations = await generate_citations(widget_data, request.widgets.primary if request.widgets else [])
                if citations:
                    for citation in citations:
                        yield citation.model_dump()

            yield reasoning_step(REASONING_PROMPTS["complete"]).model_dump()

        except httpx.ReadTimeout:
            error_msg = f"{ERROR_MESSAGES['timeout']} The model may be processing a complex request. Try a simpler query or increase timeout settings."
            yield reasoning_step(error_msg, event_type="ERROR").model_dump()
        except httpx.ConnectTimeout:
            error_msg = f"Connection timeout to Ollama server. Check if Ollama is running: 'ollama serve'"
            yield reasoning_step(error_msg, event_type="ERROR").model_dump()
        except httpx.ConnectError:
            error_msg = f"{ERROR_MESSAGES['ollama_connection']} Make sure Ollama is running with: 'ollama serve'"
            yield reasoning_step(error_msg, event_type="ERROR").model_dump()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = f"Ollama endpoint not found. Check if the model 'gemma3n:e2b' is available. Run: 'ollama pull gemma3n:e2b'"
                yield reasoning_step(error_msg, event_type="ERROR").model_dump()
            else:
                yield reasoning_step(f"HTTP Error {e.response.status_code}: {str(e)}", event_type="ERROR").model_dump()
        except Exception as e:
            yield reasoning_step(f"{ERROR_MESSAGES['general']}: {str(e)}", event_type="ERROR").model_dump()

    return EventSourceResponse(
        content=execution_loop(),
        media_type="text/event-stream",
    )

async def _process_uploaded_files(files) -> str:
    """Process uploaded files and extract relevant content"""
    file_contents = []
    
    for file in files:
        try:
            filename = getattr(file, 'filename', 'unknown')
            content = getattr(file, 'content', None)
            widget = getattr(file, 'widget', None)
            
            # Debug file object
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Processing file: {filename}")
            logger.info(f"Content available: {content is not None}")
            
            if widget:
                logger.info(f"Widget available, checking for file data...")
                # Try multiple ways to get content from widget
                if hasattr(widget, '__dict__'):
                    logger.info(f"Widget dict: {widget.__dict__}")
                
                # Try to get content from widget data
                if hasattr(widget, 'data'):
                    logger.info(f"Widget has data attribute")
                    widget_data = widget.data
                    if isinstance(widget_data, dict):
                        content = widget_data.get('content') or widget_data.get('base64_content') or widget_data.get('file_content')
                        logger.info(f"Found content in widget data: {content is not None}")
                
                # Try metadata
                if not content and hasattr(widget, 'metadata'):
                    content = widget.metadata.get('file_content') or widget.metadata.get('base64_content') or widget.metadata.get('content')
                    logger.info(f"Found content in widget metadata: {content is not None}")
                
                # Try params
                if not content and hasattr(widget, 'params'):
                    for param in widget.params:
                        if hasattr(param, 'current_value') and param.current_value:
                            if isinstance(param.current_value, str) and len(param.current_value) > 100:
                                content = param.current_value
                                logger.info(f"Found content in widget param: {param.name}")
                                break
            
            if not content:
                # Try to find content in file object itself
                for attr in ['data', 'file_content', 'base64_content', 'raw_content']:
                    if hasattr(file, attr):
                        potential_content = getattr(file, attr)
                        if potential_content:
                            content = potential_content
                            logger.info(f"Found content in file.{attr}")
                            break
            
            if not content:
                file_contents.append(f"=== {filename} ===\nFile uploaded but content not accessible. Available attributes: {list(getattr(file, '__dict__', {}).keys())}")
                continue
                
            # Process different file types
            if filename.lower().endswith('.pdf'):
                logger.info(f"Processing PDF: {filename}")
                pdf_text = await _process_pdf_content(content, filename)
                if pdf_text:
                    logger.info(f"Successfully extracted {len(pdf_text)} characters from PDF")
                    file_contents.append(f"=== {filename} (PDF) ===\n{pdf_text[:3000]}...")
                else:
                    logger.warning(f"Failed to extract PDF content from {filename}")
                    file_contents.append(f"=== {filename} (PDF) ===\nFailed to extract PDF content - please ensure the file is a valid PDF")
                    
            elif filename.lower().endswith(('.xlsx', '.xls', '.csv')):
                sheet_data = await _process_spreadsheet_content(content, filename)
                if sheet_data:
                    file_contents.append(f"=== {filename} (Spreadsheet) ===\n{sheet_data}")
                else:
                    file_contents.append(f"=== {filename} (Spreadsheet) ===\nFailed to process spreadsheet")
                    
            elif filename.lower().endswith('.json'):
                json_data = await _process_json_content(content, filename)
                if json_data:
                    file_contents.append(f"=== {filename} (JSON) ===\n{json_data}")
                else:
                    file_contents.append(f"=== {filename} (JSON) ===\nFailed to parse JSON")
                    
            else:
                text_data = await _process_text_content(content, filename)
                if text_data:
                    file_contents.append(f"=== {filename} (Text) ===\n{text_data[:1000]}...")
                else:
                    file_contents.append(f"=== {filename} ===\nBinary file - unable to extract text content")
                    
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            file_contents.append(f"=== {filename} ===\nError processing file: {str(e)}")
    
    return "\n\n".join(file_contents) if file_contents else ""

async def _download_file(url: str) -> bytes:
    """Download file from URL"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading file from {url}")
    async with httpx.AsyncClient() as client:
        file_content = await client.get(url)
        return file_content.content

async def _get_url_pdf_text(data: SingleFileReference) -> str:
    """Extract text from PDF served from URL"""
    import pdfplumber
    from io import BytesIO
    
    file_content = await _download_file(str(data.url))
    with pdfplumber.open(BytesIO(file_content)) as pdf:
        document_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                document_text += page_text + "\n\n"
        return document_text

async def _get_base64_pdf_text(data: SingleDataContent) -> str:
    """Extract text from base64-encoded PDF"""
    import base64
    import pdfplumber
    from io import BytesIO
    
    file_content = base64.b64decode(data.content)
    with pdfplumber.open(BytesIO(file_content)) as pdf:
        document_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                document_text += page_text + "\n\n"
        return document_text

async def handle_widget_data_with_files(data: list[DataContent | DataFileReferences]) -> str:
    """Handle widget data including PDF files using official OpenBB approach"""
    import logging
    logger = logging.getLogger(__name__)
    
    result_str = "--- Data ---\n"
    for result in data:
        for item in result.items:
            if isinstance(item.data_format, PdfDataFormat):
                filename = item.data_format.filename
                result_str += f"===== {filename} =====\n"
                logger.info(f"Processing PDF: {filename}")
                
                try:
                    if isinstance(item, SingleDataContent):
                        # Handle the base64 PDF case
                        pdf_text = await _get_base64_pdf_text(item)
                        result_str += pdf_text
                        logger.info(f"Successfully extracted {len(pdf_text)} characters from base64 PDF")
                    elif isinstance(item, SingleFileReference):
                        # Handle the URL PDF case
                        pdf_text = await _get_url_pdf_text(item)
                        result_str += pdf_text
                        logger.info(f"Successfully extracted {len(pdf_text)} characters from URL PDF")
                except Exception as e:
                    logger.error(f"Failed to process PDF {filename}: {str(e)}")
                    result_str += f"Error processing PDF: {str(e)}\n"
            else:
                # Handle other data formats by just dumping the content as a string
                result_str += f"{item.content}\n"
            result_str += "------\n"
    return result_str

def _check_for_file_upload_error(widget_data) -> str:
    """Check if widget data contains file upload error messages"""
    try:
        # Handle list of DataContent objects
        if isinstance(widget_data, list):
            for item in widget_data:
                if hasattr(item, 'items'):
                    for sub_item in item.items:
                        if hasattr(sub_item, 'content'):
                            content = sub_item.content
                            if isinstance(content, str) and "Failed to get data for data source" in content:
                                if "file-undefined" in content:
                                    return "File upload failed - OpenBB could not process the uploaded file. The file may be corrupted or the upload mechanism needs to be fixed."
                                else:
                                    return f"Data source error: {content}"
        
        # Handle single DataContent object
        elif hasattr(widget_data, 'items'):
            for item in widget_data.items:
                if hasattr(item, 'content'):
                    content = item.content
                    if isinstance(content, str) and "Failed to get data for data source" in content:
                        if "file-undefined" in content:
                            return "File upload failed - OpenBB could not process the uploaded file. The file may be corrupted or the upload mechanism needs to be fixed."
                        else:
                            return f"Data source error: {content}"
        
        return ""
    except Exception as e:
        return f"Error checking for file upload issues: {str(e)}"

async def _process_pdf_content(content: str, filename: str) -> str:
    """Process PDF content from base64 string"""
    try:
        from .processors.pdf import _extract_pdf_from_base64
        return _extract_pdf_from_base64(content)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"PDF processing failed for {filename}: {str(e)}")
        return ""

async def _process_spreadsheet_content(content: str, filename: str) -> str:
    """Process spreadsheet content"""
    try:
        import base64
        import pandas as pd
        from io import BytesIO
        
        # Decode base64 content
        file_bytes = base64.b64decode(content)
        
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(BytesIO(file_bytes))
            return f"CSV with {len(df)} rows and {len(df.columns)} columns\nColumns: {', '.join(df.columns.tolist())}\n\nFirst 5 rows:\n{df.head().to_string()}"
        else:
            # Excel file
            excel_file = pd.ExcelFile(BytesIO(file_bytes))
            result = f"Excel file with {len(excel_file.sheet_names)} sheet(s): {', '.join(excel_file.sheet_names)}\n\n"
            
            # Process first sheet
            df = pd.read_excel(BytesIO(file_bytes), sheet_name=0)
            result += f"First sheet ({excel_file.sheet_names[0]}):\n{len(df)} rows and {len(df.columns)} columns\n"
            result += f"Columns: {', '.join(df.columns.tolist())}\n\nFirst 5 rows:\n{df.head().to_string()}"
            
            return result
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Spreadsheet processing failed for {filename}: {str(e)}")
        return ""

async def _process_json_content(content: str, filename: str) -> str:
    """Process JSON content"""
    try:
        import base64
        import json
        
        json_bytes = base64.b64decode(content)
        json_data = json.loads(json_bytes.decode('utf-8'))
        
        if isinstance(json_data, dict):
            return f"JSON object with {len(json_data)} keys: {', '.join(list(json_data.keys())[:10])}\n\nSample data:\n{json.dumps(json_data, indent=2)[:1000]}..."
        elif isinstance(json_data, list):
            return f"JSON array with {len(json_data)} items\n\nFirst item:\n{json.dumps(json_data[0] if json_data else {}, indent=2)[:1000]}..."
        else:
            return f"JSON data: {str(json_data)[:1000]}..."
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"JSON processing failed for {filename}: {str(e)}")
        return ""

async def _process_text_content(content: str, filename: str) -> str:
    """Process text content"""
    try:
        import base64
        
        text_content = base64.b64decode(content).decode('utf-8')
        return text_content
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Text processing failed for {filename}: {str(e)}")
        return ""

def _format_web_search_results(web_search_results: dict) -> str:
    """Format web search results for context"""
    if not web_search_results or web_search_results.get("error"):
        return "No web search results available."
    
    results = web_search_results.get("results", [])
    if not results:
        return "No web search results found."
    
    formatted_results = []
    for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
        title = result.get("title", "No title")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        content_preview = result.get("content_preview", "")
        
        result_text = f"{i}. **{title}**\n"
        if url:
            result_text += f"   URL: {url}\n"
        if snippet:
            result_text += f"   Summary: {snippet}\n"
        if content_preview and content_preview != snippet:
            result_text += f"   Content: {content_preview[:200]}...\n"
        
        formatted_results.append(result_text)
    
    query = web_search_results.get("query", "")
    timestamp = web_search_results.get("timestamp", "")
    
    header = f"Web Search Results for '{query}' (searched at {timestamp}):\n\n"
    return header + "\n".join(formatted_results)

def main():
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)

if __name__ == "__main__":
    main()