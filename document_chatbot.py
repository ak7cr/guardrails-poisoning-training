#!/usr/bin/env python3
"""
Document Processing System with Smart Filtering
Handles PDF and CSV attachments like ChatGPT/Gemini with guardrail protection
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import PyPDF2
import pdfplumber
import io
import os
from datetime import datetime
import re
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple

# Import production guardrail
from production_guardrail import ProductionSmartGuardrail, FilteringStrategy

# Check for PyMuPDF availability
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

@dataclass
class DocumentInfo:
    """Information about a processed document"""
    filename: str
    file_type: str
    size_bytes: int
    page_count: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    processing_time_ms: float = 0
    summary: Optional[str] = None
    error: Optional[str] = None

@dataclass
class DocumentQuery:
    """Query about documents with filtering information"""
    original_query: str
    filtered_query: str
    is_filtered: bool
    safety_score: float
    document_refs: List[str]
    query_type: str  # "general", "specific", "analysis", "summary"

class DocumentProcessor:
    """Handles PDF and CSV document processing"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.csv', '.txt'}
        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        
    def process_pdf(self, file_path: str) -> DocumentInfo:
        """Process PDF file and extract text"""
        start_time = datetime.now()
        
        try:
            # Try PyMuPDF first (better quality)
            if HAS_PYMUPDF:
                return self._process_pdf_pymupdf(file_path, start_time)
            else:
                return self._process_pdf_pdfplumber(file_path, start_time)
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return DocumentInfo(
                filename=os.path.basename(file_path),
                file_type="pdf",
                size_bytes=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                processing_time_ms=processing_time,
                error=str(e)
            )

    def _process_pdf_pymupdf(self, file_path: str, start_time: datetime) -> DocumentInfo:
        """Process PDF using PyMuPDF (better quality)"""
        doc = fitz.open(file_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        doc.close()
        
        full_text = "\n\n".join(text_content)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DocumentInfo(
            filename=os.path.basename(file_path),
            file_type="pdf",
            size_bytes=os.path.getsize(file_path),
            page_count=len(text_content),
            processing_time_ms=processing_time,
            summary=self._generate_summary(full_text)
        )

    def _process_pdf_pdfplumber(self, file_path: str, start_time: datetime) -> DocumentInfo:
        """Process PDF using pdfplumber (fallback)"""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
        
        full_text = "\n\n".join(text_content)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return DocumentInfo(
            filename=os.path.basename(file_path),
            file_type="pdf",
            size_bytes=os.path.getsize(file_path),
            page_count=len(text_content),
            processing_time_ms=processing_time,
            summary=self._generate_summary(full_text)
        )

    def process_csv(self, file_path: str) -> Tuple[DocumentInfo, Optional[pd.DataFrame]]:
        """Process CSV file and return info + dataframe"""
        start_time = datetime.now()
        
        try:
            # Read CSV with multiple encodings
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to decode CSV file with common encodings")
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Generate summary
            summary = self._generate_csv_summary(df)
            
            return DocumentInfo(
                filename=os.path.basename(file_path),
                file_type="csv",
                size_bytes=os.path.getsize(file_path),
                row_count=len(df),
                column_count=len(df.columns),
                processing_time_ms=processing_time,
                summary=summary
            ), df
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return DocumentInfo(
                filename=os.path.basename(file_path),
                file_type="csv",
                size_bytes=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                processing_time_ms=processing_time,
                error=str(e)
            ), None

    def _generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate a summary of text content"""
        if not text or len(text.strip()) == 0:
            return "Document appears to be empty or contains no readable text."
        
        # Clean and truncate text
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        if len(cleaned_text) <= max_length:
            return cleaned_text
        
        # Try to find sentence boundaries
        sentences = re.split(r'[.!?]+', cleaned_text)
        summary = ""
        
        for sentence in sentences:
            if len(summary + sentence) <= max_length:
                summary += sentence + ". "
            else:
                break
        
        if not summary:
            summary = cleaned_text[:max_length] + "..."
        
        return summary.strip()

    def _generate_csv_summary(self, df: pd.DataFrame) -> str:
        """Generate summary for CSV data"""
        summary_parts = []
        summary_parts.append(f"Dataset with {len(df)} rows and {len(df.columns)} columns")
        summary_parts.append(f"Columns: {', '.join(df.columns.tolist()[:10])}")  # First 10 columns
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"Numeric columns: {len(numeric_cols)}")
        
        return ". ".join(summary_parts)

class SmartDocumentChatbot:
    """Document-aware chatbot with smart filtering - works like ChatGPT with document support"""
    
    def __init__(self):
        print("🤖 Initializing Smart Document Chatbot...")
        
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.guardrail = ProductionSmartGuardrail()
        
        # Document storage
        self.documents: Dict[str, DocumentInfo] = {}
        self.document_texts: Dict[str, str] = {}
        self.document_dataframes: Dict[str, pd.DataFrame] = {}
        
        print("✅ Smart Document Chatbot ready!")
    
    def upload_document(self, file_path: str) -> DocumentInfo:
        """Upload and process a document"""
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        print(f"📁 Processing document: {filename}")
        
        if file_ext not in self.doc_processor.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        try:
            if file_ext == '.pdf':
                doc_info = self.doc_processor.process_pdf(file_path)
                # Read the text content for searching
                if HAS_PYMUPDF:
                    doc = fitz.open(file_path)
                    text_content = []
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text()
                        if text.strip():
                            text_content.append(text)
                    doc.close()
                    self.document_texts[filename] = "\n\n".join(text_content)
                else:
                    with pdfplumber.open(file_path) as pdf:
                        text_content = []
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text and text.strip():
                                text_content.append(text)
                        self.document_texts[filename] = "\n\n".join(text_content)
                
            elif file_ext == '.csv':
                doc_info, df = self.doc_processor.process_csv(file_path)
                if df is not None:
                    self.document_dataframes[filename] = df
                    self.document_texts[filename] = self._dataframe_to_text(df)
                    
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                self.document_texts[filename] = text_content
                doc_info = DocumentInfo(
                    filename=filename,
                    file_type="txt",
                    size_bytes=os.path.getsize(file_path),
                    processing_time_ms=0,
                    summary=self.doc_processor._generate_summary(text_content)
                )
            
            # Store document info
            self.documents[filename] = doc_info
            
            if doc_info.error:
                print(f"❌ Error processing {filename}: {doc_info.error}")
            else:
                print(f"✅ Successfully processed {filename}")
                if doc_info.page_count:
                    print(f"   📄 Pages: {doc_info.page_count}")
                if doc_info.row_count:
                    print(f"   📈 Rows: {doc_info.row_count}, Columns: {doc_info.column_count}")
                print(f"   ⏱️  Processing time: {doc_info.processing_time_ms:.1f}ms")
            
            return doc_info
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            raise

    def _dataframe_to_text(self, df: pd.DataFrame, max_rows: int = 100) -> str:
        """Convert DataFrame to text representation"""
        text_parts = []
        
        # Basic info
        text_parts.append(f"CSV Dataset: {len(df)} rows, {len(df.columns)} columns")
        text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Data types
        text_parts.append("\nData Types:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            text_parts.append(f"  {col}: {dtype}")
        
        # Summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            text_parts.append("\nNumeric Summary:")
            for col in numeric_cols:
                stats = df[col].describe()
                text_parts.append(f"  {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        # Sample data
        text_parts.append(f"\nFirst {min(max_rows, len(df))} rows:")
        sample_df = df.head(max_rows)
        text_parts.append(sample_df.to_string())
        
        return "\n".join(text_parts)

    def ask_question(self, question: str, strategy: FilteringStrategy = FilteringStrategy.BALANCED) -> Dict:
        """Ask a question - uses uploaded documents as context for natural conversation"""
        
        # First, filter the question for safety
        filter_result = self.guardrail.filter_content(question, strategy)
        
        # Generate contextual response using any uploaded documents as background knowledge
        doc_query = DocumentQuery(
            original_query=question,
            filtered_query=filter_result.filtered_text,
            is_filtered=filter_result.is_modified,
            safety_score=filter_result.safety_score,
            document_refs=list(self.documents.keys()),
            query_type="contextual_conversation"
        )
        
        # Generate response with document context
        ai_response = self._handle_contextual_conversation(doc_query)
        
        return self._create_response(doc_query, ai_response, "success")

    def _handle_contextual_conversation(self, doc_query: DocumentQuery) -> str:
        """Handle conversation with document context - natural AI chat using documents as knowledge base"""
        
        query = doc_query.filtered_query.lower()
        
        # If no documents, handle as pure general conversation
        if not self.documents:
            return self._generate_general_response(doc_query.filtered_query)
        
        # Get document context for the conversation
        context_info = self._get_document_context()
        
        # Generate contextual response
        return self._generate_contextual_response(doc_query.filtered_query, context_info)

    def _get_document_context(self) -> str:
        """Get summarized context from all uploaded documents"""
        context_parts = []
        
        for filename in self.documents.keys():
            doc_info = self.documents[filename]
            
            if doc_info.error:
                continue
                
            # Add document summary to context
            if doc_info.summary:
                context_parts.append(f"From {filename}: {doc_info.summary}")
            
            # Add key data points for CSV files
            if filename in self.document_dataframes:
                df = self.document_dataframes[filename]
                context_parts.append(f"Data from {filename}: {len(df)} records with columns {', '.join(df.columns.tolist()[:5])}")
        
        return " | ".join(context_parts) if context_parts else ""

    def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate AI response using document context"""
        
        query_lower = query.lower()
        
        # Greeting responses with context awareness
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            if context:
                return f"Hello! I'm ready to help with any questions. I also have access to your uploaded documents and can reference them in our conversation if relevant."
            else:
                return "Hello! I'm your AI assistant. Feel free to ask me anything!"
        
        # Help responses with context
        elif any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
            help_text = """I can help you with:

• **General conversations** - Ask me anything about various topics
• **Programming help** - Code, debugging, best practices
• **Writing assistance** - Essays, emails, creative writing
• **Math and calculations** - Problem solving and explanations
• **Research and analysis** - Using information to answer questions"""
            
            if context:
                help_text += f"\n\n📚 **Context Available**: I have access to your uploaded documents and can reference them naturally in our conversation."
            
            return help_text
        
        # Check if query might benefit from document context
        if context and self._query_relates_to_context(query, context):
            return self._answer_with_context(query, context)
        
        # Handle specific domain questions with context awareness
        elif any(word in query_lower for word in ['code', 'programming', 'python', 'javascript', 'html', 'software']):
            response = "I can help with programming questions! "
            if context and any(word in context.lower() for word in ['code', 'program', 'script', 'function']):
                response += "I notice you have some technical documents uploaded - I can reference them if they're relevant to your coding question. "
            response += "What programming topic would you like to discuss?"
            return response
        
        # Writing assistance with context
        elif any(word in query_lower for word in ['write', 'writing', 'essay', 'letter', 'email', 'document']):
            response = "I'd be happy to help with writing! "
            if context:
                response += "I can also incorporate information from your uploaded documents if that would be helpful. "
            response += "What kind of writing assistance do you need?"
            return response
        
        # Math with context
        elif any(word in query_lower for word in ['math', 'calculate', 'equation', 'solve', 'statistics']):
            response = "I can help with math problems and calculations! "
            if context and any(word in context.lower() for word in ['data', 'number', 'column', 'records']):
                response += "I see you have some data uploaded - I can help analyze or calculate things from that data too. "
            response += "What mathematical problem are you working on?"
            return response
        
        # Direct questions that might relate to documents
        elif any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'explain', 'tell me about']):
            if context:
                return self._answer_with_context(query, context)
            else:
                return f"I'd be happy to help explain that! Could you provide more details about what specifically you'd like to know regarding: '{query}'?"
        
        # Default contextual response
        else:
            if context:
                return self._answer_with_context(query, context)
            else:
                return self._generate_general_response(query)

    def _query_relates_to_context(self, query: str, context: str) -> bool:
        """Check if the query might be related to the uploaded documents"""
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will'}
        query_words -= common_words
        context_words -= common_words
        
        # Check for word overlap
        overlap = query_words.intersection(context_words)
        return len(overlap) > 0

    def _answer_with_context(self, query: str, context: str) -> str:
        """Generate response incorporating document context"""
        
        # Look for relevant information in the context
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        relevant_context = []
        for part in context.split(' | '):
            if any(word in part.lower() for word in query_words):
                relevant_context.append(part)
        
        # Generate response
        if relevant_context:
            response = f"Based on your uploaded documents and your question about '{query}', here's what I can tell you:\n\n"
            response += "\n".join(f"• {ctx}" for ctx in relevant_context[:3])  # Limit to 3 most relevant
            response += "\n\nWould you like me to elaborate on any of these points or help you explore this topic further?"
        else:
            # Fallback to general response but mention document availability
            response = self._generate_general_response(query)
            response += "\n\n📚 I also have your uploaded documents available if they're relevant to this topic."
        
        return response

    def _handle_general_conversation(self, question: str, filter_result, strategy: FilteringStrategy) -> Dict:
        """Handle general AI conversation when no documents are uploaded"""
        
        # Simple AI responses for common queries
        response = self._generate_general_response(filter_result.filtered_text)
        
        return {
            "status": "success",
            "ai_response": response,
            "query_info": {
                "original_query": question,
                "filtered_query": filter_result.filtered_text,
                "was_filtered": filter_result.is_modified,
                "safety_score": filter_result.safety_score,
                "query_type": "general_conversation"
            },
            "documents_info": {
                "total_documents": 0,
                "document_list": []
            },
            "timestamp": datetime.now().isoformat(),
            "filtering_notice": f"🛡️ Some content was filtered for security (Safety Score: {filter_result.safety_score:.1%})" if filter_result.is_modified else None
        }

    def _generate_general_response(self, query: str) -> str:
        """Generate responses for general conversation"""
        query_lower = query.lower()
        
        # Greeting responses
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm your AI assistant. I can help with general questions and conversations. Upload PDF or CSV files if you'd like to chat about specific documents."
        
        # Help responses
        elif any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
            return """I can help you with:

• **General conversations** - Ask me anything about various topics
• **Document analysis** - Upload PDFs or CSVs and I'll analyze them for you
• **Data insights** - I can provide summaries and answer questions about your documents
• **Safe interactions** - I have built-in security filtering to ensure helpful, appropriate responses

Try uploading a document to get started with document-specific conversations!"""
        
        # Technical questions
        elif any(word in query_lower for word in ['code', 'programming', 'python', 'javascript', 'html']):
            return "I can help with programming questions! Feel free to ask about coding concepts, debugging, best practices, or specific programming languages. What would you like to know?"
        
        # Writing assistance
        elif any(word in query_lower for word in ['write', 'writing', 'essay', 'letter', 'email']):
            return "I'd be happy to help with writing! I can assist with essays, emails, creative writing, editing, or any other writing tasks. What kind of writing help do you need?"
        
        # Math/calculations
        elif any(word in query_lower for word in ['math', 'calculate', 'equation', 'solve']):
            return "I can help with math problems and calculations! Feel free to ask about algebra, calculus, statistics, or any mathematical concepts you're working with."
        
        # Default response
        else:
            return f"I understand you're asking about: '{query}'. I'm here to help! Could you provide more details about what specific information or assistance you're looking for?"

    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['summary', 'summarize', 'overview', 'what is this', 'what are these']):
            return "summary"
        elif any(word in query_lower for word in ['analyze', 'analysis', 'insights', 'patterns', 'trends']):
            return "analysis"
        elif any(word in query_lower for word in ['find', 'search', 'look for', 'where', 'when', 'who']):
            return "specific"
        else:
            return "general"

    def _handle_summary_query(self, doc_query: DocumentQuery) -> str:
        """Handle summary requests"""
        summaries = []
        
        for filename in self.documents.keys():
            doc_info = self.documents[filename]
            if doc_info.error:
                summaries.append(f"❌ {filename}: Error - {doc_info.error}")
            else:
                summaries.append(f"📄 {filename}: {doc_info.summary}")
        
        return "Here's a summary of your uploaded documents:\n\n" + "\n\n".join(summaries)

    def _handle_specific_query(self, doc_query: DocumentQuery) -> str:
        """Handle specific search queries"""
        query = doc_query.filtered_query.lower()
        results = []
        
        for filename in self.documents.keys():
            if filename in self.document_texts:
                text = self.document_texts[filename]
                # Simple keyword search
                relevant_lines = []
                for line in text.split('\n'):
                    if any(word in line.lower() for word in query.split() if len(word) > 2):
                        relevant_lines.append(line.strip())
                        if len(relevant_lines) >= 5:  # Limit results
                            break
                
                if relevant_lines:
                    results.append(f"📄 {filename}:\n" + "\n".join(relevant_lines[:3]))
        
        if results:
            return "Found relevant information:\n\n" + "\n\n".join(results)
        else:
            return "I couldn't find specific information matching your query in the uploaded documents."

    def _handle_analysis_query(self, doc_query: DocumentQuery) -> str:
        """Handle analysis requests"""
        analyses = []
        
        for filename in self.documents.keys():
            doc_info = self.documents[filename]
            
            if filename in self.document_dataframes:
                df = self.document_dataframes[filename]
                analysis = self._analyze_dataframe(df, doc_query.filtered_query)
                analyses.append(f"📊 {filename}:\n{analysis}")
            
            elif filename in self.document_texts:
                text = self.document_texts[filename]
                analysis = self._analyze_text(text, doc_query.filtered_query)
                analyses.append(f"📄 {filename}:\n{analysis}")
        
        if analyses:
            return "Analysis results:\n\n" + "\n\n".join(analyses)
        else:
            return "I can provide basic analysis. For detailed analysis, please ask specific questions about the data."

    def _handle_general_document_query(self, doc_query: DocumentQuery) -> str:
        """Handle general queries about documents"""
        return f"I have {len(self.documents)} document(s) available. You can ask me to summarize them, search for specific information, or analyze the data. What would you like to know?"

    def _analyze_dataframe(self, df: pd.DataFrame, query: str) -> str:
        """Basic DataFrame analysis"""
        analysis_parts = []
        
        # Basic info
        analysis_parts.append(f"📊 Dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # Column info
        analysis_parts.append(f"📋 Columns: {', '.join(df.columns.tolist())}")
        
        # Missing data
        missing_data = df.isnull().sum()
        if missing_data.any():
            analysis_parts.append(f"⚠️  Missing data: {missing_data[missing_data > 0].to_dict()}")
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_parts.append(f"📈 Numeric columns: {len(numeric_cols)}")
            for col in numeric_cols[:3]:  # Show first 3
                stats = df[col].describe()
                analysis_parts.append(f"   {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        
        return "\n".join(analysis_parts)

    def _analyze_text(self, text: str, query: str) -> str:
        """Basic text analysis"""
        words = text.split()
        lines = text.split('\n')
        
        return f"📄 Text overview: {len(words)} words, {len(lines)} lines. Use specific search terms to find relevant information."

    def _create_response(self, doc_query: DocumentQuery, ai_response: str, status: str) -> Dict:
        """Create formatted response"""
        response = {
            "status": status,
            "ai_response": ai_response,
            "query_info": {
                "original_query": doc_query.original_query,
                "filtered_query": doc_query.filtered_query,
                "was_filtered": doc_query.is_filtered,
                "safety_score": doc_query.safety_score,
                "query_type": doc_query.query_type
            },
            "documents_info": {
                "total_documents": len(self.documents),
                "document_list": [
                    {
                        "filename": filename,
                        "type": info.file_type,
                        "status": "error" if info.error else "ready"
                    }
                    for filename, info in self.documents.items()
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add filtering notice if content was filtered
        if doc_query.is_filtered:
            response["filtering_notice"] = f"🛡️ Some content was filtered for security (Safety Score: {doc_query.safety_score:.1%})"
        
        return response

    def get_document_info(self) -> Dict:
        """Get information about all uploaded documents"""
        return {
            "total_documents": len(self.documents),
            "documents": [
                {
                    "filename": filename,
                    "type": info.file_type,
                    "size_mb": info.size_bytes / (1024 * 1024),
                    "pages": info.page_count,
                    "rows": info.row_count,
                    "columns": info.column_count,
                    "status": "error" if info.error else "ready",
                    "summary": info.summary
                }
                for filename, info in self.documents.items()
            ]
        }

if __name__ == "__main__":
    # Test the chatbot
    chatbot = SmartDocumentChatbot()
    
    print("\n" + "="*50)
    print("🤖 Document Chatbot Test")
    print("="*50)
    
    # Test general conversation
    result = chatbot.ask_question("Hello! What can you help me with?")
    print(f"AI: {result['ai_response']}")
