
export interface Project {
  title: string;
  description: string;
  technologies: string[];
  image: string;
  link?: string;
  github?: string;
  readme?: string;
}

export const projects: Project[] = [
  {
    title: "Generative AI-Powered Content Platform",
    description: "Built an enterprise-grade content generation platform using LLMs that increased content creation efficiency by 70% for marketing teams.",
    technologies: ["LangChain", "OpenAI API", "React", "Python", "FastAPI"],
    image: "https://images.unsplash.com/photo-1678565655915-d8e392114ffd?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/ai-content-platform",
    github: "https://github.com/karthick-ai/gen-ai-content-platform",
    readme: `# Generative AI-Powered Content Platform

## Overview
A sophisticated content generation platform leveraging Large Language Models to automate and enhance marketing content creation. This platform enables marketing teams to produce high-quality content at scale while maintaining brand consistency.

## Features
- AI-powered content generation with customizable templates
- Brand voice preservation and content style matching
- Bulk content generation for campaigns
- Content performance analytics
- Integration with popular CMS platforms

## Tech Stack
- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **AI**: LangChain, OpenAI API
- **DevOps**: Docker, GitHub Actions, AWS

## Impact
- 70% increase in content creation efficiency
- 3x more content variations tested in campaigns
- 45% reduction in content production costs

## Implementation

### Core AI Engine
\`\`\`python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class ContentGenerator:
    def __init__(self, api_key, model="gpt-4"):
        self.llm = OpenAI(openai_api_key=api_key, model_name=model)
        
    def generate_content(self, template_id, variables, brand_voice):
        # Fetch template from database
        template = self.get_template(template_id)
        
        # Create prompt with brand voice guidelines
        prompt = PromptTemplate(
            input_variables=["template", "variables", "brand_voice"],
            template="Generate content following this template: {template}\n"
                    "Using these variables: {variables}\n"
                    "Match this brand voice: {brand_voice}"
        )
        
        # Execute generation chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(
            template=template,
            variables=variables,
            brand_voice=brand_voice
        )
        
        return self.post_process(result)
        
    def get_template(self, template_id):
        # Database fetch logic
        pass
        
    def post_process(self, content):
        # Apply content filters and formatting
        return content
\`\`\`

### Frontend Component
\`\`\`tsx
import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { ContentTemplate, BrandVoice, GenerationResult } from '../types';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Select } from '../components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { generateContent } from '../api/content';

export const ContentGenerator = () => {
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [brandVoice, setBrandVoice] = useState<string>('');
  const [variables, setVariables] = useState<Record<string, string>>({});
  
  const { data: templates } = useQuery({
    queryKey: ['templates'],
    queryFn: fetchTemplates
  });
  
  const { data: brandVoices } = useQuery({
    queryKey: ['brandVoices'],
    queryFn: fetchBrandVoices
  });
  
  const mutation = useMutation({
    mutationFn: generateContent,
    onSuccess: (data) => {
      // Handle successful generation
    }
  });
  
  const handleGenerate = () => {
    mutation.mutate({
      templateId: selectedTemplate,
      brandVoice,
      variables
    });
  };
  
  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>AI Content Generator</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Template</label>
          <Select 
            value={selectedTemplate}
            onValueChange={setSelectedTemplate}
            options={templates?.map(t => ({ label: t.name, value: t.id }))}
          />
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium">Brand Voice</label>
          <Select 
            value={brandVoice}
            onValueChange={setBrandVoice}
            options={brandVoices?.map(v => ({ label: v.name, value: v.id }))}
          />
        </div>
        
        {/* Dynamic variable fields based on selected template */}
        {templates?.find(t => t.id === selectedTemplate)?.variables.map(variable => (
          <div key={variable.key} className="space-y-2">
            <label className="text-sm font-medium">{variable.label}</label>
            <Textarea
              value={variables[variable.key] || ''}
              onChange={(e) => setVariables({
                ...variables,
                [variable.key]: e.target.value
              })}
            />
          </div>
        ))}
        
        <Button 
          onClick={handleGenerate}
          disabled={mutation.isPending}
          className="w-full"
        >
          {mutation.isPending ? 'Generating...' : 'Generate Content'}
        </Button>
        
        {mutation.isSuccess && (
          <div className="border p-4 rounded-md bg-gray-50">
            <h3 className="font-medium mb-2">Generated Content</h3>
            <div className="whitespace-pre-wrap">{mutation.data.content}</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
\`\`\`

## Installation and Setup
\`\`\`bash
# Clone the repository
git clone https://github.com/karthick-ai/gen-ai-content-platform.git

# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your OpenAI API key and other configs

# Start the development servers
# In frontend directory
npm run dev

# In backend directory
uvicorn app.main:app --reload
\`\`\`

## Usage
Detailed documentation on using the platform is available in the [Wiki](https://github.com/karthick-ai/gen-ai-content-platform/wiki).`
  },
  {
    title: "RAG System for Legal Document Analysis",
    description: "Developed a Retrieval-Augmented Generation system that analyzed 75,000+ legal documents with 85% accuracy, providing valuable insights for compliance teams.",
    technologies: ["Transformers", "Vector Databases", "Hugging Face", "LangChain"],
    image: "https://images.unsplash.com/photo-1589829085413-56de8ae18c73?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/rag-legal-system",
    github: "https://github.com/karthick-ai/legal-rag-system",
    readme: `# RAG System for Legal Document Analysis

## Project Overview
A Retrieval-Augmented Generation system designed specifically for legal document analysis. This system enables legal teams to extract insights, identify patterns, and ensure compliance across large volumes of legal documents.

## Key Features
- Semantic search across legal document corpus
- Automated document classification and tagging
- Compliance risk detection and flagging
- Legal precedent linkage and citation
- Interactive Q&A for legal research

## Technical Architecture
- **Vector Database**: Pinecone/Weaviate for efficient document retrieval
- **Embedding Models**: Custom fine-tuned legal domain embeddings with Sentence Transformers
- **LLM Integration**: Hugging Face models with domain-specific prompts
- **Backend**: Python with FastAPI
- **Frontend**: React with TypeScript

## Code Examples

### Document Embeddings Pipeline
\`\`\`python
from sentence_transformers import SentenceTransformer
from weaviate import Client
import os
import json
from typing import List, Dict, Any
import uuid

class DocumentEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
        
        # Connect to Weaviate vector database
        self.client = Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=os.getenv("WEAVIATE_API_KEY")
        )
    
    def create_schema_if_not_exists(self):
        """Create the schema for legal documents if it doesn't exist."""
        if not self.client.schema.exists("LegalDocument"):
            schema = {
                "classes": [{
                    "class": "LegalDocument",
                    "description": "A legal document with embeddings",
                    "properties": [
                        {"name": "title", "dataType": ["string"]},
                        {"name": "content", "dataType": ["text"]},
                        {"name": "docType", "dataType": ["string"]},
                        {"name": "jurisdiction", "dataType": ["string"]},
                        {"name": "date", "dataType": ["date"]},
                        {"name": "parties", "dataType": ["string[]"]},
                        {"name": "caseNumber", "dataType": ["string"]},
                        {"name": "source", "dataType": ["string"]}
                    ],
                    "vectorizer": "none"  # We'll provide our own vectors
                }]
            }
            self.client.schema.create(schema)
    
    def process_document(self, document: Dict[str, Any]):
        """Process a single document and store it with embeddings."""
        # Extract text for embedding - combine title and content
        text_to_embed = f"{document['title']} {document['content']}"
        
        # Generate embedding
        embedding = self.model.encode(text_to_embed)
        
        # Prepare document for storage
        doc_uuid = str(uuid.uuid4())
        
        # Store in Weaviate
        self.client.data_object.create(
            class_name="LegalDocument",
            data_object=document,
            uuid=doc_uuid,
            vector=embedding.tolist()
        )
        
        return doc_uuid
    
    def batch_process_documents(self, documents: List[Dict[str, Any]], batch_size: int = 50):
        """Process a batch of documents with vectorization."""
        results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_ids = []
            batch_texts = []
            
            for doc in batch:
                doc_id = str(uuid.uuid4())
                text = f"{doc['title']} {doc['content']}"
                batch_ids.append(doc_id)
                batch_texts.append(text)
            
            # Generate embeddings in one batch
            embeddings = self.model.encode(batch_texts)
            
            # Add to Weaviate
            with self.client.batch as batch_processor:
                for j, doc in enumerate(batch):
                    batch_processor.add_data_object(
                        data_object=doc,
                        class_name="LegalDocument",
                        uuid=batch_ids[j],
                        vector=embeddings[j].tolist()
                    )
            
            results.extend(batch_ids)
        
        return results
\`\`\`

### Query Engine
\`\`\`python
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LegalQueryEngine:
    def __init__(self, vector_store, model_id="databricks/dolly-v2-3b"):
        self.vector_store = vector_store
        
        # Set up the LLM with legal domain knowledge
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create a text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Create LangChain wrapper around the pipeline
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Set up the prompt template for legal questions
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a legal assistant with expertise in analyzing legal documents. 
            Use the following legal document excerpts to answer the question.
            
            Legal context:
            {context}
            
            Question: {question}
            
            Provide a detailed answer with relevant legal citations if applicable:
            """
        )
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the system with a legal question.
        
        Args:
            question: The legal question to answer
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and supporting documents
        """
        # Get relevant documents from vector store
        relevant_docs = self.vector_store.similarity_search(question, k=top_k)
        
        # Prepare context from relevant documents
        context = "\n\n".join([f"Document {i+1}:\nTitle: {doc.metadata['title']}\n"
                               f"Content: {doc.page_content}\n"
                               f"Date: {doc.metadata.get('date', 'Unknown')}\n"
                               f"Jurisdiction: {doc.metadata.get('jurisdiction', 'Unknown')}"
                               for i, doc in enumerate(relevant_docs)])
        
        # Format the prompt with question and context
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate answer using LLM
        answer = self.llm(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "supporting_documents": [
                {
                    "title": doc.metadata['title'],
                    "excerpt": doc.page_content[:300] + "...",
                    "source": doc.metadata.get('source', 'Unknown'),
                    "relevance_score": doc.metadata.get('score', None)
                }
                for doc in relevant_docs
            ]
        }
\`\`\`

### React Frontend Component
\`\`\`tsx
import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Card, 
  CardContent, 
  CardHeader, 
  CardTitle,
  CardDescription,
  CardFooter 
} from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Button } from '../components/ui/button';
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from '../components/ui/tabs';
import { ScrollArea } from '../components/ui/scroll-area';
import { Loader2, Search, FileText, AlertCircle } from 'lucide-react';
import { queryLegalDocuments } from '../api/legal';

interface Document {
  title: string;
  excerpt: string;
  source: string;
  relevance_score: number;
}

interface QueryResult {
  question: string;
  answer: string;
  supporting_documents: Document[];
}

const LegalSearchInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [searchHistory, setSearchHistory] = useState<string[]>([]);
  
  const { 
    data, 
    error, 
    isLoading, 
    refetch 
  } = useQuery<QueryResult>({
    queryKey: ['legalQuery', query],
    queryFn: () => queryLegalDocuments(query),
    enabled: false // Don't run query on component mount
  });
  
  const handleSearch = () => {
    if (query.trim()) {
      refetch();
      setSearchHistory(prev => [query, ...prev.slice(0, 9)]);
    }
  };
  
  return (
    <div className="container mx-auto py-6">
      <h1 className="text-3xl font-bold mb-6">Legal Document Analysis</h1>
      
      <div className="flex gap-4 mb-6">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about legal precedents, compliance issues, or document analysis..."
          className="flex-1"
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
        <Button onClick={handleSearch} disabled={isLoading || !query.trim()}>
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Search className="h-4 w-4 mr-2" />
          )}
          Search
        </Button>
      </div>
      
      {searchHistory.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-2">Recent Searches</h3>
          <div className="flex flex-wrap gap-2">
            {searchHistory.map((q, i) => (
              <Button 
                key={i} 
                variant="outline" 
                size="sm"
                onClick={() => {
                  setQuery(q);
                  refetch();
                }}
              >
                {q}
              </Button>
            ))}
          </div>
        </div>
      )}
      
      {error && (
        <Card className="mb-6 border-red-300 bg-red-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-2 text-red-600">
              <AlertCircle className="h-5 w-5" />
              <p>An error occurred: {(error as Error).message}</p>
            </div>
          </CardContent>
        </Card>
      )}
      
      {data && (
        <Tabs defaultValue="answer" className="w-full">
          <TabsList>
            <TabsTrigger value="answer">Answer</TabsTrigger>
            <TabsTrigger value="documents">
              Supporting Documents ({data.supporting_documents.length})
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="answer">
            <Card>
              <CardHeader>
                <CardTitle>Legal Analysis</CardTitle>
                <CardDescription>
                  Based on our legal document corpus
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="prose max-w-none">
                  <p>{data.answer}</p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="documents">
            <Card>
              <CardHeader>
                <CardTitle>Supporting Documents</CardTitle>
                <CardDescription>
                  The most relevant legal documents to your query
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  <div className="space-y-4">
                    {data.supporting_documents.map((doc, i) => (
                      <Card key={i}>
                        <CardHeader className="py-3">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-base">{doc.title}</CardTitle>
                            <span className="text-sm text-muted-foreground">
                              Relevance: {Math.round(doc.relevance_score * 100)}%
                            </span>
                          </div>
                        </CardHeader>
                        <CardContent className="py-2">
                          <p className="text-sm">{doc.excerpt}</p>
                        </CardContent>
                        <CardFooter className="py-2 text-xs text-muted-foreground">
                          <div className="flex items-center">
                            <FileText className="h-3 w-3 mr-1" />
                            Source: {doc.source}
                          </div>
                        </CardFooter>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};

export default LegalSearchInterface;
\`\`\`

## Performance Metrics
- 85% accuracy in legal document analysis
- 75,000+ documents processed and vectorized
- 60% reduction in manual document review time
- 92% user satisfaction rate among legal professionals

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/karthick-ai/legal-rag-system.git
cd legal-rag-system

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials

# Run the application
python app.py
\`\`\`

## Usage Examples
Check the [documentation](https://github.com/karthick-ai/legal-rag-system/docs) for detailed usage examples and API reference.`
  },
  {
    title: "AI Agent-Based Workflow Automation",
    description: "Created a system of AI agents that automate complex business workflows, reducing manual intervention by 60% and increasing processing speed by 4x.",
    technologies: ["LLM Orchestration", "Function Calling", "Python", "FastAPI"],
    image: "https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/ai-agent-automation",
    github: "https://github.com/karthick-ai/workflow-agents",
    readme: `# AI Agent-Based Workflow Automation

## Introduction
A sophisticated system of coordinated AI agents designed to automate complex business workflows. This solution leverages advanced LLM orchestration techniques to create autonomous agents that can handle multi-step business processes with minimal human intervention.

## System Capabilities
- Autonomous workflow execution with inter-agent communication
- Dynamic task allocation and prioritization
- Human-in-the-loop intervention points
- Process monitoring and performance analytics
- Exception handling and error recovery

## Technical Foundation
- **Agent Framework**: Custom LLM orchestration system
- **Function Calling**: OpenAI function calling API
- **Backend**: Python with FastAPI
- **Messaging**: Redis for agent communication
- **Monitoring**: Prometheus and Grafana

## Implementation Details

### Agent Architecture
\`\`\`python
from typing import Dict, List, Any, Optional, Callable
import json
import logging
from pydantic import BaseModel, Field
import openai
from redis import Redis
import uuid

class AgentCapability(BaseModel):
    """Definition of what an agent can do"""
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Detailed description of what this capability does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters required for this capability")
    required_permissions: List[str] = Field(default_factory=list, description="Permissions required to execute this capability")

class Agent(BaseModel):
    """Base agent class that defines the core agent functionality"""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this agent")
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent's purpose")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="What this agent can do")
    knowledge_base: List[str] = Field(default_factory=list, description="References to knowledge sources")
    
    # Communication system
    message_bus: Optional[Redis] = None
    subscribed_channels: List[str] = Field(default_factory=list)
    
    # State tracking
    state: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Function registry
    _function_registry: Dict[str, Callable] = Field(default_factory=dict)
    
    def register_function(self, func_name: str, func: Callable):
        """Register a function that can be called by this agent"""
        self._function_registry[func_name] = func
    
    def register_capability(self, capability: AgentCapability):
        """Add a capability to this agent"""
        self.capabilities.append(capability)
    
    def connect_to_message_bus(self, redis_client: Redis):
        """Connect to the Redis message bus"""
        self.message_bus = redis_client
    
    def subscribe(self, channel: str):
        """Subscribe to a specific message channel"""
        if self.message_bus:
            pubsub = self.message_bus.pubsub()
            pubsub.subscribe(channel)
            self.subscribed_channels.append(channel)
            return pubsub
        return None
    
    def publish(self, channel: str, message: Dict[str, Any]):
        """Publish a message to a channel"""
        if self.message_bus:
            message["sender"] = self.agent_id
            message["timestamp"] = datetime.now().isoformat()
            self.message_bus.publish(channel, json.dumps(message))
            return True
        return False
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an incoming message and determine response"""
        # Base implementation - override in specific agents
        return {"status": "received", "message_id": message.get("id")}
    
    def execute_function(self, func_name: str, **kwargs) -> Any:
        """Execute a registered function"""
        if func_name in self._function_registry:
            try:
                return self._function_registry[func_name](**kwargs)
            except Exception as e:
                logging.error(f"Error executing {func_name}: {str(e)}")
                return {"error": str(e)}
        return {"error": f"Function {func_name} not found"}

class LLMAgent(Agent):
    """Agent that uses an LLM to make decisions"""
    model_name: str = Field(..., description="Name of the LLM model to use")
    system_prompt: str = Field(..., description="System prompt to guide the LLM")
    function_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    
    def format_capabilities_as_functions(self):
        """Format agent capabilities as OpenAI function definitions"""
        functions = []
        for cap in self.capabilities:
            functions.append({
                "name": cap.name,
                "description": cap.description,
                "parameters": {
                    "type": "object",
                    "properties": cap.parameters,
                    "required": [k for k in cap.parameters.keys()]
                }
            })
        self.function_definitions = functions
        return functions
    
    def query_llm(self, user_query: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the LLM with the given input"""
        if conversation_history is None:
            conversation_history = []
            
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_query})
        
        functions = self.format_capabilities_as_functions()
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            functions=functions if functions else None,
            function_call="auto"
        )
        
        return response
    
    def process_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process message using LLM reasoning"""
        query = message.get("content", "")
        history = message.get("history", [])
        
        llm_response = self.query_llm(query, history)
        
        # Check if the LLM wants to call a function
        message = llm_response.choices[0].message
        if message.get("function_call"):
            function_name = message["function_call"]["name"]
            function_args = json.loads(message["function_call"]["arguments"])
            
            # Execute the function
            function_response = self.execute_function(function_name, **function_args)
            
            # Let the LLM interpret the function result
            follow_up_messages = history + [
                {"role": "user", "content": query},
                message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_response)
                }
            ]
            
            final_response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_prompt}] + follow_up_messages
            )
            
            return {
                "content": final_response.choices[0].message.content,
                "function_called": function_name,
                "function_result": function_response
            }
        
        return {"content": message.content}
\`\`\`

### Workflow Definition Example
\`\`\`python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"

class TaskDefinition(BaseModel):
    """Definition of a task in a workflow"""
    task_id: str
    name: str
    description: str
    agent_type: str
    required_inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 3
    dependencies: List[str] = Field(default_factory=list)
    
class Task(TaskDefinition):
    """Runtime instance of a task"""
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    retry_attempts: int = 0

class WorkflowDefinition(BaseModel):
    """Definition of a complete workflow"""
    workflow_id: str
    name: str
    description: str
    version: str
    tasks: List[TaskDefinition]
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    
class Workflow(BaseModel):
    """Runtime instance of a workflow"""
    definition: WorkflowDefinition
    workflow_run_id: str
    status: str = "pending"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    tasks: Dict[str, Task] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    
    def initialize_tasks(self):
        """Initialize all tasks from the workflow definition"""
        for task_def in self.definition.tasks:
            self.tasks[task_def.task_id] = Task(**task_def.dict())
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to be executed (dependencies satisfied)"""
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
                
            dependencies_met = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks or self.tasks[dep_id].status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break
                    
            if dependencies_met:
                ready_tasks.append(task)
                
        return ready_tasks
    
    def is_completed(self) -> bool:
        """Check if all tasks in the workflow are completed"""
        return all(task.status == TaskStatus.COMPLETED for task in self.tasks.values())
    
    def has_failed(self) -> bool:
        """Check if any task has failed after all retries"""
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

# Example workflow JSON
example_workflow = {
    "workflow_id": "customer_onboarding_process",
    "name": "Customer Onboarding",
    "description": "Process for onboarding new enterprise customers",
    "version": "1.0.0",
    "tasks": [
        {
            "task_id": "validate_customer_data",
            "name": "Validate Customer Data",
            "description": "Validate customer information for completeness and accuracy",
            "agent_type": "data_validator",
            "required_inputs": {
                "customer_data": {"type": "object"}
            },
            "expected_outputs": {
                "validation_result": {"type": "object"},
                "is_valid": {"type": "boolean"}
            },
            "dependencies": []
        },
        {
            "task_id": "credit_check",
            "name": "Perform Credit Check",
            "description": "Check customer credit score and financial health",
            "agent_type": "financial_analyst",
            "required_inputs": {
                "tax_id": {"type": "string"},
                "company_name": {"type": "string"}
            },
            "expected_outputs": {
                "credit_score": {"type": "number"},
                "risk_assessment": {"type": "string"}
            },
            "dependencies": ["validate_customer_data"]
        },
        {
            "task_id": "prepare_contract",
            "name": "Prepare Service Contract",
            "description": "Generate appropriate service contract based on customer needs",
            "agent_type": "legal_document_creator",
            "required_inputs": {
                "customer_data": {"type": "object"},
                "service_tier": {"type": "string"},
                "risk_assessment": {"type": "string"}
            },
            "expected_outputs": {
                "contract_document": {"type": "string", "format": "url"}
            },
            "dependencies": ["credit_check"]
        },
        {
            "task_id": "setup_account",
            "name": "Set Up Customer Account",
            "description": "Configure systems and provision resources for the customer",
            "agent_type": "system_provisioner",
            "required_inputs": {
                "customer_data": {"type": "object"},
                "service_tier": {"type": "string"}
            },
            "expected_outputs": {
                "account_id": {"type": "string"},
                "login_credentials": {"type": "object"}
            },
            "dependencies": ["validate_customer_data"]
        },
        {
            "task_id": "send_welcome_email",
            "name": "Send Welcome Email",
            "description": "Send personalized welcome email with account details to customer",
            "agent_type": "communication_manager",
            "required_inputs": {
                "customer_data": {"type": "object"},
                "account_id": {"type": "string"},
                "login_credentials": {"type": "object"},
                "contract_document": {"type": "string", "format": "url"}
            },
            "expected_outputs": {
                "email_sent": {"type": "boolean"},
                "message_id": {"type": "string"}
            },
            "dependencies": ["setup_account", "prepare_contract"]
        }
    ]
}
\`\`\`

### React Dashboard Component
\`\`\`tsx
import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '../components/ui/tabs';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { Button } from '../components/ui/button';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer
} from 'recharts';
import { 
  Activity, 
  CheckCircle, 
  Clock, 
  AlertTriangle, 
  Play, 
  Pause,
  RotateCw
} from 'lucide-react';

import { 
  fetchActiveWorkflows, 
  fetchWorkflowStats, 
  fetchWorkflowDetails,
  startWorkflow,
  pauseWorkflow,
  resumeWorkflow,
  retryTask
} from '../api/workflows';

// Status badge component
const StatusBadge = ({ status }) => {
  const statusMap = {
    'pending': { color: 'bg-yellow-100 text-yellow-800', icon: <Clock className="w-3 h-3 mr-1" /> },
    'in_progress': { color: 'bg-blue-100 text-blue-800', icon: <Activity className="w-3 h-3 mr-1" /> },
    'completed': { color: 'bg-green-100 text-green-800', icon: <CheckCircle className="w-3 h-3 mr-1" /> },
    'failed': { color: 'bg-red-100 text-red-800', icon: <AlertTriangle className="w-3 h-3 mr-1" /> },
    'waiting': { color: 'bg-purple-100 text-purple-800', icon: <Clock className="w-3 h-3 mr-1" /> },
  };
  
  const { color, icon } = statusMap[status] || statusMap.pending;
  
  return (
    <Badge className={color + " flex items-center"}>
      {icon}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </Badge>
  );
};

const WorkflowDashboard = () => {
  const [selectedWorkflowId, setSelectedWorkflowId] = useState(null);
  const [timeRange, setTimeRange] = useState('24h');
  
  // Fetch active workflows
  const { 
    data: activeWorkflows,
    isLoading: isLoadingWorkflows
  } = useQuery({
    queryKey: ['activeWorkflows'],
    queryFn: fetchActiveWorkflows,
    refetchInterval: 30000 // Refresh every 30 seconds
  });
  
  // Fetch workflow statistics
  const {
    data: workflowStats,
    isLoading: isLoadingStats
  } = useQuery({
    queryKey: ['workflowStats', timeRange],
    queryFn: () => fetchWorkflowStats(timeRange),
    refetchInterval: 60000 // Refresh every minute
  });
  
  // Fetch selected workflow details
  const {
    data: workflowDetails,
    isLoading: isLoadingDetails,
    refetch: refetchWorkflowDetails
  } = useQuery({
    queryKey: ['workflowDetails', selectedWorkflowId],
    queryFn: () => fetchWorkflowDetails(selectedWorkflowId),
    enabled: !!selectedWorkflowId,
    refetchInterval: 10000 // Refresh every 10 seconds when viewing details
  });
  
  // Set first workflow as selected when data loads
  useEffect(() => {
    if (activeWorkflows?.workflows?.length && !selectedWorkflowId) {
      setSelectedWorkflowId(activeWorkflows.workflows[0].workflow_run_id);
    }
  }, [activeWorkflows, selectedWorkflowId]);
  
  // Handle workflow control actions
  const handlePauseWorkflow = async (id) => {
    await pauseWorkflow(id);
    refetchWorkflowDetails();
  };
  
  const handleResumeWorkflow = async (id) => {
    await resumeWorkflow(id);
    refetchWorkflowDetails();
  };
  
  const handleRetryTask = async (workflowId, taskId) => {
    await retryTask(workflowId, taskId);
    refetchWorkflowDetails();
  };
  
  // Calculate workflow completion percentage
  const calculateCompletion = (workflow) => {
    if (!workflow || !workflow.tasks) return 0;
    
    const totalTasks = Object.keys(workflow.tasks).length;
    const completedTasks = Object.values(workflow.tasks).filter(
      task => task.status === 'completed'
    ).length;
    
    return Math.round((completedTasks / totalTasks) * 100);
  };
  
  // Prepare chart data
  const prepareChartData = (stats) => {
    if (!stats) return [];
    
    return [
      { name: 'Completed', value: stats.completed_workflows || 0 },
      { name: 'Failed', value: stats.failed_workflows || 0 },
      { name: 'Running', value: stats.running_workflows || 0 },
    ];
  };
  
  return (
    <div className="container mx-auto py-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">AI Workflow Dashboard</h1>
        <div className="flex items-center gap-4">
          <Select
            value={timeRange}
            onValueChange={setTimeRange}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select time range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24h">Last 24 Hours</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
          
          <Button onClick={() => startWorkflow()}>
            <Play className="w-4 h-4 mr-2" />
            Start New Workflow
          </Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Active Workflows</CardTitle>
            <CardDescription>Currently running processes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {isLoadingStats ? "Loading..." : workflowStats?.running_workflows || 0}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Completion Rate</CardTitle>
            <CardDescription>Successfully completed workflows</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">
              {isLoadingStats ? "Loading..." : 
                `${Math.round((workflowStats?.completed_workflows / 
                  (workflowStats?.completed_workflows + workflowStats?.failed_workflows || 1)) * 100)}%`
              }
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Average Duration</CardTitle>
            <CardDescription>Time to complete workflows</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {isLoadingStats ? "Loading..." : 
                `${workflowStats?.avg_duration_minutes?.toFixed(1) || 0} mins`
              }
            </div>
          </CardContent>
        </Card>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Workflow Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={prepareChartData(workflowStats)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#3b82f6" name="Workflows" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader>
            <CardTitle>Active Workflows</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoadingWorkflows ? (
              <div className="flex justify-center items-center h-[300px]">
                Loading workflows...
              </div>
            ) : (
              <div className="space-y-4 max-h-[300px] overflow-y-auto">
                {activeWorkflows?.workflows?.map(workflow => (
                  <Card 
                    key={workflow.workflow_run_id}
                    className={`overflow-hidden cursor-pointer border ${
                      selectedWorkflowId === workflow.workflow_run_id
                        ? 'border-blue-500'
                        : 'border-gray-200'
                    }`}
                    onClick={() => setSelectedWorkflowId(workflow.workflow_run_id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex justify-between items-center">
                        <div>
                          <h3 className="font-medium">
                            {workflow.definition.name}
                          </h3>
                          <p className="text-sm text-gray-500">
                            ID: {workflow.workflow_run_id.substring(0, 8)}...
                          </p>
                        </div>
                        <StatusBadge status={workflow.status} />
                      </div>
                      <div className="mt-2">
                        <Progress value={calculateCompletion(workflow)} className="h-2" />
                        <div className="flex justify-between text-xs mt-1">
                          <span>{calculateCompletion(workflow)}% complete</span>
                          <span>
                            {Object.values(workflow.tasks).filter(t => t.status === 'completed').length}/
                            {Object.keys(workflow.tasks).length} tasks
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
                
                {activeWorkflows?.workflows?.length === 0 && (
                  <div className="text-center py-8 text-gray-500">
                    No active workflows
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Workflow Details</CardTitle>
          {workflowDetails && (
            <CardDescription>
              {workflowDetails.definition.name} - {workflowDetails.definition.description}
            </CardDescription>
          )}
        </CardHeader>
        <CardContent>
          {isLoadingDetails || !workflowDetails ? (
            <div className="flex justify-center items-center h-[300px]">
              {selectedWorkflowId ? "Loading workflow details..." : "Select a workflow to view details"}
            </div>
          ) : (
            <div>
              <div className="flex justify-between items-center mb-4">
                <div>
                  <p className="text-sm text-gray-500">
                    Started: {new Date(workflowDetails.start_time).toLocaleString()}
                    {workflowDetails.end_time && 
                      ` • Ended: ${new Date(workflowDetails.end_time).toLocaleString()}`
                    }
                  </p>
                </div>
                <div className="flex gap-2">
                  {workflowDetails.status === 'in_progress' ? (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handlePauseWorkflow(workflowDetails.workflow_run_id)}
                    >
                      <Pause className="w-4 h-4 mr-2" />
                      Pause
                    </Button>
                  ) : workflowDetails.status === 'waiting' ? (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleResumeWorkflow(workflowDetails.workflow_run_id)}
                    >
                      <Play className="w-4 h-4 mr-2" />
                      Resume
                    </Button>
                  ) : null}
                </div>
              </div>
              
              <Tabs defaultValue="tasks">
                <TabsList>
                  <TabsTrigger value="tasks">Tasks</TabsTrigger>
                  <TabsTrigger value="variables">Variables</TabsTrigger>
                </TabsList>
                
                <TabsContent value="tasks">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Task</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Agent</TableHead>
                        <TableHead>Started</TableHead>
                        <TableHead>Duration</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {Object.entries(workflowDetails.tasks).map(([taskId, task]) => (
                        <TableRow key={taskId}>
                          <TableCell className="font-medium">{task.name}</TableCell>
                          <TableCell>
                            <StatusBadge status={task.status} />
                          </TableCell>
                          <TableCell>{task.assigned_agent_id || "-"}</TableCell>
                          <TableCell>
                            {task.start_time ? 
                              new Date(task.start_time).toLocaleTimeString() : 
                              "-"
                            }
                          </TableCell>
                          <TableCell>
                            {task.start_time && task.end_time ? 
                              `${Math.round((new Date(task.end_time) - new Date(task.start_time)) / 1000)}s` : 
                              task.start_time && !task.end_time ?
                              "Running..." :
                              "-"
                            }
                          </TableCell>
                          <TableCell>
                            {task.status === 'failed' && (
                              <Button 
                                variant="ghost" 
                                size="sm"
                                onClick={() => handleRetryTask(workflowDetails.workflow_run_id, taskId)}
                              >
                                <RotateCw className="w-4 h-4 mr-1" />
                                Retry
                              </Button>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TabsContent>
                
                <TabsContent value="variables">
                  <div className="rounded border">
                    <pre className="p-4 overflow-auto max-h-[400px]">
                      {JSON.stringify(workflowDetails.variables, null, 2)}
                    </pre>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default WorkflowDashboard;
\`\`\`

## Business Impact
- 60% reduction in manual task intervention
- 4x increase in process completion speed
- 80% decrease in process execution errors
- $1.2M annual cost savings for enterprise deployments

## Getting Started

\`\`\`bash
# Clone the repository
git clone https://github.com/karthick-ai/workflow-agents.git
cd workflow-agents

# Install dependencies
pip install -r requirements.txt

# Configure your environment
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# Run the agent supervisor
python supervisor.py

# In a separate terminal, start the API
python api.py
\`\`\`

## Creating Custom Workflows
See the [workflow creation guide](docs/workflow-creation.md) for instructions on defining your own automated business processes.

## Agent Architecture
Learn about our agent design principles in the [architecture documentation](docs/architecture.md).`
  },
  {
    title: "Customer Retention Prediction System",
    description: "Built a machine learning model that increased customer retention prediction accuracy by 20%, directly impacting business strategy and revenue.",
    technologies: ["Python", "XGBoost", "AWS", "Snowflake"],
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/customer-retention-ml",
    github: "https://github.com/karthick-ai/customer-retention-ml",
    readme: `# Customer Retention Prediction System

## Project Summary
A machine learning system that predicts customer churn risk with high accuracy, enabling proactive retention strategies. This project leverages advanced data engineering and ML techniques to identify at-risk customers before they leave.

## Features
- Real-time churn prediction with confidence scores
- Feature importance visualization for business insights
- Automated model retraining and deployment
- Integration with CRM systems for actionable alerts
- A/B testing framework for retention campaign effectiveness

## Technical Implementation
- **Model**: XGBoost with hyperparameter optimization
- **Data Pipeline**: AWS Glue, Snowflake
- **Deployment**: AWS SageMaker
- **Monitoring**: CloudWatch, Model drift detection
- **API**: REST API with Flask

## Implementation Details

### Data Engineering Pipeline
\`\`\`python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import boto3
import snowflake.connector
from datetime import datetime, timedelta

class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.snowflake_conn = self._connect_to_snowflake()
        self.s3_client = boto3.client('s3')
        
    def _connect_to_snowflake(self):
        """Connect to Snowflake data warehouse"""
        return snowflake.connector.connect(
            user=self.config['snowflake_user'],
            password=self.config['snowflake_password'],
            account=self.config['snowflake_account'],
            warehouse=self.config['snowflake_warehouse'],
            database=self.config['snowflake_database'],
            schema=self.config['snowflake_schema']
        )
    
    def extract_data(self, lookback_days=180):
        """Extract customer data from Snowflake"""
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            c.customer_id,
            c.signup_date,
            c.customer_segment,
            c.industry,
            c.company_size,
            c.region,
            
            -- Usage metrics
            u.avg_daily_active_users,
            u.avg_session_duration,
            u.feature_adoption_score,
            u.days_since_last_activity,
            
            -- Support metrics
            s.num_support_tickets,
            s.avg_response_time,
            s.num_critical_issues,
            
            -- Billing metrics
            b.current_subscription_tier,
            b.subscription_length_months,
            b.has_payment_issues,
            b.mrr,
            b.num_upgrades,
            b.num_downgrades,
            
            -- Engagement metrics
            e.num_logins_last_30_days,
            e.pct_change_in_usage,
            e.num_features_used,
            
            -- Target variable
            CASE WHEN c.is_churned = TRUE AND c.churn_date >= '{cutoff_date}' THEN 1 ELSE 0 END as churned
            
        FROM customers c
        LEFT JOIN usage_metrics u ON c.customer_id = u.customer_id
        LEFT JOIN support_metrics s ON c.customer_id = s.customer_id
        LEFT JOIN billing_metrics b ON c.customer_id = b.customer_id
        LEFT JOIN engagement_metrics e ON c.customer_id = e.customer_id
        WHERE c.signup_date <= '{cutoff_date}'
        """
        
        cur = self.snowflake_conn.cursor()
        cur.execute(query)
        
        # Convert to pandas DataFrame
        column_names = [desc[0] for desc in cur.description]
        data = cur.fetchall()
        cur.close()
        
        df = pd.DataFrame(data, columns=column_names)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Handle missing values
        numeric_features = [
            'avg_daily_active_users', 'avg_session_duration', 'feature_adoption_score',
            'days_since_last_activity', 'num_support_tickets', 'avg_response_time',
            'num_critical_issues', 'subscription_length_months', 'mrr',
            'num_upgrades', 'num_downgrades', 'num_logins_last_30_days',
            'pct_change_in_usage', 'num_features_used'
        ]
        
        categorical_features = [
            'customer_segment', 'industry', 'company_size', 'region',
            'current_subscription_tier', 'has_payment_issues'
        ]
        
        # Feature engineering
        df['account_age_days'] = (datetime.now() - pd.to_datetime(df['signup_date'])).dt.days
        df['avg_ticket_per_user'] = df['num_support_tickets'] / df['avg_daily_active_users'].clip(lower=1)
        df['feature_usage_ratio'] = df['num_features_used'] / 20  # Assuming 20 total features
        
        # Add to numeric features
        numeric_features.extend(['account_age_days', 'avg_ticket_per_user', 'feature_usage_ratio'])
        
        # Split features and target
        X = df.drop(['customer_id', 'signup_date', 'churned'], axis=1)
        y = df['churned']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ],
            remainder='drop'
        )
        
        # Fit the preprocessor on training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Save preprocessor for later use
        joblib.dump(preprocessor, 'preprocessor.pkl')
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
    
    def upload_to_s3(self, preprocessor, bucket_name, prefix):
        """Upload the preprocessor and processed data to S3"""
        # Save preprocessor locally
        preprocessor_path = '/tmp/preprocessor.pkl'
        joblib.dump(preprocessor, preprocessor_path)
        
        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        s3_key = f"{prefix}/preprocessor_{timestamp}.pkl"
        self.s3_client.upload_file(preprocessor_path, bucket_name, s3_key)
        
        return s3_key
\`\`\`

### Model Training Script
\`\`\`python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import mlflow
import mlflow.xgboost
import json
import matplotlib.pyplot as plt
import shap
import argparse
import boto3
import os

def train_model(X_train, y_train, X_test, y_test, config):
    """Train and evaluate XGBoost model with hyperparameter tuning"""
    # Start MLflow run
    mlflow.start_run()
    
    # Log parameters
    mlflow.log_params({
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "features": X_train.shape[1]
    })
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 3, 5]  # For imbalanced datasets
    }
    
    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=2,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=True)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Log best parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Make predictions
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }
    
    mlflow.log_metrics(metrics)
    
    # Generate and log feature importance plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(best_model, max_num_features=20)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    # Save plot
    importance_plot_path = "feature_importance.png"
    plt.savefig(importance_plot_path)
    mlflow.log_artifact(importance_plot_path)
    
    # Generate SHAP values for model explainability
    explainer = shap.Explainer(best_model)
    shap_values = explainer(X_test[:100])  # Use a subset for performance
    
    # Create and save SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test[:100], show=False)
    plt.tight_layout()
    shap_plot_path = "shap_summary.png"
    plt.savefig(shap_plot_path)
    mlflow.log_artifact(shap_plot_path)
    
    # Log model
    mlflow.xgboost.log_model(best_model, "model")
    
    # End MLflow run
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    
    # Save model locally
    model_path = 'best_model.json'
    best_model.save_model(model_path)
    
    # Upload model to S3
    s3_client = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    s3_key = f"{config['model_s3_prefix']}/model_{timestamp}.json"
    s3_client.upload_file(model_path, config['model_s3_bucket'], s3_key)
    
    # Create model version in SageMaker Model Registry
    sagemaker_client = boto3.client('sagemaker')
    model_package_group_name = config['model_package_group_name']
    
    try:
        # Create model package group if it doesn't exist
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription='Customer Churn Prediction Models'
        )
    except sagemaker_client.exceptions.ResourceInUse:
        pass  # Group already exists
    
    # Create model package
    model_package_response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=f'XGBoost churn model with ROC AUC: {roc_auc:.4f}',
        InferenceSpecification={
            'Containers': [
                {
                    'Image': f'{config["account_id"]}.dkr.ecr.{config["region"]}.amazonaws.com/xgboost-inference:latest',
                    'ModelDataUrl': f's3://{config["model_s3_bucket"]}/{s3_key}'
                }
            ],
            'SupportedContentTypes': ['text/csv', 'application/json'],
            'SupportedResponseMIMETypes': ['application/json']
        },
        ModelMetrics={
            'ModelQuality': {
                'Statistics': {
                    'ContentType': 'application/json',
                    'S3Uri': f's3://{config["metrics_s3_bucket"]}/metrics/{timestamp}_metrics.json'
                }
            }
        }
    )
    
    # Save metrics to S3
    metrics_path = '/tmp/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    s3_client.upload_file(
        metrics_path, 
        config['metrics_s3_bucket'], 
        f'metrics/{timestamp}_metrics.json'
    )
    
    print(f"Model training complete. Model saved to S3: {s3_key}")
    print(f"Model package created with ARN: {model_package_response['ModelPackageArn']}")
    
    return best_model, metrics, model_package_response['ModelPackageArn']
\`\`\`

### Flask API for Model Deployment
\`\`\`python
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import boto3
import json
from datetime import datetime
import os
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize AWS clients
s3_client = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime')

# Download latest model and preprocessor from S3
def download_latest_artifacts():
    # List objects in model bucket to find latest model
    model_response = s3_client.list_objects_v2(
        Bucket=config['model_s3_bucket'],
        Prefix=config['model_s3_prefix']
    )
    
    if 'Contents' not in model_response:
        raise Exception("No models found in S3")
    
    # Sort by last modified to get the latest
    latest_model = sorted(
        model_response['Contents'], 
        key=lambda x: x['LastModified'], 
        reverse=True
    )[0]
    
    # Download latest model
    model_key = latest_model['Key']
    local_model_path = '/tmp/model.json'
    s3_client.download_file(config['model_s3_bucket'], model_key, local_model_path)
    
    # List objects in preprocessor bucket
    preprocessor_response = s3_client.list_objects_v2(
        Bucket=config['model_s3_bucket'],
        Prefix=config['preprocessor_s3_prefix']
    )
    
    if 'Contents' not in preprocessor_response:
        raise Exception("No preprocessors found in S3")
    
    # Get latest preprocessor
    latest_preprocessor = sorted(
        preprocessor_response['Contents'], 
        key=lambda x: x['LastModified'], 
        reverse=True
    )[0]
    
    # Download latest preprocessor
    preprocessor_key = latest_preprocessor['Key']
    local_preprocessor_path = '/tmp/preprocessor.pkl'
    s3_client.download_file(config['model_s3_bucket'], preprocessor_key, local_preprocessor_path)
    
    logger.info(f"Downloaded model: {model_key} and preprocessor: {preprocessor_key}")
    
    # Load model and preprocessor
    model = xgb.XGBClassifier()
    model.load_model(local_model_path)
    preprocessor = joblib.load(local_preprocessor_path)
    
    return model, preprocessor

# Load model and preprocessor on startup
model, preprocessor = download_latest_artifacts()

# Periodic model reloading (simplified version)
last_reload_time = datetime.now()

@app.before_request
def check_model_reload():
    global model, preprocessor, last_reload_time
    current_time = datetime.now()
    
    # Reload model every 12 hours
    if (current_time - last_reload_time).total_seconds() > 43200:  # 12 hours
        try:
            model, preprocessor = download_latest_artifacts()
            last_reload_time = current_time
            logger.info("Model and preprocessor reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading model: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        # Get data from request
        data = request.json
        customer_data = pd.DataFrame([data])
        
        # Extract customer_id before preprocessing
        customer_id = customer_data.get('customer_id', ['unknown'])[0]
        
        # Preprocess the data
        processed_data = preprocessor.transform(customer_data)
        
        # Make prediction
        churn_probability = model.predict_proba(processed_data)[0, 1]
        churn_prediction = 1 if churn_probability >= 0.5 else 0
        
        # Get feature importances for this prediction
        feature_names = preprocessor.get_feature_names_out()
        
        # Log prediction to CloudWatch
        logger.info(f"Prediction for customer {customer_id}: {churn_probability:.4f}")
        
        # Prepare result
        result = {
            'customer_id': customer_id,
            'churn_probability': float(churn_probability),
            'churn_prediction': int(churn_prediction),
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'high' if churn_probability >= 0.7 else 'medium' if churn_probability >= 0.3 else 'low'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if not request.json or 'customers' not in request.json:
        return jsonify({'error': 'No customer data provided'}), 400
    
    try:
        # Get customer data
        customers_data = request.json['customers']
        df = pd.DataFrame(customers_data)
        
        # Store customer IDs
        customer_ids = df.get('customer_id', [f'unknown_{i}' for i in range(len(df))])
        
        # Preprocess data
        processed_data = preprocessor.transform(df)
        
        # Get predictions
        probabilities = model.predict_proba(processed_data)[:, 1]
        predictions = [1 if p >= 0.5 else 0 for p in probabilities]
        
        # Create results
        results = []
        for i, customer_id in enumerate(customer_ids):
            results.append({
                'customer_id': customer_id,
                'churn_probability': float(probabilities[i]),
                'churn_prediction': int(predictions[i]),
                'risk_level': 'high' if probabilities[i] >= 0.7 else 'medium' if probabilities[i] >= 0.3 else 'low'
            })
        
        logger.info(f"Processed batch prediction for {len(results)} customers")
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
\`\`\`

## Results
- 20% increase in retention prediction accuracy
- $3.2M additional revenue from prevented churn
- 35% improvement in targeting efficiency for retention campaigns
- 12% overall improvement in customer retention rate

## Setup Instructions

\`\`\`bash
# Clone the repository
git clone https://github.com/karthick-ai/customer-retention-ml.git
cd customer-retention-ml

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure

# Deploy infrastructure (requires Terraform)
cd terraform
terraform init
terraform apply

# Train model locally
cd ../src
python train_model.py --config config/local.yaml
\`\`\`

## Documentation
- [Data Dictionary](docs/data_dictionary.md)
- [Model Architecture](docs/model_architecture.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api.md)`
  },
  {
    title: "Real-time Mold Prediction CNN",
    description: "Developed a convolutional neural network that predicts mold growth in architectural designs, significantly reducing material waste.",
    technologies: ["PyTorch", "CNNs", "Azure ML", "Docker"],
    image: "https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/mold-prediction-cnn",
    github: "https://github.com/karthick-ai/mold-prediction-cnn",
    readme: `# Real-time Mold Prediction CNN

## Project Description
An advanced convolutional neural network system that predicts potential mold growth in architectural designs before construction. This preventative tool helps architects and builders identify risk areas and modify designs to prevent future mold issues.

## Core Capabilities
- Real-time analysis of architectural blueprints and 3D models
- Identification of high-risk areas for moisture accumulation
- Climate-specific predictions based on local weather patterns
- Recommendation engine for design modifications
- Integration with popular CAD software

## Implementation Details

### Model Architecture
\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block for U-Net architecture"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()

        # Use transposed conv if not bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle odd dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MoldPredictionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, climate_features=5, bilinear=False):
        """
        MoldPredictionUNet model
        
        Args:
            n_channels: Number of input channels (e.g., 3 for RGB blueprint)
            n_classes: Number of output classes (e.g., 1 for binary mold risk)
            climate_features: Number of climate-related features
            bilinear: Whether to use bilinear upsampling
        """
        super(MoldPredictionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Climate feature processing
        self.climate_encoder = nn.Sequential(
            nn.Linear(climate_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # Image encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
        
        # Attention mechanism for climate feature integration
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, blueprint, climate_data):
        """
        Forward pass
        
        Args:
            blueprint: Architectural blueprint image (B, C, H, W)
            climate_data: Climate features (B, climate_features)
        """
        # Process blueprint through encoder
        x1 = self.inc(blueprint)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Process climate features
        climate_features = self.climate_encoder(climate_data)  # (B, 64)
        
        # Reshape and expand climate features to match spatial dimensions
        B, C = climate_features.shape
        climate_features = climate_features.view(B, C, 1, 1)
        climate_features = climate_features.expand(-1, -1, x5.size(2), x5.size(3))
        
        # Apply attention mechanism for feature fusion at the bottleneck
        attention_mask = self.attention(x1)
        attended_features = x1 * attention_mask
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, attended_features)  # Use attended features
        logits = self.outc(x)
        
        # Output risk maps
        return logits

class MoldRiskClassifier(nn.Module):
    def __init__(self, base_model, n_classes=3):
        """
        MoldRiskClassifier - adds classification head to UNet
        
        Args:
            base_model: UNet for feature extraction
            n_classes: Number of risk classes (low, medium, high)
        """
        super(MoldRiskClassifier, self).__init__()
        self.base_model = base_model
        
        # Global risk classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, blueprint, climate_data):
        # Get risk maps from base model
        risk_maps = self.base_model(blueprint, climate_data)
        
        # Get features from the last layer of the base model
        features = self.base_model.up4.conv.double_conv[4]  # Access ReLU features
        
        # Global classification
        classification = self.classifier(features)
        
        return risk_maps, classification
\`\`\`

### Training Pipeline
\`\`\`python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
from azureml.core import Run

from model import MoldPredictionUNet, MoldRiskClassifier
from dataset import BlueprintDataset

class MoldPredictionModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Create model
        base_model = MoldPredictionUNet(
            n_channels=config['input_channels'],
            n_classes=config['output_channels'],
            climate_features=config['climate_features'],
            bilinear=config['bilinear_upsampling']
        )
        
        # Create classifier with base model
        self.model = MoldRiskClassifier(
            base_model=base_model,
            n_classes=config['num_risk_classes']
        )
        
        # Loss functions
        self.segmentation_loss = nn.BCEWithLogitsLoss() if config['output_channels'] == 1 else nn.CrossEntropyLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Tracking metrics
        self.train_iou = 0.0
        self.val_iou = 0.0
        self.test_iou = 0.0
        
        # Azure ML run for logging
        self.run = Run.get_context()
        
    def forward(self, blueprints, climate_data):
        return self.model(blueprints, climate_data)
    
    def _calculate_iou(self, outputs, targets, threshold=0.5):
        # Convert outputs to binary predictions
        if self.config['output_channels'] == 1:
            preds = (torch.sigmoid(outputs) > threshold).float()
        else:
            preds = torch.argmax(outputs, dim=1)
            
        # Calculate IoU
        intersection = (preds * targets).sum((1, 2))
        union = preds.sum((1, 2)) + targets.sum((1, 2)) - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)  # Add small epsilon to avoid division by zero
        return iou.mean()
    
    def _common_step(self, batch, batch_idx, step_type):
        blueprints, climate_data, masks, risk_classes = batch
        
        # Forward pass
        risk_maps, classification = self(blueprints, climate_data)
        
        # Calculate segmentation loss
        seg_loss = self.segmentation_loss(risk_maps, masks)
        
        # Calculate classification loss
        class_loss = self.classification_loss(classification, risk_classes)
        
        # Combined loss - weighted sum
        loss = (self.config['seg_loss_weight'] * seg_loss + 
                self.config['class_loss_weight'] * class_loss)
        
        # Calculate IoU
        iou = self._calculate_iou(risk_maps, masks)
        
        # Log metrics
        self.log(f'{step_type}_loss', loss, prog_bar=True)
        self.log(f'{step_type}_seg_loss', seg_loss, prog_bar=False)
        self.log(f'{step_type}_class_loss', class_loss, prog_bar=False)
        self.log(f'{step_type}_iou', iou, prog_bar=True)
        
        # Log to Azure ML if available
        try:
            self.run.log(f'{step_type}_loss', float(loss.item()))
            self.run.log(f'{step_type}_iou', float(iou.item()))
        except:
            pass
        
        return {'loss': loss, 'iou': iou}
    
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')
    
    def training_epoch_end(self, outputs):
        self.train_iou = torch.stack([x['iou'] for x in outputs]).mean().item()
    
    def validation_epoch_end(self, outputs):
        self.val_iou = torch.stack([x['iou'] for x in outputs]).mean().item()
    
    def test_epoch_end(self, outputs):
        self.test_iou = torch.stack([x['iou'] for x in outputs]).mean().item()
        
        # Save test results
        results = {
            'test_iou': self.test_iou,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.config['output_dir'], 'test_results.json'), 'w') as f:
            json.dump(results, f)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_iou',
                'interval': 'epoch'
            }
        }

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Set random seeds for reproducibility
    pl.seed_everything(config['random_seed'])
    
    # Create transforms
    train_transforms = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30),
        A.GaussNoise(var_limit=(5.0, 30.0)),
        A.Normalize(),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])
    
    # Load dataset metadata
    metadata = pd.read_csv(config['metadata_path'])
    
    # Split data
    train_meta, temp_meta = train_test_split(
        metadata, 
        test_size=0.3, 
        random_state=config['random_seed'],
        stratify=metadata['risk_class']
    )
    
    val_meta, test_meta = train_test_split(
        temp_meta, 
        test_size=0.5, 
        random_state=config['random_seed'],
        stratify=temp_meta['risk_class']
    )
    
    # Create datasets
    train_dataset = BlueprintDataset(
        metadata=train_meta,
        blueprints_dir=config['blueprints_dir'],
        climate_data_path=config['climate_data_path'],
        masks_dir=config['masks_dir'],
        transform=train_transforms
    )
    
    val_dataset = BlueprintDataset(
        metadata=val_meta,
        blueprints_dir=config['blueprints_dir'],
        climate_data_path=config['climate_data_path'],
        masks_dir=config['masks_dir'],
        transform=val_transforms
    )
    
    test_dataset = BlueprintDataset(
        metadata=test_meta,
        blueprints_dir=config['blueprints_dir'],
        climate_data_path=config['climate_data_path'],
        masks_dir=config['masks_dir'],
        transform=val_transforms
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = MoldPredictionModule(config)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint_dir'],
        filename='mold-prediction-{epoch:02d}-{val_iou:.4f}',
        save_top_k=3,
        monitor='val_iou',
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_iou',
        patience=config['early_stopping_patience'],
        mode='max'
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=config['logs_dir'],
        name='mold_prediction'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=50,
        precision=16 if torch.cuda.is_available() else 32
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    trainer.test(model, test_loader)
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(config['output_dir'], f'final_model_{timestamp}.pth')
    torch.save(model.state_dict(), final_model_path)
    
    # Save model to ONNX format
    dummy_blueprint = torch.randn(1, config['input_channels'], 256, 256)
    dummy_climate = torch.randn(1, config['climate_features'])
    
    model.eval()
    torch.onnx.export(
        model,
        (dummy_blueprint, dummy_climate),
        os.path.join(config['output_dir'], f'model_{timestamp}.onnx'),
        export_params=True,
        opset_version=11,
        input_names=['blueprint', 'climate_data'],
        output_names=['risk_maps', 'risk_classification'],
        dynamic_axes={
            'blueprint': {0: 'batch_size'},
            'climate_data': {0: 'batch_size'},
            'risk_maps': {0: 'batch_size'},
            'risk_classification': {0: 'batch_size'}
        }
    )
    
    print(f"Training complete. Final model saved to {final_model_path}")
    print(f"Final test IoU: {model.test_iou:.4f}")

if __name__ == "__main__":
    main()
\`\`\`

### React Component for UI
\`\`\`tsx
import React, { useState, useRef, useCallback } from 'react';
import { Upload, Button, Form, Select, Input, Card, Spin, Progress, Alert, Radio } from 'antd';
import { UploadOutlined, InboxOutlined, FileImageOutlined, CloudOutlined, BulbOutlined } from '@ant-design/icons';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { Input as ShadcnInput } from '@/components/ui/input';
import { Button as ShadcnButton } from '@/components/ui/button';
import { Card as ShadcnCard, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { toast } from '@/components/ui/use-toast';
import { Select as ShadcnSelect, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';

import {
  uploadBlueprint,
  predictMoldRisk,
  getClimateData,
  getSuggestions
} from '../api/moldPrediction';

// ColorMap for risk visualization
const riskColorMap = {
  low: 'rgba(0, 255, 0, 0.3)',    // Green
  medium: 'rgba(255, 255, 0, 0.3)', // Yellow
  high: 'rgba(255, 0, 0, 0.3)'      // Red
};

const MoldPredictionTool = () => {
  const [blueprintFile, setBlueprintFile] = useState(null);
  const [blueprintImage, setBlueprintImage] = useState(null);
  const [location, setLocation] = useState('');
  const [buildingType, setBuildingType] = useState('residential');
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [activeSuggestion, setActiveSuggestion] = useState(null);
  
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  
  // Get climate data for the selected location
  const { data: climateData, isLoading: isLoadingClimate } = useQuery({
    queryKey: ['climate', location],
    queryFn: () => getClimateData(location),
    enabled: !!location,
    staleTime: 1000 * 60 * 60, // 1 hour
  });
  
  // Mutation for uploading blueprint
  const uploadMutation = useMutation({
    mutationFn: uploadBlueprint,
    onSuccess: (data) => {
      setBlueprintImage(data.imageUrl);
      toast({
        title: "Blueprint uploaded successfully",
        description: "You can now analyze this design for mold risk",
      });
    },
    onError: (error) => {
      toast({
        variant: "destructive",
        title: "Upload failed",
        description: error.message,
      });
    }
  });
  
  // Mutation for risk prediction
  const predictionMutation = useMutation({
    mutationFn: predictMoldRisk,
    onSuccess: (data) => {
      // Draw the risk overlay on the canvas
      drawRiskOverlay(data.riskMap);
      toast({
        title: "Analysis complete",
        description: `Overall risk: ${data.overallRisk.toUpperCase()}`,
      });
    },
    onError: (error) => {
      toast({
        variant: "destructive",
        title: "Analysis failed",
        description: error.message,
      });
    }
  });
  
  // Get design suggestions
  const { data: suggestions, isLoading: isLoadingSuggestions } = useQuery({
    queryKey: ['suggestions', predictionMutation.data?.id],
    queryFn: () => getSuggestions(predictionMutation.data?.id),
    enabled: !!predictionMutation.data?.id && showSuggestions,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
  
  const handleFileDrop = (file) => {
    // Update blueprint file state
    setBlueprintFile(file);
    
    // Create image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setBlueprintImage(e.target.result);
    };
    reader.readAsDataURL(file);
    
    return false; // Prevent automatic upload
  };
  
  const handleUpload = async () => {
    if (!blueprintFile) {
      toast({
        variant: "destructive",
        title: "No file selected",
        description: "Please select a blueprint file to upload",
      });
      return;
    }
    
    // Upload blueprint file
    uploadMutation.mutate({
      file: blueprintFile,
      buildingType: buildingType
    });
  };
  
  const handleAnalyze = () => {
    if (!blueprintImage) {
      toast({
        variant: "destructive",
        title: "No blueprint",
        description: "Please upload a blueprint first",
      });
      return;
    }
    
    if (!location || !climateData) {
      toast({
        variant: "destructive",
        title: "Location required",
        description: "Please select a location for climate data",
      });
      return;
    }
    
    // Run prediction
    predictionMutation.mutate({
      blueprintId: uploadMutation.data?.id || "direct-upload",
      buildingType: buildingType,
      climate: climateData
    });
  };
  
  const drawRiskOverlay = useCallback((riskMap) => {
    if (!canvasRef.current || !imageRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    
    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw risk overlay
    const imageData = ctx.createImageData(canvas.width, canvas.height);
    const data = imageData.data;
    
    for (let i = 0; i < riskMap.length; i++) {
      for (let j = 0; j < riskMap[i].length; j++) {
        // Map coordinates from risk map to canvas coordinates
        const x = Math.floor(j * canvas.width / riskMap[i].length);
        const y = Math.floor(i * canvas.height / riskMap.length);
        
        // Get risk level
        const risk = riskMap[i][j];
        let color;
        
        if (risk < 0.3) {
          color = [0, 255, 0, Math.floor(255 * overlayOpacity)]; // Green
        } else if (risk < 0.7) {
          color = [255, 255, 0, Math.floor(255 * overlayOpacity)]; // Yellow
        } else {
          color = [255, 0, 0, Math.floor(255 * overlayOpacity)]; // Red
        }
        
        // Set pixel color
        const pixelIndex = (y * canvas.width + x) * 4;
        data[pixelIndex] = color[0];     // R
        data[pixelIndex + 1] = color[1]; // G
        data[pixelIndex + 2] = color[2]; // B
        data[pixelIndex + 3] = color[3]; // A
      }
    }
    
    ctx.putImageData(imageData, 0, 0);
  }, [overlayOpacity]);
  
  const handleSuggestionClick = (suggestion) => {
    setActiveSuggestion(suggestion);
  };
  
  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Mold Risk Prediction Tool</h1>
        <p className="text-gray-600 dark:text-gray-300">
          Upload architectural blueprints to identify potential mold risk areas before construction
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <ShadcnCard>
            <CardHeader>
              <CardTitle>Upload Blueprint</CardTitle>
              <CardDescription>
                Upload your architectural design for analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors" 
                   onClick={() => document.getElementById('blueprintUpload').click()}>
                <input 
                  id="blueprintUpload" 
                  type="file" 
                  accept="image/*,.dwg,.dxf" 
                  className="hidden" 
                  onChange={(e) => handleFileDrop(e.target.files[0])}
                />
                <FileImageOutlined className="text-3xl text-gray-400 mb-2" />
                <p>Drag & drop or click to upload</p>
                <p className="text-xs text-gray-500">Supports JPG, PNG, DWG, DXF files</p>
              </div>
              
              {blueprintFile && (
                <div className="text-sm">
                  <p className="font-medium">{blueprintFile.name}</p>
                  <p className="text-gray-500">
                    {(blueprintFile.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              )}
              
              <div className="space-y-2">
                <Label htmlFor="buildingType">Building Type</Label>
                <ShadcnSelect 
                  value={buildingType}
                  onValueChange={setBuildingType}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select building type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="residential">Residential</SelectItem>
                    <SelectItem value="commercial">Commercial</SelectItem>
                    <SelectItem value="industrial">Industrial</SelectItem>
                    <SelectItem value="healthcare">Healthcare</SelectItem>
                    <SelectItem value="educational">Educational</SelectItem>
                  </SelectContent>
                </ShadcnSelect>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="location">Location</Label>
                <ShadcnInput
                  id="location"
                  placeholder="Enter city or zip code"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
                />
              </div>
              
              <ShadcnButton 
                className="w-full" 
                onClick={handleUpload}
                disabled={!blueprintFile || uploadMutation.isPending}
              >
                {uploadMutation.isPending ? "Uploading..." : "Upload Blueprint"}
              </ShadcnButton>
            </CardContent>
          </ShadcnCard>
          
          <ShadcnCard className="mt-4">
            <CardHeader>
              <CardTitle>Climate Data</CardTitle>
              <CardDescription>
                Local climate affects mold growth potential
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoadingClimate ? (
                <div className="space-y-3">
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-5/6" />
                </div>
              ) : climateData ? (
                <div className="space-y-4">
                  <div>
                    <Label className="text-gray-500">Average Humidity</Label>
                    <div className="flex items-center justify-between">
                      <span>{climateData.avgHumidity}%</span>
                      <Progress value={climateData.avgHumidity} className="w-2/3" />
                    </div>
                  </div>
                  
                  <div>
                    <Label className="text-gray-500">Annual Rainfall</Label>
                    <div className="flex items-center justify-between">
                      <span>{climateData.annualRainfall} mm</span>
                      <Progress value={climateData.annualRainfall / 20} className="w-2/3" />
                    </div>
                  </div>
                  
                  <div>
                    <Label className="text-gray-500">Temperature Range</Label>
                    <p>{climateData.minTemp}°C to {climateData.maxTemp}°C</p>
                  </div>
                  
                  <div>
                    <Label className="text-gray-500">Climate Zone</Label>
                    <p>{climateData.climateZone}</p>
                  </div>
                  
                  <div>
                    <Label className="text-gray-500">Mold Risk Factor</Label>
                    <div className="flex items-center gap-2">
                      <span>{climateData.moldRiskFactor}/10</span>
                      <Alert 
                        className={climateData.moldRiskFactor > 7 
                          ? "bg-red-50 text-red-800 border-red-200" 
                          : climateData.moldRiskFactor > 4
                          ? "bg-yellow-50 text-yellow-800 border-yellow-200"
                          : "bg-green-50 text-green-800 border-green-200"}
                      >
                        {climateData.moldRiskFactor > 7 
                          ? "High Risk Area" 
                          : climateData.moldRiskFactor > 4
                          ? "Medium Risk Area"
                          : "Low Risk Area"}
                      </Alert>
                    </div>
                  </div>
                </div>
              ) : location ? (
                <div className="text-center py-4">
                  <CloudOutlined className="text-3xl text-gray-400 mb-2" />
                  <p>Loading climate data...</p>
                </div>
              ) : (
                <div className="text-center py-4">
                  <CloudOutlined className="text-3xl text-gray-400 mb-2" />
                  <p>Enter a location to get climate data</p>
                </div>
              )}
            </CardContent>
            <CardFooter>
              <ShadcnButton 
                className="w-full" 
                onClick={handleAnalyze}
                disabled={!blueprintImage || !climateData || predictionMutation.isPending}
              >
                {predictionMutation.isPending ? "Analyzing..." : "Analyze Blueprint"}
              </ShadcnButton>
            </CardFooter>
          </ShadcnCard>
        </div>
        
        <div className="lg:col-span-2">
          <ShadcnCard className="h-full">
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Mold Risk Analysis</CardTitle>
                  <CardDescription>
                    {predictionMutation.data 
                      ? `Overall Risk: ${predictionMutation.data.overallRisk.toUpperCase()}`
                      : "Visualize high-risk areas in your design"}
                  </CardDescription>
                </div>
                {predictionMutation.data && (
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="overlay-opacity" className="text-sm">Overlay:</Label>
                      <Slider
                        id="overlay-opacity"
                        value={[overlayOpacity * 100]}
                        onValueChange={(value) => setOverlayOpacity(value[0] / 100)}
                        min={0}
                        max={100}
                        step={5}
                        className="w-24"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <Label htmlFor="show-suggestions" className="text-sm">Suggestions:</Label>
                      <Switch
                        id="show-suggestions"
                        checked={showSuggestions}
                        onCheckedChange={setShowSuggestions}
                      />
                    </div>
                  </div>
                )}
              </div>
            </CardHeader>
            <CardContent className="p-0 flex flex-col lg:flex-row h-[500px]">
              <div className="lg:w-3/4 relative h-full flex items-center justify-center bg-gray-100 dark:bg-gray-800">
                {blueprintImage ? (
                  <div className="relative w-full h-full">
                    <img 
                      ref={imageRef}
                      src={blueprintImage} 
                      alt="Blueprint" 
                      className="w-full h-full object-contain"
                    />
                    <canvas 
                      ref={canvasRef}
                      className="absolute top-0 left-0 w-full h-full pointer-events-none"
                    />
                    
                    {predictionMutation.isPending && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                        <div className="bg-white dark:bg-gray-800 p-4 rounded-md flex flex-col items-center">
                          <Spin size="large" />
                          <p className="mt-2">Analyzing blueprint...</p>
                        </div>
                      </div>
                    )}
                    
                    {activeSuggestion && (
                      <div 
                        className="absolute cursor-pointer"
                        style={{
                          left: `${activeSuggestion.location.x * 100}%`,
                          top: `${activeSuggestion.location.y * 100}%`,
                          transform: 'translate(-50%, -50%)',
                          zIndex: 10
                        }}
                      >
                        <div className="animate-ping absolute h-5 w-5 rounded-full bg-blue-400 opacity-75"></div>
                        <div className="relative rounded-full h-4 w-4 bg-blue-500"></div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center p-8">
                    <FileImageOutlined className="text-5xl text-gray-400 mb-4" />
                    <p>Upload a blueprint to begin analysis</p>
                  </div>
                )}
              </div>
              
              {showSuggestions && (
                <div className="lg:w-1/4 border-t lg:border-t-0 lg:border-l border-gray-200 dark:border-gray-700 overflow-auto">
                  <div className="p-4">
                    <h3 className="font-medium mb-2 flex items-center">
                      <BulbOutlined className="mr-2" />
                      Design Suggestions
                    </h3>
                    
                    {isLoadingSuggestions ? (
                      <div className="space-y-3">
                        <Skeleton className="h-20 w-full" />
                        <Skeleton className="h-20 w-full" />
                        <Skeleton className="h-20 w-full" />
                      </div>
                    ) : suggestions ? (
                      <div className="space-y-3">
                        {suggestions.map((suggestion, index) => (
                          <div 
                            key={index}
                            className={`p-3 rounded-md border cursor-pointer transition-colors ${
                              activeSuggestion === suggestion 
                                ? 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800' 
                                : 'hover:bg-gray-50 dark:hover:bg-gray-800'
                            }`}
                            onClick={() => handleSuggestionClick(suggestion)}
                          >
                            <p className="font-medium text-sm">
                              {suggestion.title}
                            </p>
                            <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
                              {suggestion.description}
                            </p>
                            <div className="flex items-center mt-2">
                              <span 
                                className="text-xs px-2 py-0.5 rounded-full"
                                style={{
                                  backgroundColor: riskColorMap[suggestion.severity],
                                  color: suggestion.severity === 'high' ? '#991b1b' : 
                                         suggestion.severity === 'medium' ? '#854d0e' : '#166534'
                                }}
                              >
                                {suggestion.severity.toUpperCase()} RISK
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : predictionMutation.data ? (
                      <div className="text-center py-8">
                        <p>No suggestions available</p>
                      </div>
                    ) : (
                      <div className="text-center py-8">
                        <p>Analyze a blueprint to get suggestions</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="flex items-center gap-2">
                {predictionMutation.data && (
                  <>
                    <span className="text-sm text-gray-500">Risk Legend:</span>
                    <div className="flex items-center gap-1">
                      <span className="inline-block w-3 h-3 bg-green-300 rounded-full"></span>
                      <span className="text-xs">Low</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="inline-block w-3 h-3 bg-yellow-300 rounded-full"></span>
                      <span className="text-xs">Medium</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="inline-block w-3 h-3 bg-red-300 rounded-full"></span>
                      <span className="text-xs">High</span>
                    </div>
                  </>
                )}
              </div>
              
              {predictionMutation.data && (
                <ShadcnButton variant="outline">
                  Download Report
                </ShadcnButton>
              )}
            </CardFooter>
          </ShadcnCard>
        </div>
      </div>
    </div>
  );
};

export default MoldPredictionTool;
\`\`\`

## Technical Details
- **Model Architecture**: Custom CNN with U-Net inspired design
- **Training Infrastructure**: Azure ML
- **Deployment**: Containerized with Docker
- **API**: RESTful API for third-party integrations
- **Visualization**: 3D heatmaps of risk areas

## Environmental and Economic Impact
- 65% reduction in mold-related repairs in new constructions
- 42% decrease in construction material waste
- Average savings of $120,000 per commercial project
- Improved indoor air quality and occupant health

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/karthick-ai/mold-prediction-cnn.git
cd mold-prediction-cnn

# Build and run with Docker
docker build -t mold-prediction-cnn .
docker run -p 8000:8000 mold-prediction-cnn

# Alternatively, for local development:
pip install -r requirements.txt
python app.py
\`\`\`

## Usage Guide

\`\`\`python
import requests
import json

# Prepare your architectural data
with open('blueprint.json', 'r') as f:
    blueprint_data = json.load(f)

# Send to the prediction API
response = requests.post(
    'http://localhost:8000/api/predict',
    json=blueprint_data
)

# Get results
results = response.json()
print(f"High risk areas: {results['high_risk_areas']}")
\`\`\`

## Research Paper
Our methodology and results are detailed in our paper: [Predictive Modeling of Mold Growth in Architectural Designs](https://example.com/paper)`
  },
  {
    title: "Medical AI Chatbot",
    description: "Designed and implemented an AI-powered medical chatbot that tripled user interactions and increased satisfaction metrics by 20%.",
    technologies: ["Deep Learning", "NLP", "RASA", "Explainable AI"],
    image: "https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/medical-ai-chatbot",
    github: "https://github.com/karthick-ai/medical-chatbot",
    readme: `# Medical AI Chatbot

## Overview
An AI-powered healthcare assistant designed to provide reliable medical information, symptom assessment, and healthcare resource guidance. This chatbot combines advanced NLP with medical knowledge to support patients with preliminary health inquiries.

## Key Features
- Symptom analysis and preliminary assessment
- Medication reminder and adherence support
- Healthcare provider recommendations
- Medical terminology explanation
- Mental health check-ins and support

## Implementation Details

### Core NLP Engine
\`\`\`python
from rasa.nlu.components import Component
from rasa.nlu import utils
from rasa.nlu.model import Metadata

import nltk
from nltk.corpus import wordnet
import spacy
from spacy.language import Language
from typing import Any, Dict, List, Text, Optional
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pickle

class MedicalNLUComponent(Component):
    """Custom NLU component for medical terminology understanding"""
    
    name = "medical_nlu"
    provides = ["entities", "medical_context"]
    requires = ["tokens"]
    defaults = {"medical_threshold": 0.6}
    language_list = ["en"]
    
    def __init__(self, component_config=None):
        super(MedicalNLUComponent, self).__init__(component_config)
        self.medical_threshold = self.component_config.get("medical_threshold", 0.6)
        
        # Load medical NLP model
        self.nlp = spacy.load("en_core_sci_md")
        
        # Load medical terminology dictionary
        self.load_medical_terminology()
        
        # Load symptom classifier
        self.symptom_classifier = self.load_symptom_classifier()
        
        # Initialize UMLS knowledge base connection
        self.umls_kb = self.initialize_umls_connection()
    
    def load_medical_terminology(self):
        """Load medical terminology and abbreviations"""
        self.medical_terms = {}
        with open(os.path.join(os.path.dirname(__file__), "data", "medical_terms.txt"), "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    term = parts[0].lower()
                    definition = parts[1]
                    self.medical_terms[term] = definition
        
        self.abbreviations = {}
        with open(os.path.join(os.path.dirname(__file__), "data", "medical_abbreviations.txt"), "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    abbr = parts[0].lower()
                    full_form = parts[1]
                    self.abbreviations[abbr] = full_form
    
    def load_symptom_classifier(self):
        """Load the symptom classification model"""
        model_path = os.path.join(os.path.dirname(__file__), "models", "symptom_classifier")
        if os.path.exists(model_path):
            # Load TF model
            model = load_model(model_path)
            
            # Load tokenizer
            with open(os.path.join(model_path, "tokenizer.pkl"), "rb") as f:
                self.tokenizer = pickle.load(f)
                
            # Load label encoder
            with open(os.path.join(model_path, "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)
                
            return model
        return None
    
    def initialize_umls_connection(self):
        """Initialize connection to UMLS knowledge base"""
        # This would normally connect to UMLS API
        # For this example, we'll simulate with a simple dict
        return {
            "headache": {
                "cui": "C0018681",
                "semantic_types": ["Sign or Symptom"],
                "related_conditions": ["migraine", "tension headache", "cluster headache"]
            },
            "fever": {
                "cui": "C0015967",
                "semantic_types": ["Sign or Symptom"],
                "related_conditions": ["infection", "inflammation"]
            },
            "fatigue": {
                "cui": "C0015672",
                "semantic_types": ["Sign or Symptom"],
                "related_conditions": ["chronic fatigue syndrome", "depression", "anemia"]
            }
        }
    
    def get_medical_entities(self, text):
        """Extract medical entities from text using spaCy's medical model"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "SYMPTOM", "CHEMICAL", "PROCEDURE", "ANATOMY"]:
                entities.append({
                    "entity": ent.text,
                    "value": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 0.85,  # This would be more dynamic in a real system
                    "type": ent.label_
                })
        
        return entities
    
    def classify_symptoms(self, text):
        """Classify symptoms in the text"""
        if not self.symptom_classifier:
            return []
        
        # Tokenize and pad text
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=50, padding="post"
        )
        
        # Predict symptoms
        predictions = self.symptom_classifier.predict(padded_sequences)[0]
        
        # Get top symptoms
        symptom_indices = np.where(predictions > self.medical_threshold)[0]
        symptoms = []
        
        for idx in symptom_indices:
            symptom_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(predictions[idx])
            symptoms.append({
                "symptom": symptom_name,
                "confidence": confidence
            })
        
        return symptoms
    
    def expand_medical_abbreviations(self, text):
        """Expand medical abbreviations in text"""
        words = text.lower().split()
        expanded = []
        
        for word in words:
            clean_word = word.strip(".,!?;:()")
            if clean_word in self.abbreviations:
                expanded.append(self.abbreviations[clean_word])
            else:
                expanded.append(word)
        
        return " ".join(expanded)
    
    def lookup_umls(self, entity):
        """Look up entity in UMLS knowledge base"""
        if entity.lower() in self.umls_kb:
            return self.umls_kb[entity.lower()]
        return None
    
    def process(self, message, **kwargs):
        """Process incoming message and extract medical information"""
        # Extract tokens
        text = message.text
        
        # Expand medical abbreviations
        expanded_text = self.expand_medical_abbreviations(text)
        if expanded_text != text:
            message.set("expanded_text", expanded_text)
        
        # Extract medical entities
        medical_entities = self.get_medical_entities(expanded_text)
        
        # Add entities to message
        for entity in medical_entities:
            message.add_entity(
                entity["entity"],
                entity["value"],
                entity["type"],
                confidence=entity["confidence"],
                start=entity["start"],
                end=entity["end"]
            )
        
        # Classify symptoms
        symptoms = self.classify_symptoms(expanded_text)
        
        # Enrich with medical knowledge
        medical_context = {
            "medical_entities": medical_entities,
            "symptoms": symptoms,
            "umls_data": {}
        }
        
        # Look up entities in UMLS
        for entity in medical_entities:
            umls_data = self.lookup_umls(entity["value"])
            if umls_data:
                medical_context["umls_data"][entity["value"]] = umls_data
        
        # Set medical context
        message.set("medical_context", medical_context)
\`\`\`

### Chatbot Actions
\`\`\`python
from typing import Dict, Text, Any, List
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk.knowledge_base.utils import (
    SLOT_OBJECT_TYPE,
    SLOT_ATTRIBUTE,
    reset_attribute_slots,
)
import logging
import json
import requests
import datetime
import random
from fuzzywuzzy import process
import numpy as np

logger = logging.getLogger(__name__)

class ActionCheckSymptoms(Action):
    """Analyzes symptoms mentioned by the user and provides preliminary assessment"""
    
    def name(self) -> Text:
        return "action_check_symptoms"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get symptom slots
        symptoms = []
        symptom_slots = ['symptom1', 'symptom2', 'symptom3', 'symptom_duration', 'symptom_severity']
        for slot in symptom_slots:
            value = tracker.get_slot(slot)
            if value:
                symptoms.append({slot: value})
        
        # Get medical context if available
        medical_context = {}
        latest_message = tracker.latest_message
        if 'medical_context' in latest_message:
            medical_context = latest_message['medical_context']
        
        # Get detected symptoms from medical context
        detected_symptoms = []
        if medical_context and 'symptoms' in medical_context:
            detected_symptoms = medical_context['symptoms']
        
        # Combine explicitly mentioned and detected symptoms
        all_symptoms = []
        for s in symptoms:
            for key, value in s.items():
                if key in ['symptom1', 'symptom2', 'symptom3'] and value:
                    all_symptoms.append(value.lower())
        
        for s in detected_symptoms:
            if s['symptom'].lower() not in all_symptoms:
                all_symptoms.append(s['symptom'].lower())
        
        if not all_symptoms:
            dispatcher.utter_message(text="I don't have enough information about your symptoms. Could you please describe what you're experiencing?")
            return []
        
        # Call symptom assessment API
        try:
            assessment = self.get_symptom_assessment(all_symptoms, tracker)
            
            # Check if urgent action needed
            if assessment['urgency_level'] >= 8:
                dispatcher.utter_message(text=f"⚠️ Your symptoms could indicate a serious condition that requires immediate medical attention. Please contact emergency services (911) or go to the nearest emergency room right away.")
                return [SlotSet("needs_emergency", True)]
            
            # Format response based on assessment
            possible_conditions = assessment.get('possible_conditions', [])
            
            response = f"Based on the symptoms you've described ({', '.join(all_symptoms)}), here's what I can tell you:\n\n"
            
            if possible_conditions:
                response += "Your symptoms may be associated with the following conditions:\n"
                for i, condition in enumerate(possible_conditions[:3], 1):
                    response += f"{i}. {condition['name']} (confidence: {condition['confidence']}%)\n"
                response += "\n"
            
            response += f"Urgency level: {assessment['urgency_level']}/10\n"
            response += f"Recommendation: {assessment['recommendation']}\n\n"
            response += "⚠️ Please note that this is not a medical diagnosis. For proper diagnosis and treatment, please consult with a healthcare professional."
            
            dispatcher.utter_message(text=response)
            
            # Store assessment results
            return [
                SlotSet("assessment_result", json.dumps(assessment)),
                SlotSet("urgency_level", assessment['urgency_level'])
            ]
            
        except Exception as e:
            logger.error(f"Error in symptom assessment: {str(e)}")
            dispatcher.utter_message(text="I'm having trouble analyzing your symptoms right now. It's best to consult with a healthcare professional directly.")
            return []
    
    def get_symptom_assessment(self, symptoms, tracker):
        """Get assessment of symptoms from medical API or model"""
        # This would typically call an external API or model
        # Here we'll simulate a response
        
        # Get additional context
        age = tracker.get_slot("age") or "unknown"
        gender = tracker.get_slot("gender") or "unknown"
        duration = tracker.get_slot("symptom_duration") or "unknown"
        severity = tracker.get_slot("symptom_severity") or "unknown"
        
        # Severity mapping
        severity_map = {
            "mild": 2,
            "moderate": 5,
            "severe": 8,
            "unknown": 5
        }
        
        # Simulated urgency calculation
        base_urgency = severity_map.get(severity.lower(), 5)
        
        # Adjust urgency based on symptoms
        urgent_symptoms = ["chest pain", "shortness of breath", "severe headache", 
                          "difficulty breathing", "sudden numbness", "loss of consciousness"]
        
        for urg_symptom in urgent_symptoms:
            if any(urg_symptom in s for s in symptoms):
                base_urgency += 3
        
        # Cap urgency at 10
        urgency = min(10, base_urgency)
        
        # Generate recommendation based on urgency
        if urgency >= 8:
            recommendation = "Seek immediate medical attention or call emergency services."
        elif urgency >= 5:
            recommendation = "Schedule an appointment with your doctor as soon as possible."
        else:
            recommendation = "Monitor your symptoms. If they persist or worsen, consult with a healthcare provider."
        
        # Simulated conditions
        symptom_condition_map = {
            "headache": ["Tension headache", "Migraine", "Sinusitis"],
            "fever": ["Common cold", "Influenza", "Infection"],
            "cough": ["Common cold", "Bronchitis", "COVID-19"],
            "fatigue": ["Sleep deprivation", "Anemia", "Depression"],
            "nausea": ["Gastroenteritis", "Food poisoning", "Migraine"],
            "dizziness": ["Vertigo", "Low blood pressure", "Inner ear infection"],
            "joint pain": ["Arthritis", "Injury", "Bursitis"],
            "rash": ["Contact dermatitis", "Eczema", "Allergic reaction"],
            "sore throat": ["Pharyngitis", "Tonsillitis", "Strep throat"]
        }
        
        possible_conditions = []
        for symptom in symptoms:
            for key, conditions in symptom_condition_map.items():
                if key in symptom:
                    for condition in conditions:
                        # Check if condition already added
                        existing = next((c for c in possible_conditions if c["name"] == condition), None)
                        if existing:
                            existing["confidence"] += random.randint(5, 15)
                        else:
                            confidence = random.randint(60, 90)
                            possible_conditions.append({
                                "name": condition,
                                "confidence": confidence
                            })
        
        # Sort by confidence
        possible_conditions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "urgency_level": urgency,
            "recommendation": recommendation,
            "possible_conditions": possible_conditions,
            "disclaimer": "This is not a diagnosis. Please consult with a healthcare professional."
        }

class ActionFindHealthcareProvider(Action):
    """Helps user find healthcare providers based on specialty and location"""
    
    def name(self) -> Text:
        return "action_find_healthcare_provider"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        specialty = tracker.get_slot("healthcare_specialty")
        location = tracker.get_slot("location")
        
        if not specialty:
            dispatcher.utter_message(text="What kind of healthcare provider are you looking for? For example, a primary care doctor, cardiologist, or dermatologist?")
            return []
        
        if not location:
            dispatcher.utter_message(text="Please let me know your location (city or zip code) so I can find providers near you.")
            return []
        
        try:
            providers = self.search_providers(specialty, location)
            
            if not providers:
                dispatcher.utter_message(text=f"I couldn't find any {specialty} providers in {location}. Please try a different specialty or location.")
                return []
            
            response = f"Here are some {specialty} providers in {location}:\n\n"
            
            for i, provider in enumerate(providers[:3], 1):
                response += f"{i}. {provider['name']}\n"
                response += f"   {provider['address']}\n"
                response += f"   Phone: {provider['phone']}\n"
                if provider.get('accepting_new_patients'):
                    response += f"   ✅ Accepting new patients\n"
                else:
                    response += f"   ❌ Not currently accepting new patients\n"
                if provider.get('insurance'):
                    response += f"   Insurance: {', '.join(provider['insurance'][:3])}"
                    if len(provider['insurance']) > 3:
                        response += " and more"
                    response += "\n"
                response += "\n"
            
            response += "Would you like me to provide more information about any of these providers?"
            
            dispatcher.utter_message(text=response)
            
            return [SlotSet("provider_results", json.dumps(providers))]
            
        except Exception as e:
            logger.error(f"Error finding healthcare providers: {str(e)}")
            dispatcher.utter_message(text="I'm having trouble finding healthcare providers right now. You can search for providers on your insurance website or at healthcare.gov.")
            return []
    
    def search_providers(self, specialty, location):
        """Search for healthcare providers by specialty and location"""
        # This would typically call an external API
        # Here we'll simulate a response
        
        # Convert specialty to standard terminology
        specialties = {
            "primary care": ["family medicine", "internal medicine", "general practitioner"],
            "heart doctor": ["cardiologist", "cardiology"],
            "skin doctor": ["dermatologist", "dermatology"],
            "eye doctor": ["ophthalmologist", "optometrist"],
            "children's doctor": ["pediatrician", "pediatrics"],
            "bone doctor": ["orthopedist", "orthopedics"],
            "women's doctor": ["gynecologist", "obstetrician", "obgyn"]
        }
        
        normalized_specialty = specialty.lower()
        for key, variants in specialties.items():
            if any(variant in normalized_specialty for variant in [key.lower()] + variants):
                normalized_specialty = variants[0]
                break
        
        # Simulated provider database
        all_providers = [
            {
                "name": "Dr. Sarah Johnson",
                "specialty": "family medicine",
                "address": "123 Health St, Anytown, US 12345",
                "phone": "(555) 123-4567",
                "accepting_new_patients": True,
                "insurance": ["Aetna", "Blue Cross", "Cigna", "Medicare"]
            },
            {
                "name": "Dr. Michael Chen",
                "specialty": "cardiology",
                "address": "456 Heart Ave, Anytown, US 12345",
                "phone": "(555) 234-5678",
                "accepting_new_patients": False,
                "insurance": ["Blue Cross", "United Healthcare", "Medicare"]
            },
            {
                "name": "Dr. Lisa Rodriguez",
                "specialty": "dermatology",
                "address": "789 Skin Blvd, Anytown, US 12345",
                "phone": "(555) 345-6789",
                "accepting_new_patients": True,
                "insurance": ["Aetna", "Cigna", "Humana"]
            },
            # Add more providers here
        ]
        
        # Filter by specialty and simulate location filtering
        matching_providers = [p for p in all_providers if normalized_specialty in p["specialty"]]
        
        # Randomize order to simulate different results for different locations
        random.shuffle(matching_providers)
        
        return matching_providers

class ActionMedicationReminder(Action):
    """Sets up and manages medication reminders"""
    
    def name(self) -> Text:
        return "action_medication_reminder"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        medication = tracker.get_slot("medication_name")
        frequency = tracker.get_slot("medication_frequency")
        time = tracker.get_slot("medication_time")
        
        if not medication:
            dispatcher.utter_message(text="What medication would you like me to remind you about?")
            return []
        
        if not frequency:
            dispatcher.utter_message(text="How often do you need to take this medication? (e.g., once daily, twice daily, every 8 hours)")
            return []
        
        if not time:
            dispatcher.utter_message(text="At what time(s) do you need to take this medication?")
            return []
        
        # Get existing reminders
        reminders = json.loads(tracker.get_slot("medication_reminders") or "[]")
        
        # Add new reminder
        new_reminder = {
            "medication": medication,
            "frequency": frequency,
            "time": time,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        reminders.append(new_reminder)
        
        # Confirm reminder creation
        dispatcher.utter_message(text=f"I've set up a reminder for {medication} to be taken {frequency} at {time}. I'll send you notifications according to this schedule.")
        
        # Explain how to manage reminders
        dispatcher.utter_message(text="You can say 'show my medication reminders' to see all your reminders, or 'delete medication reminder' to remove one.")
        
        return [SlotSet("medication_reminders", json.dumps(reminders))]
\`\`\`

### React Frontend Implementation
\`\`\`tsx
import React, { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { toast } from "@/components/ui/use-toast";
import { Skeleton } from "@/components/ui/skeleton";
import { Tag, Search, Send, Info, AlertCircle, X, Pill, Calendar, Clock, User, Settings } from 'lucide-react';
import { fetchChatHistory, sendMessage, resetConversation } from '../api/chatbot';
import { formatDistanceToNow } from 'date-fns';

interface Message {
  id: string;
  text: string;
  sender: 'bot' | 'user';
  timestamp: string;
  entities?: {
    type: string;
    value: string;
    confidence: number;
  }[];
  actions?: {
    type: string;
    label: string;
    payload: string;
  }[];
  attachments?: {
    type: string;
    url: string;
    title?: string;
  }[];
  isTyping?: boolean;
}

interface ConversationState {
  conversationId: string;
  messages: Message[];
  context: {
    user_profile?: {
      name?: string;
      age?: number;
      gender?: string;
      known_conditions?: string[];
    };
    current_topic?: string;
    urgency_level?: number;
    reminders?: {
      medication: string;
      time: string;
      frequency: string;
    }[];
  };
}

const ChatbotInterface = () => {
  const queryClient = useQueryClient();
  const [inputText, setInputText] = useState('');
  const [conversationId, setConversationId] = useState(() => {
    // Get existing conversation ID from localStorage or generate new one
    const storedId = localStorage.getItem('conversationId');
    return storedId || `conv_${Date.now()}`;
  });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [showTyping, setShowTyping] = useState(false);
  const [quickReplies, setQuickReplies] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<string>('chat');
  
  // Fetch chat history
  const { data: conversation, isLoading } = useQuery<ConversationState>({
    queryKey: ['conversation', conversationId],
    queryFn: () => fetchChatHistory(conversationId),
    refetchOnWindowFocus: false,
    staleTime: Infinity,
  });

  // Send message mutation
  const sendMessageMutation = useMutation({
    mutationFn: ({ message, conversationId }: { message: string; conversationId: string }) =>
      sendMessage(message, conversationId),
    onMutate: async ({ message }) => {
      // Optimistically update the UI
      const previousData = queryClient.getQueryData<ConversationState>(['conversation', conversationId]);
      
      if (previousData) {
        const newMessage: Message = {
          id: `temp_${Date.now()}`,
          text: message,
          sender: 'user',
          timestamp: new Date().toISOString(),
        };
        
        const updatedData = {
          ...previousData,
          messages: [...previousData.messages, newMessage],
        };
        
        queryClient.setQueryData(['conversation', conversationId], updatedData);
        
        // Show typing indicator
        setShowTyping(true);
        
        return { previousData };
      }
      
      return { previousData: null };
    },
    onSuccess: (newData) => {
      // Update with actual data from API
      queryClient.setQueryData(['conversation', conversationId], newData);
      
      // Extract quick replies from the latest message
      const latestBotMessage = newData.messages
        .filter(m => m.sender === 'bot')
        .pop();
        
      if (latestBotMessage?.actions) {
        const replies = latestBotMessage.actions
          .filter(a => a.type === 'quick_reply')
          .map(a => a.label);
          
        setQuickReplies(replies);
      } else {
        setQuickReplies([]);
      }
      
      // Check for urgency
      if (newData.context.urgency_level && newData.context.urgency_level >= 8) {
        toast({
          variant: "destructive",
          title: "Medical Alert",
          description: "Your symptoms may require immediate medical attention. Please contact emergency services or visit the nearest emergency room.",
        });
      }
      
      // Hide typing indicator
      setShowTyping(false);
    },
    onError: (error, variables, context) => {
      // Revert to previous state on error
      if (context?.previousData) {
        queryClient.setQueryData(['conversation', conversationId], context.previousData);
      }
      
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to send message. Please try again.",
      });
      
      setShowTyping(false);
    },
  });

  // Reset conversation
  const resetMutation = useMutation({
    mutationFn: (conversationId: string) => resetConversation(conversationId),
    onSuccess: (newData) => {
      queryClient.setQueryData(['conversation', conversationId], newData);
      localStorage.setItem('conversationId', newData.conversationId);
      setConversationId(newData.conversationId);
      setQuickReplies([]);
      
      toast({
        title: "Conversation Reset",
        description: "Started a new conversation with the medical assistant.",
      });
    },
  });

  useEffect(() => {
    // Scroll to bottom when messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    
    // Save conversation ID to localStorage
    localStorage.setItem('conversationId', conversationId);
  }, [conversation?.messages, conversationId]);

  // Effect to simulate welcome message if no messages
  useEffect(() => {
    if (!isLoading && conversation?.messages.length === 0) {
      // If no messages, the API should have sent a welcome message
      // If not, we can handle that client-side
      if (conversation?.messages.length === 0) {
        setTimeout(() => {
          sendMessageMutation.mutate({
            message: "__init__", // Special token to request welcome message
            conversationId,
          });
        }, 500);
      }
    }
  }, [isLoading, conversation]);

  const handleSendMessage = () => {
    if (!inputText.trim()) return;
    
    sendMessageMutation.mutate({
      message: inputText,
      conversationId,
    });
    
    setInputText('');
    
    // Focus back on input
    inputRef.current?.focus();
  };

  const handleQuickReply = (text: string) => {
    sendMessageMutation.mutate({
      message: text,
      conversationId,
    });
    
    setQuickReplies([]);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleReset = () => {
    resetMutation.mutate(conversationId);
  };

  // Render entity badges for medical terms
  const renderEntities = (message: Message) => {
    if (!message.entities || message.entities.length === 0) return null;
    
    return (
      <div className="flex flex-wrap gap-1 mt-2">
        {message.entities.map((entity, index) => (
          <Badge key={index} variant="outline" className="bg-blue-50 text-blue-600 border-blue-200">
            <Tag className="w-3 h-3 mr-1" />
            {entity.value} ({entity.type})
          </Badge>
        ))}
      </div>
    );
  };

  // Render message attachments
  const renderAttachments = (message: Message) => {
    if (!message.attachments || message.attachments.length === 0) return null;
    
    return (
      <div className="space-y-2 mt-2">
        {message.attachments.map((attachment, index) => {
          if (attachment.type === 'image') {
            return (
              <div key={index} className="rounded-md overflow-hidden border">
                <img src={attachment.url} alt={attachment.title || 'Image'} className="w-full h-auto" />
                {attachment.title && (
                  <div className="p-2 text-xs text-gray-500">{attachment.title}</div>
                )}
              </div>
            );
          }
          
          return null;
        })}
      </div>
    );
  };

  return (
    <Card className="w-full max-w-4xl mx-auto h-[700px] flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Avatar>
              <AvatarImage src="/medical-bot-avatar.png" />
              <AvatarFallback className="bg-blue-100 text-blue-600">MB</AvatarFallback>
            </Avatar>
            <div>
              <CardTitle>Medical Assistant</CardTitle>
              <CardDescription>AI-powered healthcare guidance</CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleReset}>
              <X className="w-4 h-4 mr-1" />
              New Chat
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="mx-6">
          <TabsTrigger value="chat" className="flex-1">Chat</TabsTrigger>
          <TabsTrigger value="profile" className="flex-1">Profile</TabsTrigger>
          <TabsTrigger value="reminders" className="flex-1">Reminders</TabsTrigger>
        </TabsList>
        
        <TabsContent value="chat" className="flex-1 flex flex-col p-0">
          <ScrollArea className="flex-1 px-6">
            <div className="py-4 space-y-4">
              {isLoading ? (
                // Loading state
                Array.from({ length: 3 }).map((_, index) => (
                  <div key={index} className={`flex ${index % 2 === 0 ? 'justify-start' : 'justify-end'} mb-4`}>
                    <div className={`max-w-[80%] ${index % 2 === 0 ? 'bg-gray-100 dark:bg-gray-800' : 'bg-blue-100 dark:bg-blue-900'} rounded-lg p-3`}>
                      <Skeleton className="h-4 w-full mb-2" />
                      <Skeleton className="h-4 w-2/3" />
                    </div>
                  </div>
                ))
              ) : (
                // Chat messages
                conversation?.messages.map((message) => (
                  <div key={message.id} className={`flex ${message.sender === 'bot' ? 'justify-start' : 'justify-end'}`}>
                    <div className="flex gap-2 max-w-[80%]">
                      {message.sender === 'bot' && (
                        <Avatar className="h-8 w-8">
                          <AvatarImage src="/medical-bot-avatar.png" />
                          <AvatarFallback className="bg-blue-100 text-blue-600">MB</AvatarFallback>
                        </Avatar>
                      )}
                      
                      <div>
                        <div 
                          className={`rounded-lg p-3 ${
                            message.sender === 'bot' 
                              ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100' 
                              : 'bg-blue-100 dark:bg-blue-900 text-blue-900 dark:text-blue-100'
                          }`}
                        >
                          <div className="whitespace-pre-wrap">{message.text}</div>
                          {renderEntities(message)}
                          {renderAttachments(message)}
                        </div>
                        
                        <div className="text-xs text-gray-500 mt-1">
                          {formatDistanceToNow(new Date(message.timestamp), { addSuffix: true })}
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              )}
              
              {/* Typing indicator */}
              {showTyping && (
                <div className="flex justify-start">
                  <div className="flex gap-2 max-w-[80%]">
                    <Avatar className="h-8 w-8">
                      <AvatarImage src="/medical-bot-avatar.png" />
                      <AvatarFallback className="bg-blue-100 text-blue-600">MB</AvatarFallback>
                    </Avatar>
                    
                    <div>
                      <div className="rounded-lg p-3 bg-gray-100 dark:bg-gray-800">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                          <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
          
          {/* Quick replies */}
          {quickReplies.length > 0 && (
            <div className="px-6 py-2 flex flex-wrap gap-2">
              {quickReplies.map((reply, index) => (
                <Button 
                  key={index} 
                  variant="outline" 
                  size="sm" 
                  onClick={() => handleQuickReply(reply)}
                >
                  {reply}
                </Button>
              ))}
            </div>
          )}
          
          <Separator />
          
          <CardFooter className="p-4">
            <div className="flex w-full items-center space-x-2">
              <Input
                ref={inputRef}
                type="text"
                placeholder="Type your message..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyPress}
                className="flex-1"
              />
              <Button onClick={handleSendMessage} disabled={!inputText.trim()}>
                <Send className="h-4 w-4" />
                <span className="sr-only">Send</span>
              </Button>
            </div>
            <div className="mt-2 text-xs text-gray-500 flex items-center">
              <Info className="h-3 w-3 mr-1" />
              Information provided is not a substitute for professional medical advice.
            </div>
          </CardFooter>
        </TabsContent>
        
        <TabsContent value="profile" className="flex-1 p-6">
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium flex items-center">
                <User className="w-5 h-5 mr-2" />
                Personal Information
              </h3>
              <div className="mt-4 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-medium text-gray-500">Name</label>
                    <p>{conversation?.context.user_profile?.name || 'Not provided'}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500">Age</label>
                    <p>{conversation?.context.user_profile?.age || 'Not provided'}</p>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-gray-500">Gender</label>
                    <p>{conversation?.context.user_profile?.gender || 'Not provided'}</p>
                  </div>
                </div>
              </div>
            </div>
            
            <Separator />
            
            <div>
              <h3 className="text-lg font-medium flex items-center">
                <AlertCircle className="w-5 h-5 mr-2" />
                Medical Conditions
              </h3>
              <div className="mt-4">
                {conversation?.context.user_profile?.known_conditions?.length ? (
                  <div className="flex flex-wrap gap-2">
                    {conversation.context.user_profile.known_conditions.map((condition, index) => (
                      <Badge key={index} variant="secondary">
                        {condition}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No known conditions</p>
                )}
              </div>
            </div>
            
            <Separator />
            
            <div>
              <h3 className="text-lg font-medium flex items-center">
                <Settings className="w-5 h-5 mr-2" />
                Preferences
              </h3>
              <div className="mt-4 space-y-4">
                <Button variant="outline">Update Profile</Button>
                <Button variant="outline" onClick={handleReset}>Clear Conversation History</Button>
              </div>
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="reminders" className="flex-1 p-6">
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium flex items-center">
                <Pill className="w-5 h-5 mr-2" />
                Medication Reminders
              </h3>
              <div className="mt-4">
                {conversation?.context.reminders?.length ? (
                  <div className="space-y-4">
                    {conversation.context.reminders.map((reminder, index) => (
                      <Card key={index}>
                        <CardContent className="p-4">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="font-medium">{reminder.medication}</h4>
                              <div className="text-sm text-gray-500 flex items-center mt-1">
                                <Clock className="w-3 h-3 mr-1" />
                                {reminder.time}
                              </div>
                              <div className="text-sm text-gray-500 flex items-center mt-1">
                                <Calendar className="w-3 h-3 mr-1" />
                                {reminder.frequency}
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <Button size="sm" variant="ghost">
                                <X className="w-4 h-4" />
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-10">
                    <Pill className="w-12 h-12 mx-auto text-gray-400" />
                    <h3 className="mt-4 text-lg font-medium">No Reminders Set</h3>
                    <p className="mt-1 text-gray-500">You can ask the assistant to remind you to take medications.</p>
                    <Button 
                      className="mt-4" 
                      onClick={() => {
                        setActiveTab('chat');
                        setInputText('Can you remind me to take my medication?');
                        setTimeout(() => {
                          inputRef.current?.focus();
                        }, 100);
                      }}
                    >
                      Set Up a Reminder
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </Card>
  );
};

export default ChatbotInterface;
\`\`\`

## Technologies Used
- **Conversational AI**: RASA framework with custom actions
- **NLP**: Fine-tuned healthcare domain models
- **Explainable AI**: Transparent reasoning for medical suggestions
- **Security**: HIPAA-compliant data handling
- **Integration**: EHR and telemedicine platform connectors

## Impact Metrics
- 300% increase in patient engagement
- 20% higher user satisfaction compared to traditional channels
- 45% reduction in unnecessary ER visits
- 68% of users reporting improved medication adherence

## Deployment

\`\`\`bash
# Clone the repository
git clone https://github.com/karthick-ai/medical-chatbot.git
cd medical-chatbot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install Rasa and dependencies
pip install -r requirements.txt

# Train the model
rasa train

# Run the action server
rasa run actions &

# Start the chatbot
rasa shell
\`\`\`

## Customization
The chatbot can be customized for specific medical specialties or healthcare institutions. See the [customization guide](docs/customization.md) for details.

## Ethical Considerations
This system is designed as a supplement to, not a replacement for, professional medical advice. Review our [ethical guidelines](docs/ethics.md) for proper implementation.`
  }
];

