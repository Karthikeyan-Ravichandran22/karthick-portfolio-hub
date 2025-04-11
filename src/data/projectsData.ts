
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

