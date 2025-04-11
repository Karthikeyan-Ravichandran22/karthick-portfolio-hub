
export interface Project {
  title: string;
  description: string;
  technologies: string[];
  image: string;
  link?: string;
}

export const projects: Project[] = [
  {
    title: "Generative AI-Powered Content Platform",
    description: "Built an enterprise-grade content generation platform using LLMs that increased content creation efficiency by 70% for marketing teams.",
    technologies: ["LangChain", "OpenAI API", "React", "Python", "FastAPI"],
    image: "https://images.unsplash.com/photo-1678565655915-d8e392114ffd?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/ai-content-platform"
  },
  {
    title: "RAG System for Legal Document Analysis",
    description: "Developed a Retrieval-Augmented Generation system that analyzed 75,000+ legal documents with 85% accuracy, providing valuable insights for compliance teams.",
    technologies: ["Transformers", "Vector Databases", "Hugging Face", "LangChain"],
    image: "https://images.unsplash.com/photo-1589829085413-56de8ae18c73?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/rag-legal-system"
  },
  {
    title: "AI Agent-Based Workflow Automation",
    description: "Created a system of AI agents that automate complex business workflows, reducing manual intervention by 60% and increasing processing speed by 4x.",
    technologies: ["LLM Orchestration", "Function Calling", "Python", "FastAPI"],
    image: "https://images.unsplash.com/photo-1531746790731-6c087fecd65a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/ai-agent-automation"
  },
  {
    title: "Customer Retention Prediction System",
    description: "Built a machine learning model that increased customer retention prediction accuracy by 20%, directly impacting business strategy and revenue.",
    technologies: ["Python", "XGBoost", "AWS", "Snowflake"],
    image: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/customer-retention-ml"
  },
  {
    title: "Real-time Mold Prediction CNN",
    description: "Developed a convolutional neural network that predicts mold growth in architectural designs, significantly reducing material waste.",
    technologies: ["PyTorch", "CNNs", "Azure ML", "Docker"],
    image: "https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/mold-prediction-cnn"
  },
  {
    title: "Medical AI Chatbot",
    description: "Designed and implemented an AI-powered medical chatbot that tripled user interactions and increased satisfaction metrics by 20%.",
    technologies: ["Deep Learning", "NLP", "RASA", "Explainable AI"],
    image: "https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80",
    link: "https://example.com/medical-ai-chatbot"
  }
];
