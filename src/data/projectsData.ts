
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
    image: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80"
  },
  {
    title: "RAG System for Legal Document Analysis",
    description: "Developed a Retrieval-Augmented Generation system that analyzed 75,000+ legal documents with 85% accuracy, providing valuable insights for compliance teams.",
    technologies: ["Transformers", "Vector Databases", "Hugging Face", "LangChain"],
    image: "https://images.unsplash.com/photo-1487058792275-0ad4aaf24ca7?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80"
  },
  {
    title: "AI Agent-Based Workflow Automation",
    description: "Created a system of AI agents that automate complex business workflows, reducing manual intervention by 60% and increasing processing speed by 4x.",
    technologies: ["LLM Orchestration", "Function Calling", "Python", "FastAPI"],
    image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80"
  },
  {
    title: "Customer Retention Prediction System",
    description: "Built a machine learning model that increased customer retention prediction accuracy by 20%, directly impacting business strategy and revenue.",
    technologies: ["Python", "XGBoost", "AWS", "Snowflake"],
    image: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80"
  },
  {
    title: "Real-time Mold Prediction CNN",
    description: "Developed a convolutional neural network that predicts mold growth in architectural designs, significantly reducing material waste.",
    technologies: ["PyTorch", "CNNs", "Azure ML", "Docker"],
    image: "https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80"
  },
  {
    title: "Medical AI Chatbot",
    description: "Designed and implemented an AI-powered medical chatbot that tripled user interactions and increased satisfaction metrics by 20%.",
    technologies: ["Deep Learning", "NLP", "RASA", "Explainable AI"],
    image: "https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?ixlib=rb-1.2.1&auto=format&fit=crop&w=1200&q=80"
  }
];
