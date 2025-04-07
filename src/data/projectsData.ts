
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
    image: "/placeholder.svg"
  },
  {
    title: "RAG System for Legal Document Analysis",
    description: "Developed a Retrieval-Augmented Generation system that analyzed 75,000+ legal documents with 85% accuracy, providing valuable insights for compliance teams.",
    technologies: ["Transformers", "Vector Databases", "Hugging Face", "LangChain"],
    image: "/placeholder.svg"
  },
  {
    title: "AI Agent-Based Workflow Automation",
    description: "Created a system of AI agents that automate complex business workflows, reducing manual intervention by 60% and increasing processing speed by 4x.",
    technologies: ["LLM Orchestration", "Function Calling", "Python", "FastAPI"],
    image: "/placeholder.svg"
  },
  {
    title: "Customer Retention Prediction System",
    description: "Built a machine learning model that increased customer retention prediction accuracy by 20%, directly impacting business strategy and revenue.",
    technologies: ["Python", "XGBoost", "AWS", "Snowflake"],
    image: "/placeholder.svg"
  },
  {
    title: "Real-time Mold Prediction CNN",
    description: "Developed a convolutional neural network that predicts mold growth in architectural designs, significantly reducing material waste.",
    technologies: ["PyTorch", "CNNs", "Azure ML", "Docker"],
    image: "/placeholder.svg"
  },
  {
    title: "Medical AI Chatbot",
    description: "Designed and implemented an AI-powered medical chatbot that tripled user interactions and increased satisfaction metrics by 20%.",
    technologies: ["Deep Learning", "NLP", "RASA", "Explainable AI"],
    image: "/placeholder.svg"
  }
];
