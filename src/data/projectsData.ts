
export interface Project {
  title: string;
  description: string;
  technologies: string[];
  image: string;
  link?: string;
}

export const projects: Project[] = [
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
    title: "Legal Document Analysis System",
    description: "Created an NLP-based system that analyzed 75,000+ legal documents with 80% clustering accuracy, providing valuable insights for compliance teams.",
    technologies: ["Transformers", "BERT", "Hugging Face", "LangChain"],
    image: "/placeholder.svg"
  },
  {
    title: "ML Pipeline Optimization",
    description: "Engineered optimizations that reduced ML model training time by 70%, significantly improving team productivity and business responsiveness.",
    technologies: ["MLflow", "Python", "Distributed Computing", "GPU Optimization"],
    image: "/placeholder.svg"
  },
  {
    title: "Medical AI Chatbot",
    description: "Designed and implemented an AI-powered medical chatbot that tripled user interactions and increased satisfaction metrics by 20%.",
    technologies: ["Deep Learning", "NLP", "RASA", "Explainable AI"],
    image: "/placeholder.svg"
  },
  {
    title: "Campaign Analytics Dashboard",
    description: "Developed statistical models and visualizations that increased campaign effectiveness by 20% through better donor response analysis.",
    technologies: ["Power BI", "SQL", "Logistic Regression", "Decision Trees"],
    image: "/placeholder.svg"
  }
];
