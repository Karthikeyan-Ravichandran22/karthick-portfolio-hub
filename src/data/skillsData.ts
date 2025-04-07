
export interface SkillCategory {
  category: string;
  skills: Skill[];
}

export interface Skill {
  name: string;
  proficiency: number; // 0-100
}

export const skillsData: SkillCategory[] = [
  {
    category: "Generative AI",
    skills: [
      { name: "Large Language Models", proficiency: 95 },
      { name: "Prompt Engineering", proficiency: 95 },
      { name: "RAG Systems", proficiency: 90 },
      { name: "LLM Fine-tuning", proficiency: 85 },
      { name: "AI Agent Development", proficiency: 85 }
    ]
  },
  {
    category: "Machine Learning",
    skills: [
      { name: "Supervised Learning", proficiency: 95 },
      { name: "Unsupervised Learning", proficiency: 90 },
      { name: "Reinforcement Learning", proficiency: 85 },
      { name: "Deep Learning", proficiency: 90 },
      { name: "NLP", proficiency: 85 }
    ]
  },
  {
    category: "Programming",
    skills: [
      { name: "Python", proficiency: 95 },
      { name: "SQL", proficiency: 90 },
      { name: "R", proficiency: 80 },
      { name: "JavaScript", proficiency: 75 },
      { name: "Java", proficiency: 65 }
    ]
  },
  {
    category: "ML Technologies",
    skills: [
      { name: "TensorFlow/Keras", proficiency: 90 },
      { name: "PyTorch", proficiency: 85 },
      { name: "Scikit-learn", proficiency: 95 },
      { name: "XGBoost", proficiency: 90 },
      { name: "Hugging Face", proficiency: 85 }
    ]
  },
  {
    category: "Cloud & MLOps",
    skills: [
      { name: "AWS", proficiency: 90 },
      { name: "Azure ML", proficiency: 85 },
      { name: "Docker", proficiency: 80 },
      { name: "MLflow", proficiency: 85 },
      { name: "Kubernetes", proficiency: 75 }
    ]
  },
  {
    category: "Data Engineering",
    skills: [
      { name: "Snowflake", proficiency: 85 },
      { name: "Spark", proficiency: 80 },
      { name: "Data Pipelines", proficiency: 85 },
      { name: "ETL Processes", proficiency: 90 },
      { name: "Data Visualization", proficiency: 85 }
    ]
  }
];
