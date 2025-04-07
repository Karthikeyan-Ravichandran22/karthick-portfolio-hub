
export interface Experience {
  title: string;
  company: string;
  period: string;
  responsibilities: string[];
}

export const experiences: Experience[] = [
  {
    title: "Machine Learning Engineer (Contract)",
    company: "City Fibre",
    period: "Feb 2024 – Aug 2024",
    responsibilities: [
      "Built and deployed scalable data models using Python, AWS, and Snowflake, helping teams make accurate decisions on customer behavior and service coverage.",
      "Applied advanced preprocessing techniques (e.g., handling unbalanced data using XGBoost) to improve data quality and prediction accuracy.",
      "Collaborated cross-functionally with senior teams to deliver analytical insights in an Agile environment.",
      "Led a predictive analytics project improving customer retention predictions by 20%, directly enhancing strategic planning."
    ]
  },
  {
    title: "Machine Learning Scientist (Contract)",
    company: "Leeds Beckett University",
    period: "Oct 2023 – Feb 2024",
    responsibilities: [
      "Developed real-time mold prediction systems using CNNs, helping reduce material waste in architectural design.",
      "Deployed ML solutions on Azure for broader team access and long-term scalability.",
      "Worked with multidisciplinary teams, translating technical concepts for non-technical stakeholders.",
      "Enhanced efficiency in sustainability-focused building designs through data-driven decision-making."
    ]
  },
  {
    title: "Data Scientist & ML Specialist",
    company: "Mondaq",
    period: "Oct 2022 – Jan 2023",
    responsibilities: [
      "Led NLP projects analyzing 75K+ documents for legal insights, achieving 80% clustering accuracy.",
      "Created and deployed models for fraud detection and semantic search using Transformers and PyTorch.",
      "Used tools like Hugging Face and LangChain to optimize models for business outcomes.",
      "Delivered insights on legal documents to help reduce compliance risks and improve operational clarity."
    ]
  },
  {
    title: "Machine Learning Engineer",
    company: "iOPEX Technologies",
    period: "Jan 2021 – Dec 2021",
    responsibilities: [
      "Reduced ML model training time by 70% and improved large dataset handling efficiency.",
      "Monitored ML pipelines and delivered dashboards tailored to business needs.",
      "Streamlined research workflows and contributed to faster delivery of analytics solutions."
    ]
  },
  {
    title: "Deep Learning Developer",
    company: "Technocolabs",
    period: "Apr 2020 – Jan 2021",
    responsibilities: [
      "Designed an AI-powered medical chatbot, tripling user interactions and increasing satisfaction by 20%.",
      "Used explainable AI tools (SHAP, LIME) to foster trust among stakeholders."
    ]
  },
  {
    title: "Analyst (Key Account Management)",
    company: "NTC Logistics India Pvt Ltd",
    period: "Jan 2018 – Mar 2020",
    responsibilities: [
      "Created statistical models (logistic regression, decision trees) for donor response analysis, increasing campaign effectiveness by 20%.",
      "Provided strategic data insights via Power BI and SQL, enabling better financial decisions and operational planning.",
      "Delivered regular analytical reports to internal teams and stakeholders to support evidence-based decision-making."
    ]
  }
];
