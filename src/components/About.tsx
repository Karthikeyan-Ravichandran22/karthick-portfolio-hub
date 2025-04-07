
import { Card, CardContent } from "@/components/ui/card";
import { Briefcase, Code, User, Brain, Sparkles, BarChart } from "lucide-react";
import { useEffect, useState } from "react";

const About = () => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    const section = document.getElementById("about");
    if (section) observer.observe(section);

    return () => {
      if (section) observer.unobserve(section);
    };
  }, []);

  return (
    <section id="about" className="bg-gradient-to-br from-white to-blue-50 dark:from-gray-900 dark:to-gray-800 py-24">
      <div className="section-container">
        <div className="text-center mb-16">
          <h2 className="mb-4 relative inline-block">
            About Me
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-3/4 h-1 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full"></div>
          </h2>
        </div>

        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className={`transform transition-all duration-700 ${isVisible ? 'translate-x-0 opacity-100' : '-translate-x-20 opacity-0'}`}>
            <h3 className="mb-6 text-3xl font-bold gradient-text bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 bg-clip-text text-transparent">
              Machine Learning Expert with a Passion for Innovation
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg">
              I specialize in developing cutting-edge machine learning solutions that solve complex business problems. With expertise across the entire ML lifecycle—from data preparation to model deployment and monitoring—I deliver scalable, production-ready solutions that drive tangible business value.
            </p>
            <p className="text-gray-600 dark:text-gray-300 text-lg">
              My experience spans multiple industries including telecommunications, education, legal tech, and healthcare, where I've consistently delivered projects that improve efficiency, accuracy, and decision-making capabilities.
            </p>
            
            <div className="mt-8 grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <span className="text-gray-700 dark:text-gray-300">Data Science</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-teal-500"></div>
                <span className="text-gray-700 dark:text-gray-300">Machine Learning</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                <span className="text-gray-700 dark:text-gray-300">NLP & LLMs</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-pink-500"></div>
                <span className="text-gray-700 dark:text-gray-300">Generative AI</span>
              </div>
            </div>
          </div>

          <div className="grid gap-6">
            {[
              {
                icon: <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
                title: "Who I Am",
                description: "A data scientist and ML engineer with a strong background in predictive modeling, NLP, and deep learning, dedicated to turning data into strategic advantages.",
                color: "bg-blue-100 dark:bg-blue-900/50",
                delay: 0.1
              },
              {
                icon: <Code className="h-6 w-6 text-teal-600 dark:text-teal-400" />,
                title: "What I Do",
                description: "I build and deploy machine learning models that solve real business problems, from customer retention prediction to document analysis and automated insights.",
                color: "bg-teal-100 dark:bg-teal-900/50",
                delay: 0.2
              },
              {
                icon: <Sparkles className="h-6 w-6 text-purple-600 dark:text-purple-400" />,
                title: "AI Expertise",
                description: "I create cutting-edge generative AI solutions, including RAG systems, LLM fine-tuning, and AI agent development for businesses seeking innovative technology.",
                color: "bg-purple-100 dark:bg-purple-900/50",
                delay: 0.3
              },
              {
                icon: <Briefcase className="h-6 w-6 text-gray-600 dark:text-gray-400" />,
                title: "My Experience",
                description: "Over 6 years of professional experience in data science and machine learning roles, working with diverse technologies and delivering impactful solutions.",
                color: "bg-gray-100 dark:bg-gray-800",
                delay: 0.4
              },
            ].map((item, index) => (
              <Card 
                key={index} 
                className={`overflow-hidden shadow-lg hover:shadow-xl transition-all duration-500 transform hover:scale-[1.02] ${
                  isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'
                }`}
                style={{ transitionDelay: `${item.delay}s` }}
              >
                <div className="h-1 bg-gradient-to-r from-blue-600 to-teal-500"></div>
                <CardContent className="p-6 flex items-start gap-4">
                  <div className={`${item.color} p-3 rounded-lg`}>
                    {item.icon}
                  </div>
                  <div>
                    <h4 className="text-xl font-semibold mb-2">{item.title}</h4>
                    <p className="text-gray-600 dark:text-gray-300">
                      {item.description}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
