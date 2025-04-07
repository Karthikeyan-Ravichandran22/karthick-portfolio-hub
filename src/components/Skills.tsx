
import { Progress } from "@/components/ui/progress";
import { skillsData } from "@/data/skillsData";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";

const Skills = () => {
  const [activeCategory, setActiveCategory] = useState(skillsData[0].category);
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

    const section = document.getElementById("skills");
    if (section) observer.observe(section);

    return () => {
      if (section) observer.unobserve(section);
    };
  }, []);

  return (
    <section id="skills" className="relative overflow-hidden py-20 bg-gradient-to-b from-gray-50 to-white dark:from-gray-800 dark:to-gray-900">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/5 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl"></div>
      </div>
      
      <div className="section-container relative z-10">
        <div className="text-center mb-16">
          <h2 className="mb-4 relative inline-block">
            My Skills
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-3/4 h-1 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full"></div>
          </h2>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            The tools, technologies, and methods I use to build powerful AI and machine learning solutions
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            {skillsData.map((category, index) => (
              <button
                key={category.category}
                onClick={() => setActiveCategory(category.category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 transform hover:scale-105 ${
                  activeCategory === category.category
                    ? "bg-gradient-to-r from-blue-600 to-teal-500 text-white shadow-lg"
                    : "bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 shadow"
                }`}
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                {category.category}
              </button>
            ))}
          </div>

          <div className="grid gap-6">
            {skillsData
              .find((category) => category.category === activeCategory)
              ?.skills.map((skill, index) => (
                <div 
                  key={index} 
                  className={`transform transition-all duration-500 ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-10 opacity-0'}`}
                  style={{ transitionDelay: `${index * 0.1}s` }}
                >
                  <div className="flex justify-between mb-2">
                    <span className="font-medium">{skill.name}</span>
                    <span className="text-gray-500 dark:text-gray-400 font-mono">{skill.proficiency}%</span>
                  </div>
                  <div className="h-3 w-full bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden shadow-inner">
                    <div 
                      className="h-full rounded-full bg-gradient-to-r from-blue-600 to-teal-500 relative"
                      style={{ 
                        width: isVisible ? `${skill.proficiency}%` : '0%', 
                        transition: 'width 1.5s cubic-bezier(0.17, 0.67, 0.83, 0.67)' 
                      }}
                    >
                      {skill.proficiency > 90 && (
                        <div className="absolute -right-1 -top-1 w-5 h-5 bg-yellow-300 rounded-full opacity-75 animate-ping"></div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Skills;
