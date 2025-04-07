
import { Progress } from "@/components/ui/progress";
import { skillsData } from "@/data/skillsData";
import { useState } from "react";

const Skills = () => {
  const [activeCategory, setActiveCategory] = useState(skillsData[0].category);

  return (
    <section id="skills" className="relative overflow-hidden bg-gray-50 dark:bg-gray-800">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/5 rounded-full blur-3xl"></div>
      </div>
      
      <div className="section-container relative z-10">
        <div className="text-center mb-16">
          <h2 className="mb-4">My Skills</h2>
          <div className="h-1 w-20 bg-gradient-to-r from-blue-600 to-teal-500 mx-auto rounded-full"></div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            The tools, technologies, and methods I use to build powerful AI and machine learning solutions
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            {skillsData.map((category) => (
              <button
                key={category.category}
                onClick={() => setActiveCategory(category.category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 transform hover:scale-105 ${
                  activeCategory === category.category
                    ? "bg-gradient-to-r from-blue-600 to-teal-500 text-white shadow-md"
                    : "bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600"
                }`}
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
                  className="animate-fade-in hover:transform hover:translate-x-2 transition-all duration-300" 
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="flex justify-between mb-2">
                    <span className="font-medium">{skill.name}</span>
                    <span className="text-gray-500 dark:text-gray-400">{skill.proficiency}%</span>
                  </div>
                  <div className="h-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full bg-gradient-to-r from-blue-600 to-teal-500 transition-all duration-1000" 
                      style={{ width: `${skill.proficiency}%`, transition: 'width 1s ease-in-out' }}
                    />
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
