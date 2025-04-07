
import { Progress } from "@/components/ui/progress";
import { skillsData } from "@/data/skillsData";
import { useState } from "react";

const Skills = () => {
  const [activeCategory, setActiveCategory] = useState(skillsData[0].category);

  return (
    <section id="skills" className="bg-gray-50 dark:bg-gray-800">
      <div className="section-container">
        <div className="text-center mb-16">
          <h2 className="mb-4">My Skills</h2>
          <div className="h-1 w-20 bg-blue-600 mx-auto rounded-full"></div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            The tools, technologies, and methods I use to build powerful machine learning solutions
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="flex flex-wrap justify-center gap-4 mb-12">
            {skillsData.map((category) => (
              <button
                key={category.category}
                onClick={() => setActiveCategory(category.category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  activeCategory === category.category
                    ? "bg-blue-600 text-white"
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
                <div key={index} className="animate-fade-in" style={{ animationDelay: `${index * 0.1}s` }}>
                  <div className="flex justify-between mb-2">
                    <span className="font-medium">{skill.name}</span>
                    <span className="text-gray-500 dark:text-gray-400">{skill.proficiency}%</span>
                  </div>
                  <Progress value={skill.proficiency} className="h-2" />
                </div>
              ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Skills;
