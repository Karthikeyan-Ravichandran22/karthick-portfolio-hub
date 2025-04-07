
import { Progress } from "@/components/ui/progress";
import { skillsData } from "@/data/skillsData";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

const Skills = () => {
  const [activeCategory, setActiveCategory] = useState(skillsData[0].category);
  const [isVisible, setIsVisible] = useState(false);
  const [hoverSkill, setHoverSkill] = useState<string | null>(null);

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

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.6, ease: "easeOut" }
    }
  };

  return (
    <section id="skills" className="relative overflow-hidden py-20 bg-gradient-to-b from-gray-50 to-white dark:from-gray-800 dark:to-gray-900">
      <div className="absolute inset-0 -z-10">
        {/* Dynamic animated background blobs */}
        <div className="blob absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full"></div>
        <div className="blob absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/5 rounded-full" style={{ animationDelay: "3s" }}></div>
        <div className="blob absolute top-1/2 left-1/2 w-80 h-80 bg-purple-500/5 rounded-full" style={{ animationDelay: "6s" }}></div>
        <div className="blob absolute bottom-1/4 right-1/3 w-72 h-72 bg-pink-500/5 rounded-full" style={{ animationDelay: "9s" }}></div>
      </div>
      
      <div className="section-container relative z-10">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 50 }}
          animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
          transition={{ duration: 0.7, ease: "easeOut" }}
        >
          <h2 className="mb-4 relative inline-block">
            My Skills
            <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-3/4 h-1 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full"></div>
            <motion.div 
              className="absolute -top-8 -right-10 text-3xl"
              animate={{ 
                rotate: [0, 10, -10, 10, 0],
                scale: [1, 1.2, 1, 1.1, 1]
              }}
              transition={{ 
                duration: 5, 
                repeat: Infinity,
                repeatType: "reverse" 
              }}
            >
              <Sparkles className="text-yellow-400" />
            </motion.div>
          </h2>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            The tools, technologies, and methods I use to build powerful AI and machine learning solutions
          </p>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          <motion.div 
            className="flex flex-wrap justify-center gap-4 mb-12"
            variants={containerVariants}
            initial="hidden"
            animate={isVisible ? "visible" : "hidden"}
          >
            {skillsData.map((category, index) => (
              <motion.button
                key={category.category}
                onClick={() => setActiveCategory(category.category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-300 transform hover:scale-105 ${
                  activeCategory === category.category
                    ? "bg-gradient-to-r from-blue-600 to-teal-500 text-white shadow-lg"
                    : "bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 shadow shine"
                }`}
                variants={itemVariants}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {category.category}
              </motion.button>
            ))}
          </motion.div>

          <motion.div 
            className="grid gap-6"
            variants={containerVariants}
            initial="hidden"
            animate={isVisible ? "visible" : "hidden"}
          >
            {skillsData
              .find((category) => category.category === activeCategory)
              ?.skills.map((skill, index) => (
                <motion.div 
                  key={index} 
                  variants={itemVariants}
                  className="group"
                  onMouseEnter={() => setHoverSkill(skill.name)}
                  onMouseLeave={() => setHoverSkill(null)}
                >
                  <div className="flex justify-between mb-2">
                    <span className="font-medium group-hover:text-blue-600 transition-colors duration-300">{skill.name}</span>
                    <span className="text-gray-500 dark:text-gray-400 font-mono">{skill.proficiency}%</span>
                  </div>
                  <div className="h-3 w-full bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden shadow-inner relative">
                    <motion.div 
                      className="h-full rounded-full relative"
                      style={{ 
                        background: `linear-gradient(90deg, 
                          ${hoverSkill === skill.name 
                            ? 'rgb(37, 99, 235), rgb(20, 184, 166)' 
                            : 'rgb(59, 130, 246), rgb(20, 184, 166)'}
                        )`
                      }}
                      initial={{ width: "0%" }}
                      animate={{ 
                        width: isVisible ? `${skill.proficiency}%` : "0%",
                        x: hoverSkill === skill.name ? [0, 5, -5, 3, -3, 0] : 0
                      }}
                      transition={{ 
                        duration: 1.5, 
                        ease: "easeOut",
                        x: { duration: 0.5, repeat: hoverSkill === skill.name ? 2 : 0 }
                      }}
                    >
                      {skill.proficiency > 90 && (
                        <>
                          <div className="absolute -right-1 -top-1 w-5 h-5 bg-yellow-300 rounded-full opacity-75 animate-ping"></div>
                          <motion.div 
                            className="absolute right-0 top-0 w-4 h-4 bg-yellow-400 rounded-full z-10"
                            animate={{ scale: [1, 1.2, 1] }}
                            transition={{ duration: 2, repeat: Infinity }}
                          />
                        </>
                      )}
                      
                      {/* Animated particles for all skills */}
                      <div className="absolute inset-0 overflow-hidden">
                        {[...Array(Math.floor(skill.proficiency / 10))].map((_, i) => (
                          <motion.div
                            key={i}
                            className="absolute w-1 h-1 bg-white/50 rounded-full"
                            initial={{ 
                              x: Math.random() * (skill.proficiency * 3), 
                              y: 6 * Math.random()
                            }}
                            animate={{ 
                              y: [6 * Math.random(), -3, 6 * Math.random()],
                              opacity: [0, 1, 0]
                            }}
                            transition={{ 
                              duration: 2 + Math.random() * 3, 
                              repeat: Infinity,
                              delay: Math.random() * 2
                            }}
                          />
                        ))}
                      </div>
                    </motion.div>
                  </div>
                </motion.div>
              ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Skills;
