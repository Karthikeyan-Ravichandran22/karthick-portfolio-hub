
import { Card, CardContent } from "@/components/ui/card";
import { Briefcase, Code, User, Brain, Sparkles, BarChart, Cpu, Database, ChevronRight } from "lucide-react";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

const About = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

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

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { x: -50, opacity: 0 },
    visible: {
      x: 0,
      opacity: 1,
      transition: { duration: 0.8, ease: "easeOut" }
    }
  };

  const cardVariants = {
    hidden: { y: 30, opacity: 0 },
    visible: (i: number) => ({
      y: 0,
      opacity: 1,
      transition: { 
        duration: 0.6, 
        ease: "easeOut",
        delay: 0.2 * i 
      }
    })
  };

  return (
    <section id="about" className="bg-gradient-to-br from-white to-blue-50 dark:from-gray-900 dark:to-gray-800 py-24 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 -z-10">
        <motion.div 
          className="absolute -top-10 -right-10 w-72 h-72 bg-blue-500/5 rounded-full blur-3xl"
          animate={{ 
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{ duration: 10, repeat: Infinity }}
        ></motion.div>
        <motion.div 
          className="absolute bottom-20 -left-10 w-80 h-80 bg-teal-500/5 rounded-full blur-3xl"
          animate={{ 
            scale: [1.2, 1, 1.2],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{ duration: 12, repeat: Infinity, delay: 2 }}
        ></motion.div>
        
        {/* Floating tech icons */}
        <motion.div
          className="absolute top-20 right-[20%]"
          animate={{
            y: [0, -15, 0],
            rotate: [0, 5, 0]
          }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        >
          <Cpu size={24} className="text-blue-500/40" />
        </motion.div>
        <motion.div
          className="absolute bottom-40 left-[15%]"
          animate={{
            y: [0, 15, 0],
            rotate: [0, -5, 0]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        >
          <Database size={20} className="text-teal-500/40" />
        </motion.div>
        <motion.div
          className="absolute top-1/3 right-[10%]"
          animate={{
            y: [0, -10, 0],
            rotate: [0, 10, 0]
          }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
        >
          <Brain size={22} className="text-purple-500/40" />
        </motion.div>
      </div>

      <div className="section-container relative z-10">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.7 }}
        >
          <h2 className="mb-4 relative inline-block">
            About Me
            <motion.div 
              className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-3/4 h-1 bg-gradient-to-r from-blue-600 to-teal-500 rounded-full"
              initial={{ scaleX: 0 }}
              animate={{ scaleX: isVisible ? 1 : 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            ></motion.div>
          </h2>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-12 items-center">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate={isVisible ? "visible" : "hidden"}
          >
            <motion.h3 
              className="mb-6 text-3xl font-bold gradient-text bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 bg-clip-text text-transparent"
              variants={itemVariants}
            >
              Machine Learning Expert with a Passion for Innovation
              <motion.span
                className="inline-block ml-2"
                animate={{
                  rotate: [0, 10, -10, 0],
                  scale: [1, 1.2, 0.9, 1]
                }}
                transition={{ duration: 4, repeat: Infinity, repeatDelay: 2 }}
              >
                <Sparkles className="h-6 w-6 text-yellow-400" />
              </motion.span>
            </motion.h3>
            
            <motion.p 
              className="text-gray-600 dark:text-gray-300 mb-6 text-lg"
              variants={itemVariants}
            >
              I specialize in developing cutting-edge machine learning solutions that solve complex business problems. With expertise across the entire ML lifecycle—from data preparation to model deployment and monitoring—I deliver scalable, production-ready solutions that drive tangible business value.
            </motion.p>
            
            <motion.p 
              className="text-gray-600 dark:text-gray-300 text-lg"
              variants={itemVariants}
            >
              My experience spans multiple industries including telecommunications, education, legal tech, and healthcare, where I've consistently delivered projects that improve efficiency, accuracy, and decision-making capabilities.
            </motion.p>
            
            <motion.div 
              className="mt-8 grid grid-cols-2 gap-4"
              variants={containerVariants}
              initial="hidden"
              animate={isVisible ? "visible" : "hidden"}
            >
              {[
                { color: "bg-blue-500", label: "Data Science" },
                { color: "bg-teal-500", label: "Machine Learning" },
                { color: "bg-purple-500", label: "NLP & LLMs" },
                { color: "bg-pink-500", label: "Generative AI" }
              ].map((item, index) => (
                <motion.div 
                  key={index} 
                  className="flex items-center gap-2 group cursor-pointer"
                  variants={itemVariants}
                  whileHover={{ x: 5 }}
                >
                  <motion.div 
                    className={`w-3 h-3 rounded-full ${item.color}`}
                    whileHover={{ scale: 1.5 }}
                  ></motion.div>
                  <motion.span 
                    className="text-gray-700 dark:text-gray-300 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors"
                  >
                    {item.label}
                  </motion.span>
                  <motion.span
                    initial={{ opacity: 0, x: -5 }}
                    animate={{ opacity: 0, x: -5 }}
                    whileHover={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronRight className="h-4 w-4 text-blue-500" />
                  </motion.span>
                </motion.div>
              ))}
            </motion.div>
          </motion.div>

          <motion.div 
            className="grid gap-6"
            variants={containerVariants}
            initial="hidden"
            animate={isVisible ? "visible" : "hidden"}
          >
            {[
              {
                icon: <Brain className="h-6 w-6 text-blue-600 dark:text-blue-400" />,
                title: "Who I Am",
                description: "A data scientist and ML engineer with a strong background in predictive modeling, NLP, and deep learning, dedicated to turning data into strategic advantages.",
                color: "bg-blue-100 dark:bg-blue-900/50",
                index: 0
              },
              {
                icon: <Code className="h-6 w-6 text-teal-600 dark:text-teal-400" />,
                title: "What I Do",
                description: "I build and deploy machine learning models that solve real business problems, from customer retention prediction to document analysis and automated insights.",
                color: "bg-teal-100 dark:bg-teal-900/50",
                index: 1
              },
              {
                icon: <Sparkles className="h-6 w-6 text-purple-600 dark:text-purple-400" />,
                title: "AI Expertise",
                description: "I create cutting-edge generative AI solutions, including RAG systems, LLM fine-tuning, and AI agent development for businesses seeking innovative technology.",
                color: "bg-purple-100 dark:bg-purple-900/50",
                index: 2
              },
              {
                icon: <Briefcase className="h-6 w-6 text-gray-600 dark:text-gray-400" />,
                title: "My Experience",
                description: "Over 6 years of professional experience in data science and machine learning roles, working with diverse technologies and delivering impactful solutions.",
                color: "bg-gray-100 dark:bg-gray-800",
                index: 3
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                custom={item.index}
                variants={cardVariants}
                initial="hidden"
                animate={isVisible ? "visible" : "hidden"}
                whileHover={{ y: -5, scale: 1.02 }}
                onHoverStart={() => setHoveredCard(index)}
                onHoverEnd={() => setHoveredCard(null)}
              >
                <Card 
                  className="overflow-hidden shadow-lg hover:shadow-xl transition-all duration-500"
                >
                  <motion.div 
                    className="h-1 bg-gradient-to-r from-blue-600 to-teal-500"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: hoveredCard === index ? 1 : 0 }}
                    transition={{ duration: 0.3 }}
                    style={{ transformOrigin: "left" }}
                  ></motion.div>
                  
                  <CardContent className="p-6 flex items-start gap-4">
                    <motion.div 
                      className={`${item.color} p-3 rounded-lg transition-colors`}
                      whileHover={{ scale: 1.1 }}
                      animate={{ 
                        rotate: hoveredCard === index ? [0, 5, -5, 0] : 0,
                        y: hoveredCard === index ? [0, -3, 3, 0] : 0
                      }}
                      transition={{ duration: 0.5 }}
                    >
                      {item.icon}
                    </motion.div>
                    
                    <div>
                      <motion.h4 
                        className="text-xl font-semibold mb-2"
                        animate={{ 
                          color: hoveredCard === index ? '#3b82f6' : ''
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {item.title}
                      </motion.h4>
                      
                      <motion.p 
                        className="text-gray-600 dark:text-gray-300"
                        animate={{ 
                          opacity: hoveredCard === index ? [0.8, 1] : 0.8,
                          y: hoveredCard === index ? [2, 0] : 0
                        }}
                        transition={{ duration: 0.3 }}
                      >
                        {item.description}
                      </motion.p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default About;
