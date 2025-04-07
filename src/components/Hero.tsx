
import { ArrowDown, Cpu, Database, BrainCircuit, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

const Hero = () => {
  const [animateBackground, setAnimateBackground] = useState(false);
  
  useEffect(() => {
    // Start animation after component mounts
    setAnimateBackground(true);
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.8, ease: "easeOut" }
    }
  };

  return (
    <section
      id="home"
      className="min-h-screen flex flex-col justify-center relative overflow-hidden pt-16"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-teal-50 dark:from-blue-950/20 dark:to-teal-950/20 -z-10"></div>
      
      {/* Enhanced animated background */}
      <div className="absolute inset-0 -z-10">
        <motion.div 
          className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [1, 1.2, 1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{ duration: 8, repeat: Infinity }}
        ></motion.div>
        <motion.div 
          className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [1.2, 1, 1.2],
            opacity: [0.5, 0.7, 0.5],
          }}
          transition={{ duration: 10, repeat: Infinity, delay: 1 }}
        ></motion.div>
        <motion.div 
          className="absolute top-1/3 left-1/4 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [1, 1.3, 1],
            opacity: [0.4, 0.6, 0.4],
          }}
          transition={{ duration: 12, repeat: Infinity, delay: 2 }}
        ></motion.div>
        
        {/* New animated elements */}
        <motion.div 
          className="absolute bottom-1/4 right-1/3 w-72 h-72 bg-pink-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [1.1, 0.9, 1.1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{ duration: 9, repeat: Infinity, delay: 3 }}
        ></motion.div>
        <motion.div 
          className="absolute top-2/3 right-1/5 w-48 h-48 bg-yellow-500/10 rounded-full blur-3xl"
          animate={{ 
            scale: [0.9, 1.1, 0.9],
            opacity: [0.4, 0.7, 0.4],
          }}
          transition={{ duration: 11, repeat: Infinity, delay: 4 }}
        ></motion.div>
        
        {/* Floating elements */}
        <motion.div
          className="absolute top-1/3 right-1/4"
          animate={{
            y: [0, -20, 0],
            rotate: [0, 5, 0]
          }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        >
          <BrainCircuit size={40} className="text-blue-500/40" />
        </motion.div>
        
        <motion.div
          className="absolute bottom-1/3 left-1/4"
          animate={{
            y: [0, 20, 0],
            rotate: [0, -5, 0]
          }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
        >
          <Database size={30} className="text-teal-500/40" />
        </motion.div>
        
        <motion.div
          className="absolute top-2/3 right-1/3"
          animate={{
            y: [0, -15, 0],
            rotate: [0, 10, 0]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        >
          <Cpu size={35} className="text-purple-500/40" />
        </motion.div>
      </div>

      <div className="section-container">
        <motion.div 
          className="max-w-3xl mx-auto text-center"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div 
            className="relative inline-block mb-6"
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-teal-400 blur-xl opacity-30 rounded-full"></div>
            <motion.span 
              className="relative bg-gradient-to-r from-blue-600 to-teal-500 px-6 py-2 text-white text-sm font-medium rounded-full shadow-lg"
              animate={{ 
                backgroundPosition: ["0% center", "100% center", "0% center"],
              }}
              transition={{ 
                duration: 8, 
                repeat: Infinity, 
                ease: "linear",
              }}
              style={{
                backgroundSize: "200% 200%"
              }}
            >
              <motion.span 
                className="inline-block"
                animate={{ 
                  scale: [1, 1.1, 1],
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                Available for Projects
              </motion.span>
            </motion.span>
          </motion.div>
          
          <motion.h1 
            variants={itemVariants}
            className="relative"
          >
            <span className="block text-gray-800 dark:text-gray-200">Machine Learning &</span>
            <span className="gradient-text bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 bg-clip-text text-transparent">
              Generative AI Engineer
            </span>
            <motion.div
              className="absolute -top-10 -right-10 text-yellow-400"
              animate={{
                rotate: [0, 10, -10, 5, -5, 0],
                scale: [1, 1.2, 0.9, 1.1, 1]
              }}
              transition={{ duration: 5, repeat: Infinity }}
            >
              <Sparkles size={40} />
            </motion.div>
          </motion.h1>

          {/* Enhanced creative text animation */}
          <motion.div 
            className="h-16 mt-2 mb-6 relative overflow-hidden"
            variants={itemVariants}
          >
            <div className="absolute inset-0 flex flex-col animate-marquee">
              <motion.span 
                className="text-xl font-medium text-blue-600 dark:text-blue-400 py-4"
                whileInView={{
                  textShadow: ["0 0 0px rgba(59, 130, 246, 0)", "0 0 10px rgba(59, 130, 246, 0.5)", "0 0 0px rgba(59, 130, 246, 0)"]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                LLM Specialist
              </motion.span>
              <motion.span 
                className="text-xl font-medium text-teal-600 dark:text-teal-400 py-4"
                whileInView={{
                  textShadow: ["0 0 0px rgba(20, 184, 166, 0)", "0 0 10px rgba(20, 184, 166, 0.5)", "0 0 0px rgba(20, 184, 166, 0)"]
                }}
                transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
              >
                RAG Systems Developer
              </motion.span>
              <motion.span 
                className="text-xl font-medium text-purple-600 dark:text-purple-400 py-4"
                whileInView={{
                  textShadow: ["0 0 0px rgba(147, 51, 234, 0)", "0 0 10px rgba(147, 51, 234, 0.5)", "0 0 0px rgba(147, 51, 234, 0)"]
                }}
                transition={{ duration: 2, repeat: Infinity, delay: 1 }}
              >
                AI Agent Builder
              </motion.span>
              <span className="text-xl font-medium text-blue-600 dark:text-blue-400 py-4">LLM Specialist</span>
            </div>
          </motion.div>

          <motion.p 
            variants={itemVariants}
            className="mt-6 text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto"
          >
            Transforming complex data challenges into strategic business advantages through innovative machine learning and generative AI solutions.
          </motion.p>

          <motion.div 
            className="mt-10 flex flex-col sm:flex-row justify-center gap-4"
            variants={itemVariants}
          >
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button 
                size="lg" 
                className="relative overflow-hidden group" 
                asChild
              >
                <a href="#contact">
                  <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-blue-600 to-teal-500 transition-all duration-300 group-hover:scale-105"></span>
                  <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-blue-700 to-teal-600 opacity-0 transition-opacity duration-300 group-hover:opacity-100"></span>
                  <span className="relative z-10">Hire Me</span>
                  <motion.span
                    className="absolute inset-0 z-0"
                    animate={{
                      backgroundPosition: ["0% 0%", "100% 100%"],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      repeatType: "reverse",
                    }}
                    style={{
                      background: "radial-gradient(circle, rgba(255,255,255,0.25) 0%, rgba(255,255,255,0) 70%)",
                      backgroundSize: "200% 200%",
                    }}
                  />
                </a>
              </Button>
            </motion.div>
            
            <motion.div
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Button 
                variant="outline" 
                size="lg" 
                className="border-2 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all duration-300" 
                asChild
              >
                <a href="#projects">View My Work</a>
              </Button>
            </motion.div>
          </motion.div>
        </motion.div>
      </div>

      <motion.div 
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        animate={{ 
          y: [0, 10, 0],
          opacity: [0.5, 1, 0.5]
        }}
        transition={{ 
          duration: 2, 
          repeat: Infinity,
          repeatType: "reverse"
        }}
      >
        <a href="#about" aria-label="Scroll down" className="hover:text-blue-600 transition-colors group">
          <div className="p-2 rounded-full bg-white/50 backdrop-blur-sm shadow-md group-hover:bg-blue-100 transition-colors duration-300">
            <ArrowDown className="text-gray-500 dark:text-gray-400 group-hover:text-blue-600 transition-colors duration-300" />
          </div>
        </a>
      </motion.div>
    </section>
  );
};

export default Hero;
