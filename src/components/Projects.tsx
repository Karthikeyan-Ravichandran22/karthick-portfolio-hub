
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { projects } from "@/data/projectsData";
import { ArrowRight, ExternalLink, Github, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import ReactMarkdown from 'react-markdown';

const Projects = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [hoveredProject, setHoveredProject] = useState<number | null>(null);
  const [loadedImages, setLoadedImages] = useState<Record<number, boolean>>({});
  const [selectedReadme, setSelectedReadme] = useState<string | null>(null);
  const [isReadmeOpen, setIsReadmeOpen] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    const section = document.getElementById("projects");
    if (section) observer.observe(section);

    return () => {
      if (section) observer.unobserve(section);
    };
  }, []);

  const handleImageLoad = (index: number) => {
    setLoadedImages(prev => ({
      ...prev,
      [index]: true
    }));
  };

  const handleReadmeClick = (index: number) => {
    const project = projects[index];
    if (project?.readme) {
      setSelectedReadme(project.readme);
      setIsReadmeOpen(true);
    }
  };

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
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.8, ease: "easeOut" }
    }
  };

  return (
    <section id="projects" className="relative overflow-hidden py-24 bg-gradient-to-br from-white to-blue-50 dark:from-gray-900 dark:to-gray-800">
      <div className="absolute inset-0 -z-10">
        {/* Animated background blobs */}
        <div className="blob absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full"></div>
        <div className="blob absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/5 rounded-full" style={{ animationDelay: "3s" }}></div>
        <div className="blob absolute top-1/2 left-1/2 w-80 h-80 bg-purple-500/5 rounded-full" style={{ animationDelay: "6s" }}></div>
        <div className="absolute -top-24 left-1/2 transform -translate-x-1/2 w-full h-48 bg-gradient-to-b from-transparent to-white/90 dark:to-gray-900/90 backdrop-blur-sm"></div>
        
        {/* Animated particles */}
        {Array.from({ length: 20 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-blue-500/30 rounded-full"
            initial={{ 
              x: Math.random() * 100 + "%", 
              y: Math.random() * 100 + "%" 
            }}
            animate={{ 
              y: [
                Math.random() * 100 + "%", 
                Math.random() * 100 + "%",
                Math.random() * 100 + "%"
              ]
            }}
            transition={{ 
              duration: 10 + Math.random() * 20, 
              repeat: Infinity,
              repeatType: "reverse"
            }}
          />
        ))}
      </div>
      
      <div className="section-container relative z-10">
        <motion.div 
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={isVisible ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
          transition={{ duration: 0.7 }}
        >
          {/* Section heading */}
          <div className="inline-block mb-4 relative">
            <motion.div
              className="absolute -right-8 -top-8"
              animate={{
                rotate: [0, 10, -10, 5, -5, 0],
                scale: [1, 1.2, 0.9, 1.1, 1]
              }}
              transition={{ duration: 6, repeat: Infinity }}
            >
              <Sparkles className="text-blue-500/70 h-6 w-6" />
            </motion.div>
            
            <h2 className="relative z-10">
              <span className="relative">
                My Projects
                <motion.div 
                  className="absolute -bottom-2 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 rounded-full"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: isVisible ? 1 : 0 }}
                  transition={{ duration: 1, delay: 0.5 }}
                ></motion.div>
              </span>
            </h2>
          </div>
          <motion.p 
            className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: isVisible ? 1 : 0 }}
            transition={{ duration: 1, delay: 0.7 }}
          >
            Showcasing my key machine learning and generative AI projects
          </motion.p>
        </motion.div>

        <motion.div 
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-8"
          variants={containerVariants}
          initial="hidden"
          animate={isVisible ? "visible" : "hidden"}
        >
          {projects.map((project, index) => (
            <motion.div 
              key={index}
              variants={itemVariants}
              onHoverStart={() => setHoveredProject(index)}
              onHoverEnd={() => setHoveredProject(null)}
              whileHover={{ 
                scale: 1.03,
                transition: { duration: 0.3 }
              }}
            >
              <Card 
                className="h-full bg-white dark:bg-gray-800 overflow-hidden border-0 shadow-lg relative group"
              >
                {/* Enhanced animated gradient top bar */}
                <motion.div 
                  className="h-2 bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600"
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: hoveredProject === index ? 1 : 0 }}
                  transition={{ duration: 0.3 }}
                  style={{ transformOrigin: "left" }}
                ></motion.div>
                
                <div className="h-48 overflow-hidden relative">
                  <motion.div 
                    className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-teal-500/20 z-10"
                    animate={{ 
                      opacity: hoveredProject === index ? 0 : 1 
                    }}
                    transition={{ duration: 0.5 }}
                  ></motion.div>
                  
                  {/* Image with loading placeholder */}
                  <div className="w-full h-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                    {!loadedImages[index] && (
                      <div className="animate-pulse flex space-x-4">
                        <div className="w-full h-full bg-gray-300 dark:bg-gray-600"></div>
                      </div>
                    )}
                    <motion.img
                      src={project.image}
                      alt={project.title}
                      className={`w-full h-full object-cover transition-opacity duration-300 ${loadedImages[index] ? 'opacity-100' : 'opacity-0'}`}
                      onLoad={() => handleImageLoad(index)}
                      animate={{ 
                        scale: hoveredProject === index ? 1.1 : 1 
                      }}
                      transition={{ duration: 0.7 }}
                    />
                  </div>
                  
                  <motion.div 
                    className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-4 z-20"
                    initial={{ opacity: 0 }}
                    animate={{ 
                      opacity: hoveredProject === index ? 1 : 0 
                    }}
                    transition={{ duration: 0.5 }}
                  >
                    <motion.h4 
                      className="text-white text-lg font-bold"
                      initial={{ y: 20 }}
                      animate={{ 
                        y: hoveredProject === index ? 0 : 20 
                      }}
                      transition={{ duration: 0.5 }}
                    >
                      {project.title}
                    </motion.h4>
                  </motion.div>
                </div>
                
                <CardContent className="p-6 flex-grow">
                  <motion.h3 
                    className="text-xl font-bold mb-3 text-gray-900 dark:text-white transition-colors duration-300"
                    style={{ 
                      color: hoveredProject === index ? '#3b82f6' : '' 
                    }}
                  >
                    {project.title}
                  </motion.h3>
                  
                  <motion.p 
                    className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3"
                    animate={{ 
                      opacity: [0.8, 1],
                      y: hoveredProject === index ? [2, 0] : 0 
                    }}
                    transition={{ duration: 0.3 }}
                  >
                    {project.description}
                  </motion.p>
                  
                  <div className="flex flex-wrap gap-2 mt-auto">
                    {project.technologies.map((tech, techIndex) => (
                      <motion.div
                        key={techIndex}
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ 
                          opacity: 1, 
                          scale: 1,
                          y: hoveredProject === index ? [5, 0] : 0
                        }}
                        transition={{ 
                          duration: 0.4, 
                          delay: 0.1 * techIndex + (hoveredProject === index ? 0.2 : 0) 
                        }}
                      >
                        <Badge 
                          variant="outline" 
                          className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-blue-100 dark:border-blue-800 hover:bg-blue-100 transition-colors"
                        >
                          {tech}
                        </Badge>
                      </motion.div>
                    ))}
                  </div>
                </CardContent>
                
                <CardFooter className="p-6 pt-0 flex flex-wrap gap-4">
                  {project.link && (
                    <motion.a
                      href={project.link}
                      className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium group/link"
                      whileHover={{ x: 3 }}
                      transition={{ duration: 0.2 }}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Demo 
                      <motion.span
                        className="ml-2"
                        animate={{ 
                          x: hoveredProject === index ? [0, 4, 0] : 0 
                        }}
                        transition={{ 
                          duration: 0.6, 
                          repeat: hoveredProject === index ? Infinity : 0,
                          repeatType: "loop",
                          repeatDelay: 0.5
                        }}
                      >
                        <ExternalLink className="h-4 w-4" />
                      </motion.span>
                    </motion.a>
                  )}
                  
                  {project.github && (
                    <motion.a
                      href={project.github}
                      className="inline-flex items-center text-purple-600 dark:text-purple-400 hover:text-purple-800 dark:hover:text-purple-300 font-medium group/link"
                      whileHover={{ x: 3 }}
                      transition={{ duration: 0.2 }}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub 
                      <motion.span
                        className="ml-2"
                        animate={{ 
                          y: hoveredProject === index ? [0, -2, 0] : 0 
                        }}
                        transition={{ 
                          duration: 0.6, 
                          repeat: hoveredProject === index ? Infinity : 0,
                          repeatType: "loop",
                          repeatDelay: 0.5
                        }}
                      >
                        <Github className="h-4 w-4" />
                      </motion.span>
                    </motion.a>
                  )}
                  
                  {project.readme && (
                    <motion.button
                      onClick={() => handleReadmeClick(index)}
                      className="inline-flex items-center text-green-600 dark:text-green-400 hover:text-green-800 dark:hover:text-green-300 font-medium group/link"
                      whileHover={{ x: 3 }}
                      transition={{ duration: 0.2 }}
                    >
                      README
                      <motion.span
                        className="ml-2"
                        animate={{ 
                          rotate: hoveredProject === index ? [0, 10, 0] : 0 
                        }}
                        transition={{ 
                          duration: 0.6, 
                          repeat: hoveredProject === index ? Infinity : 0,
                          repeatType: "loop",
                          repeatDelay: 0.5
                        }}
                      >
                        <ArrowRight className="h-4 w-4" />
                      </motion.span>
                    </motion.button>
                  )}
                </CardFooter>
                
                {/* Enhanced corner decorative element */}
                <AnimatePresence>
                  {hoveredProject === index && (
                    <motion.div 
                      className="absolute top-0 right-0 w-16 h-16 overflow-hidden"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <motion.div 
                        className="absolute transform rotate-45 bg-gradient-to-r from-blue-600 to-teal-500 text-white w-24 text-center text-xs py-1 right-[-35px] top-[12px]"
                        initial={{ y: -30 }}
                        animate={{ y: 0 }}
                        exit={{ y: -30 }}
                        transition={{ duration: 0.3 }}
                      >
                        New
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* README Dialog */}
      <Dialog open={isReadmeOpen} onOpenChange={setIsReadmeOpen}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>Project README</DialogTitle>
            <DialogDescription>
              Details and instructions for this project
            </DialogDescription>
          </DialogHeader>
          <ScrollArea className="mt-6 h-[60vh] pr-4">
            {selectedReadme && (
              <div className="prose dark:prose-invert max-w-none">
                <ReactMarkdown>
                  {selectedReadme}
                </ReactMarkdown>
              </div>
            )}
          </ScrollArea>
        </DialogContent>
      </Dialog>
    </section>
  );
};

export default Projects;

