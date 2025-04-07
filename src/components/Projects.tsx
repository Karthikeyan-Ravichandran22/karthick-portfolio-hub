
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { projects } from "@/data/projectsData";
import { ArrowRight, ExternalLink } from "lucide-react";
import { useEffect, useState } from "react";

const Projects = () => {
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

    const section = document.getElementById("projects");
    if (section) observer.observe(section);

    return () => {
      if (section) observer.unobserve(section);
    };
  }, []);

  return (
    <section id="projects" className="relative overflow-hidden py-24 bg-gradient-to-br from-white to-blue-50 dark:from-gray-900 dark:to-gray-800">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/5 rounded-full blur-3xl"></div>
        <div className="absolute -top-24 left-1/2 transform -translate-x-1/2 w-full h-48 bg-gradient-to-b from-transparent to-white/90 dark:to-gray-900/90 backdrop-blur-sm"></div>
      </div>
      
      <div className="section-container relative z-10">
        <div className="text-center mb-16">
          <div className="inline-block mb-4">
            <h2 className="relative z-10">
              <span className="relative">
                My Projects
                <div className="absolute -bottom-2 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 rounded-full transform"></div>
              </span>
            </h2>
          </div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Showcasing my key machine learning and generative AI projects
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <div 
              key={index}
              className={`transform transition-all duration-700 ${
                isVisible 
                  ? 'translate-y-0 opacity-100' 
                  : 'translate-y-20 opacity-0'
              }`}
              style={{ transitionDelay: `${index * 0.1}s` }}
            >
              <Card 
                className="group overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-500 h-full bg-white dark:bg-gray-800 relative"
              >
                {/* Decorative top gradient bar */}
                <div className="h-2 bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 transform transition-transform duration-500 group-hover:scale-x-100 origin-left"></div>
                
                <div className="h-48 overflow-hidden relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-teal-500/20 z-10 group-hover:opacity-0 transition-opacity duration-500"></div>
                  <img
                    src={project.image}
                    alt={project.title}
                    className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 flex items-end p-4 z-20">
                    <h4 className="text-white text-lg font-bold">{project.title}</h4>
                  </div>
                </div>
                
                <CardContent className="p-6 flex-grow">
                  <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-300">{project.title}</h3>
                  <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
                    {project.description}
                  </p>
                  <div className="flex flex-wrap gap-2 mt-auto">
                    {project.technologies.map((tech, techIndex) => (
                      <Badge 
                        key={techIndex} 
                        variant="outline" 
                        className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-blue-100 dark:border-blue-800 hover:bg-blue-100 transition-colors"
                      >
                        {tech}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
                
                <CardFooter className="p-6 pt-0">
                  <a
                    href={project.link || "#"}
                    className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium group/link"
                  >
                    View Project <ExternalLink className="ml-2 h-4 w-4 transform group-hover/link:translate-x-1 transition-transform" />
                  </a>
                </CardFooter>
                
                {/* Corner decorative element */}
                <div className="absolute top-0 right-0 w-16 h-16 overflow-hidden">
                  <div className="absolute transform rotate-45 bg-gradient-to-r from-blue-600 to-teal-500 text-white w-24 text-center text-xs py-1 right-[-35px] top-[12px] opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    New
                  </div>
                </div>
              </Card>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Projects;
