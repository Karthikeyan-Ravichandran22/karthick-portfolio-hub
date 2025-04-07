
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { projects } from "@/data/projectsData";
import { ArrowRight } from "lucide-react";

const Projects = () => {
  return (
    <section id="projects" className="relative overflow-hidden bg-white dark:bg-gray-900">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/5 rounded-full blur-3xl"></div>
      </div>
      
      <div className="section-container relative z-10">
        <div className="text-center mb-16">
          <h2 className="mb-4">My Projects</h2>
          <div className="h-1 w-20 bg-gradient-to-r from-blue-600 to-teal-500 mx-auto rounded-full"></div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Showcasing my key machine learning and generative AI projects
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <Card 
              key={index} 
              className="overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02] bg-white dark:bg-gray-800"
            >
              <div className="h-2 bg-gradient-to-r from-blue-600 to-teal-500"></div>
              <div className="h-48 bg-muted overflow-hidden relative">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-teal-500/20 z-10"></div>
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover transition-transform duration-500 hover:scale-110"
                />
              </div>
              <CardContent className="p-6">
                <h3 className="text-xl font-bold mb-3 text-gray-900 dark:text-white">{project.title}</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  {project.description}
                </p>
                <div className="flex flex-wrap gap-2">
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
              {project.link && (
                <CardFooter className="p-6 pt-0">
                  <a
                    href={project.link}
                    className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium group"
                  >
                    View Project <ArrowRight className="ml-2 h-4 w-4 transform group-hover:translate-x-1 transition-transform" />
                  </a>
                </CardFooter>
              )}
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Projects;
