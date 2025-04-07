
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { projects } from "@/data/projectsData";
import { ArrowRight } from "lucide-react";

const Projects = () => {
  return (
    <section id="projects" className="bg-white dark:bg-gray-900">
      <div className="section-container">
        <div className="text-center mb-16">
          <h2 className="mb-4">My Projects</h2>
          <div className="h-1 w-20 bg-blue-600 mx-auto rounded-full"></div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Showcasing my key machine learning and data science projects
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {projects.map((project, index) => (
            <Card key={index} className="overflow-hidden card-hover">
              <div className="h-48 bg-muted overflow-hidden">
                <img
                  src={project.image}
                  alt={project.title}
                  className="w-full h-full object-cover transition-transform duration-300 hover:scale-105"
                />
              </div>
              <CardContent className="p-6">
                <h3 className="text-xl font-bold mb-3">{project.title}</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  {project.description}
                </p>
                <div className="flex flex-wrap gap-2">
                  {project.technologies.map((tech, techIndex) => (
                    <Badge key={techIndex} variant="outline" className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-blue-100 dark:border-blue-800">
                      {tech}
                    </Badge>
                  ))}
                </div>
              </CardContent>
              {project.link && (
                <CardFooter className="p-6 pt-0">
                  <a
                    href={project.link}
                    className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium"
                  >
                    View Project <ArrowRight className="ml-2 h-4 w-4" />
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
