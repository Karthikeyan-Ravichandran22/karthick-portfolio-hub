
import { Card, CardContent } from "@/components/ui/card";
import { Briefcase, Code, User } from "lucide-react";

const About = () => {
  return (
    <section id="about" className="bg-white dark:bg-gray-900">
      <div className="section-container">
        <div className="text-center mb-16">
          <h2 className="mb-4">About Me</h2>
          <div className="h-1 w-20 bg-blue-600 mx-auto rounded-full"></div>
        </div>

        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div>
            <h3 className="mb-6">
              Machine Learning Expert with a Passion for Innovation
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-6">
              I specialize in developing cutting-edge machine learning solutions that solve complex business problems. With expertise across the entire ML lifecycle—from data preparation to model deployment and monitoring—I deliver scalable, production-ready solutions that drive tangible business value.
            </p>
            <p className="text-gray-600 dark:text-gray-300">
              My experience spans multiple industries including telecommunications, education, legal tech, and healthcare, where I've consistently delivered projects that improve efficiency, accuracy, and decision-making capabilities.
            </p>
          </div>

          <div className="grid gap-6">
            <Card className="card-hover">
              <CardContent className="p-6 flex items-start gap-4">
                <div className="bg-blue-100 dark:bg-blue-900/50 p-3 rounded-lg">
                  <User className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2">Who I Am</h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    A data scientist and ML engineer with a strong background in predictive modeling, NLP, and deep learning, dedicated to turning data into strategic advantages.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="card-hover">
              <CardContent className="p-6 flex items-start gap-4">
                <div className="bg-teal-100 dark:bg-teal-900/50 p-3 rounded-lg">
                  <Code className="h-6 w-6 text-teal-600 dark:text-teal-400" />
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2">What I Do</h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    I build and deploy machine learning models that solve real business problems, from customer retention prediction to document analysis and automated insights.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card className="card-hover">
              <CardContent className="p-6 flex items-start gap-4">
                <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-lg">
                  <Briefcase className="h-6 w-6 text-gray-600 dark:text-gray-400" />
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2">My Experience</h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    Over 6 years of professional experience in data science and machine learning roles, working with diverse technologies and delivering impactful solutions.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default About;
