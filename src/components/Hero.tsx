
import { ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";

const Hero = () => {
  return (
    <section
      id="home"
      className="min-h-screen flex flex-col justify-center relative overflow-hidden pt-16"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-teal-50 dark:from-blue-950/20 dark:to-teal-950/20 -z-10"></div>
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl"></div>
      </div>

      <div className="section-container">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="animate-fade-in">
            <span className="block">Machine Learning</span>
            <span className="gradient-text">Expert & Data Scientist</span>
          </h1>

          <p className="mt-6 text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
            Transforming complex data challenges into strategic business advantages through innovative machine learning solutions and data-driven insights.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row justify-center gap-4 animate-fade-in-up" style={{ animationDelay: "0.4s" }}>
            <Button size="lg" asChild>
              <a href="#contact">Hire Me</a>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <a href="#projects">View My Work</a>
            </Button>
          </div>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <a href="#about" aria-label="Scroll down">
          <ArrowDown className="text-gray-500 dark:text-gray-400" />
        </a>
      </div>
    </section>
  );
};

export default Hero;
