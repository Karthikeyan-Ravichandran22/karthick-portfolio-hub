
import { ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";

const Hero = () => {
  const [animateBackground, setAnimateBackground] = useState(false);
  
  useEffect(() => {
    // Start animation after component mounts
    setAnimateBackground(true);
  }, []);

  return (
    <section
      id="home"
      className="min-h-screen flex flex-col justify-center relative overflow-hidden pt-16"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-teal-50 dark:from-blue-950/20 dark:to-teal-950/20 -z-10"></div>
      
      {/* Enhanced animated background */}
      <div className="absolute inset-0 -z-10">
        <div className={`absolute top-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl ${animateBackground ? 'animate-pulse' : ''}`}></div>
        <div className={`absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl ${animateBackground ? 'animate-pulse' : ''}`} style={{ animationDelay: "1s" }}></div>
        <div className={`absolute top-1/3 left-1/4 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl ${animateBackground ? 'animate-pulse' : ''}`} style={{ animationDelay: "2s" }}></div>
        
        {/* New animated elements */}
        <div className={`absolute bottom-1/4 right-1/3 w-72 h-72 bg-pink-500/10 rounded-full blur-3xl ${animateBackground ? 'animate-pulse' : ''}`} style={{ animationDelay: "1.5s" }}></div>
        <div className={`absolute top-2/3 right-1/5 w-48 h-48 bg-yellow-500/10 rounded-full blur-3xl ${animateBackground ? 'animate-pulse' : ''}`} style={{ animationDelay: "0.7s" }}></div>
      </div>

      <div className="section-container">
        <div className="max-w-3xl mx-auto text-center">
          <div className="relative inline-block mb-6">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-teal-400 blur-xl opacity-30 rounded-full"></div>
            <span className="relative bg-gradient-to-r from-blue-600 to-teal-500 px-6 py-2 text-white text-sm font-medium rounded-full shadow-lg transform transition-transform hover:scale-105 duration-300">
              Available for Projects
            </span>
          </div>
          
          <h1 className="animate-fade-in mb-2">
            <span className="block text-gray-800 dark:text-gray-200">Machine Learning &</span>
            <span className="gradient-text bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 bg-clip-text text-transparent">
              Generative AI Engineer
            </span>
          </h1>

          {/* Creative text animation */}
          <div className="h-16 mt-2 mb-6 relative overflow-hidden">
            <div className="absolute inset-0 flex flex-col animate-marquee">
              <span className="text-xl font-medium text-blue-600 dark:text-blue-400 py-4">LLM Specialist</span>
              <span className="text-xl font-medium text-teal-600 dark:text-teal-400 py-4">RAG Systems Developer</span>
              <span className="text-xl font-medium text-purple-600 dark:text-purple-400 py-4">AI Agent Builder</span>
              <span className="text-xl font-medium text-blue-600 dark:text-blue-400 py-4">LLM Specialist</span>
            </div>
          </div>

          <p className="mt-6 text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
            Transforming complex data challenges into strategic business advantages through innovative machine learning and generative AI solutions.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row justify-center gap-4 animate-fade-in-up" style={{ animationDelay: "0.4s" }}>
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105" asChild>
              <a href="#contact">Hire Me</a>
            </Button>
            <Button variant="outline" size="lg" className="border-2 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all duration-300 transform hover:scale-105" asChild>
              <a href="#projects">View My Work</a>
            </Button>
          </div>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <a href="#about" aria-label="Scroll down" className="hover:text-blue-600 transition-colors group">
          <div className="p-2 rounded-full bg-white/50 backdrop-blur-sm shadow-md group-hover:bg-blue-100 transition-colors duration-300">
            <ArrowDown className="text-gray-500 dark:text-gray-400 group-hover:text-blue-600 transition-colors duration-300" />
          </div>
        </a>
      </div>
    </section>
  );
};

export default Hero;
