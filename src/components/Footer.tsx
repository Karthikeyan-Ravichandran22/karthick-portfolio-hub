
import { Github, Linkedin, Mail, Twitter } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gradient-to-br from-gray-900 to-gray-800 text-white py-12">
      <div className="section-container">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div>
            <h3 className="text-xl font-bold bg-gradient-to-r from-blue-400 via-teal-400 to-purple-400 bg-clip-text text-transparent">Karthikeyan Ravichandran</h3>
            <p className="mt-2 text-gray-400 max-w-md">
              Machine Learning & Generative AI Expert. Transforming data into business value.
            </p>
          </div>

          <div className="flex gap-4">
            <a
              href="https://www.linkedin.com/in/karthikeyan-ravichandran/"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 bg-gray-800 hover:bg-blue-600 rounded-full transition-colors duration-300"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-5 w-5" />
            </a>
            <a
              href="https://github.com/"
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2 bg-gray-800 hover:bg-gray-700 rounded-full transition-colors duration-300"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>
            <a
              href="https://twitter.com/"
              target="_blank" 
              rel="noopener noreferrer"
              className="p-2 bg-gray-800 hover:bg-blue-400 rounded-full transition-colors duration-300"
              aria-label="Twitter"
            >
              <Twitter className="h-5 w-5" />
            </a>
            <a
              href="mailto:karthikeyan.ravichandran22@gmail.com"
              className="p-2 bg-gray-800 hover:bg-green-600 rounded-full transition-colors duration-300"
              aria-label="Email"
            >
              <Mail className="h-5 w-5" />
            </a>
          </div>
        </div>

        <hr className="border-gray-800 my-8" />

        <div className="flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">
            © {currentYear} Karthikeyan Ravichandran. All rights reserved.
          </p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">
              Privacy Policy
            </a>
            <a href="#" className="text-gray-400 hover:text-white text-sm transition-colors">
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
