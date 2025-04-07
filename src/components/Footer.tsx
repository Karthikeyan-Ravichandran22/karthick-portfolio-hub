
import { Github, Linkedin, Mail } from "lucide-react";

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="section-container">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div>
            <h3 className="text-xl font-bold gradient-text">ML Expert</h3>
            <p className="mt-2 text-gray-400 max-w-md">
              Machine Learning Solutions Expert. Transforming data into business value.
            </p>
          </div>

          <div className="flex gap-4">
            <a
              href="#"
              className="p-2 bg-gray-800 hover:bg-blue-600 rounded-full transition-colors"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-5 w-5" />
            </a>
            <a
              href="#"
              className="p-2 bg-gray-800 hover:bg-blue-600 rounded-full transition-colors"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>
            <a
              href="mailto:contact@mlexpert.dev"
              className="p-2 bg-gray-800 hover:bg-blue-600 rounded-full transition-colors"
              aria-label="Email"
            >
              <Mail className="h-5 w-5" />
            </a>
          </div>
        </div>

        <hr className="border-gray-800 my-8" />

        <div className="flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">
            © {currentYear} ML Expert. All rights reserved.
          </p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <a href="#" className="text-gray-400 hover:text-white text-sm">
              Privacy Policy
            </a>
            <a href="#" className="text-gray-400 hover:text-white text-sm">
              Terms of Service
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
