
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Mail, MessageSquare, Send, Linkedin, Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const Contact = () => {
  const { toast } = useToast();
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    subject: "",
    message: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate form submission
    setTimeout(() => {
      toast({
        title: "Message sent!",
        description: "Thanks for reaching out. I'll get back to you soon.",
      });
      
      setFormData({
        name: "",
        email: "",
        subject: "",
        message: "",
      });
      setIsSubmitting(false);
    }, 1500);
  };

  return (
    <section id="contact" className="relative overflow-hidden bg-white dark:bg-gray-900">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-blue-100/40 dark:bg-blue-900/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/3 left-1/3 w-96 h-96 bg-teal-100/40 dark:bg-teal-900/10 rounded-full blur-3xl"></div>
      </div>
      
      <div className="section-container relative z-10">
        <div className="text-center mb-16">
          <h2 className="mb-4">Get In Touch</h2>
          <div className="h-1 w-20 bg-gradient-to-r from-blue-600 to-teal-500 mx-auto rounded-full"></div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Have a project in mind or interested in working together? I'd love to hear from you!
          </p>
        </div>

        <div className="grid lg:grid-cols-5 gap-12 max-w-5xl mx-auto">
          <div className="lg:col-span-2 space-y-8">
            <Card className="overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300">
              <div className="h-1 bg-gradient-to-r from-blue-600 to-teal-500"></div>
              <CardContent className="p-6 flex items-start gap-4">
                <div className="bg-blue-100 dark:bg-blue-900/50 p-3 rounded-lg">
                  <Mail className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2">Email</h4>
                  <a 
                    href="mailto:karthikeyan.ravichandran22@gmail.com" 
                    className="text-blue-600 dark:text-blue-400 hover:underline transition-all"
                  >
                    karthikeyan.ravichandran22@gmail.com
                  </a>
                </div>
              </CardContent>
            </Card>

            <Card className="overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300">
              <div className="h-1 bg-gradient-to-r from-blue-600 to-teal-500"></div>
              <CardContent className="p-6 flex items-start gap-4">
                <div className="bg-teal-100 dark:bg-teal-900/50 p-3 rounded-lg">
                  <Linkedin className="h-6 w-6 text-teal-600 dark:text-teal-400" />
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2">LinkedIn</h4>
                  <a 
                    href="https://www.linkedin.com/in/karthikeyan-ravichandran/" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-teal-600 dark:text-teal-400 hover:underline transition-all"
                  >
                    Connect with me
                  </a>
                </div>
              </CardContent>
            </Card>

            <Card className="overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300">
              <div className="h-1 bg-gradient-to-r from-blue-600 to-teal-500"></div>
              <CardContent className="p-6 flex items-start gap-4">
                <div className="bg-purple-100 dark:bg-purple-900/50 p-3 rounded-lg">
                  <Sparkles className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                </div>
                <div>
                  <h4 className="text-xl font-semibold mb-2">AI Solutions</h4>
                  <p className="text-gray-600 dark:text-gray-300">
                    Specialized in custom AI solutions for businesses looking to innovate with machine learning and generative AI technologies.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="lg:col-span-3">
            <Card className="overflow-hidden border-0 shadow-lg">
              <div className="h-1 bg-gradient-to-r from-blue-600 to-teal-500"></div>
              <CardContent className="p-6">
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <label htmlFor="name" className="text-sm font-medium">
                        Name
                      </label>
                      <Input
                        id="name"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        placeholder="Your name"
                        required
                        className="border-gray-200 focus:border-blue-500 transition-colors"
                      />
                    </div>
                    <div className="space-y-2">
                      <label htmlFor="email" className="text-sm font-medium">
                        Email
                      </label>
                      <Input
                        id="email"
                        name="email"
                        type="email"
                        value={formData.email}
                        onChange={handleChange}
                        placeholder="Your email"
                        required
                        className="border-gray-200 focus:border-blue-500 transition-colors"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="subject" className="text-sm font-medium">
                      Subject
                    </label>
                    <Input
                      id="subject"
                      name="subject"
                      value={formData.subject}
                      onChange={handleChange}
                      placeholder="What is this regarding?"
                      required
                      className="border-gray-200 focus:border-blue-500 transition-colors"
                    />
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="message" className="text-sm font-medium">
                      Message
                    </label>
                    <Textarea
                      id="message"
                      name="message"
                      value={formData.message}
                      onChange={handleChange}
                      placeholder="Your message"
                      rows={6}
                      required
                      className="border-gray-200 focus:border-blue-500 transition-colors"
                    />
                  </div>

                  <Button 
                    type="submit" 
                    className="w-full bg-gradient-to-r from-blue-600 to-teal-500 hover:from-blue-700 hover:to-teal-600 transition-all duration-300"
                    disabled={isSubmitting}
                  >
                    {isSubmitting ? (
                      <span className="flex items-center">
                        Sending...
                      </span>
                    ) : (
                      <span className="flex items-center">
                        Send Message <Send className="ml-2 h-4 w-4" />
                      </span>
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
