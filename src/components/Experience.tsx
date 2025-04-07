
import { 
  Accordion, 
  AccordionContent, 
  AccordionItem, 
  AccordionTrigger 
} from "@/components/ui/accordion";
import { experiences } from "@/data/experienceData";

const Experience = () => {
  return (
    <section id="experience" className="bg-gray-50 dark:bg-gray-800">
      <div className="section-container">
        <div className="text-center mb-16">
          <h2 className="mb-4">Work Experience</h2>
          <div className="h-1 w-20 bg-blue-600 mx-auto rounded-full"></div>
          <p className="mt-6 text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            My professional journey as a machine learning specialist and data scientist
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="timeline">
            {experiences.map((exp, index) => (
              <div key={index} className="timeline-item">
                <div className="mb-2">
                  <h3 className="text-xl md:text-2xl font-bold">{exp.title}</h3>
                  <div className="text-blue-600 dark:text-blue-400 font-medium">{exp.company}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">{exp.period}</div>
                </div>
                
                <Accordion type="single" collapsible className="mt-4 border-none">
                  <AccordionItem value={`item-${index}`} className="border-none">
                    <AccordionTrigger className="py-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:no-underline">
                      Key Responsibilities & Achievements
                    </AccordionTrigger>
                    <AccordionContent>
                      <ul className="list-disc pl-5 space-y-2 text-gray-600 dark:text-gray-300 mt-2">
                        {exp.responsibilities.map((resp, idx) => (
                          <li key={idx}>{resp}</li>
                        ))}
                      </ul>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Experience;
