
import { useRef, useEffect } from "react";
import { motion, useAnimation } from "framer-motion";
import { 
  BrainCircuit, Database, Cpu, Code, LineChart, Globe, 
  FileCode, Network, BarChart, Sparkles 
} from "lucide-react";

type FloatingIcon = {
  id: number;
  Component: any;
  x: number;
  y: number;
  size: number;
  delay: number;
  duration: number;
}

const FloatingIcons = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const controls = useAnimation();
  const iconsRef = useRef<FloatingIcon[]>([]);

  useEffect(() => {
    if (!containerRef.current) return;
    
    const icons = [
      BrainCircuit, Database, Cpu, Code, LineChart, 
      Globe, FileCode, Network, BarChart
    ];
    
    const generateIcons = () => {
      const newIcons: FloatingIcon[] = [];
      const containerWidth = containerRef.current?.offsetWidth || 0;
      const containerHeight = containerRef.current?.offsetHeight || 0;
      
      for (let i = 0; i < 15; i++) {
        newIcons.push({
          id: i,
          Component: icons[Math.floor(Math.random() * icons.length)],
          x: Math.random() * containerWidth,
          y: Math.random() * containerHeight,
          size: Math.random() * (30 - 15) + 15,
          delay: Math.random() * 5,
          duration: Math.random() * (15 - 8) + 8
        });
      }
      
      iconsRef.current = newIcons;
      controls.start("animate");
    };

    generateIcons();
    
    const handleResize = () => {
      generateIcons();
    };
    
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [controls]);

  return (
    <div 
      ref={containerRef} 
      className="absolute inset-0 overflow-hidden pointer-events-none z-0"
    >
      {iconsRef.current.map((icon) => (
        <motion.div
          key={icon.id}
          className="absolute opacity-10 text-blue-500 dark:text-blue-400"
          initial={{ 
            x: icon.x, 
            y: icon.y, 
            scale: 0,
            opacity: 0 
          }}
          animate={{ 
            x: [icon.x, icon.x + (Math.random() - 0.5) * 100, icon.x],
            y: [icon.y, icon.y + (Math.random() - 0.5) * 100, icon.y],
            scale: [0, 1, 0.8, 1],
            opacity: [0, 0.1, 0.15, 0.1]
          }}
          transition={{ 
            duration: icon.duration,
            delay: icon.delay,
            repeat: Infinity,
            repeatType: "reverse"
          }}
          style={{
            width: icon.size,
            height: icon.size
          }}
        >
          <icon.Component size={icon.size} />
        </motion.div>
      ))}
      
      {/* Occasional sparkle effect */}
      <motion.div
        className="absolute"
        initial={{
          opacity: 0,
          scale: 0.5,
          x: 0,
          y: 0
        }}
        animate={{ 
          opacity: [0, 0.7, 0],
          scale: [0.5, 1.5, 0.5],
          x: ["0%", "100%"],
          y: ["0%", "100%"]
        }}
        transition={{ 
          duration: 4,
          repeat: Infinity,
          repeatType: "loop",
          repeatDelay: 10,
          times: [0, 0.5, 1]
        }}
      >
        <Sparkles className="w-6 h-6 text-yellow-400" />
      </motion.div>
    </div>
  );
};

export default FloatingIcons;
