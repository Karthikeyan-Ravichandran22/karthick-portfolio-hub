
import { useEffect, useRef } from "react";
import { motion, useAnimation, useInView } from "framer-motion";

interface TextRevealProps {
  text: string;
  className?: string;
  delay?: number;
  duration?: number;
  once?: boolean;
  highlightWords?: string[];
}

const TextReveal = ({
  text,
  className = "",
  delay = 0,
  duration = 0.05,
  once = true,
  highlightWords = [],
}: TextRevealProps) => {
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once });
  const controls = useAnimation();
  
  const words = text.split(" ");
  
  useEffect(() => {
    if (isInView) {
      controls.start("visible");
    } else if (!once) {
      controls.start("hidden");
    }
  }, [isInView, controls, once]);
  
  const shouldHighlight = (word: string): boolean => {
    return highlightWords.includes(word);
  };
  
  return (
    <motion.p 
      ref={ref}
      className={`inline-block ${className}`}
      initial="hidden"
      animate={controls}
      variants={{
        visible: {
          transition: {
            staggerChildren: duration,
            delayChildren: delay,
          },
        },
        hidden: {},
      }}
    >
      {words.map((word, i) => (
        <motion.span
          key={i}
          className="inline-block whitespace-pre"
          variants={{
            visible: {
              opacity: 1,
              y: 0,
              transition: {
                type: "spring",
                damping: 12,
                stiffness: 100,
              },
            },
            hidden: {
              opacity: 0,
              y: 20,
              transition: {
                type: "spring",
                damping: 12,
                stiffness: 100,
              },
            },
          }}
        >
          <span className={shouldHighlight(word) ? "font-bold gradient-text" : ""}>
            {word}
          </span>
          {i !== words.length - 1 ? " " : ""}
        </motion.span>
      ))}
    </motion.p>
  );
};

export default TextReveal;
