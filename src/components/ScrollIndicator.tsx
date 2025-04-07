
import { motion, useScroll } from "framer-motion";

const ScrollIndicator = () => {
  const { scrollYProgress } = useScroll();
  
  return (
    <>
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-600 via-teal-500 to-purple-600 z-50"
        style={{ scaleX: scrollYProgress, transformOrigin: "0%" }}
      />
      <motion.div
        className="fixed bottom-4 right-4 w-12 h-12 rounded-full bg-white dark:bg-gray-900 shadow-lg flex items-center justify-center z-40 overflow-hidden"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        transition={{ type: "spring", stiffness: 400, damping: 10 }}
      >
        <svg className="w-8 h-8" viewBox="0 0 100 100">
          <motion.circle
            className="text-gray-200 dark:text-gray-800"
            cx="50"
            cy="50"
            r="30"
            pathLength="1"
            strokeWidth="8"
            stroke="currentColor"
            fill="none"
          />
          <motion.circle
            className="text-blue-600 dark:text-blue-400"
            cx="50"
            cy="50"
            r="30"
            pathLength="1"
            strokeWidth="8"
            stroke="currentColor"
            fill="none"
            style={{ pathLength: scrollYProgress }}
          />
        </svg>
        <motion.div
          className="absolute text-xs font-medium text-gray-800 dark:text-gray-200"
          style={{
            opacity: scrollYProgress
          }}
        >
          <motion.span
            style={{
              opacity: scrollYProgress
            }}
          >
            {Math.round(Number(scrollYProgress) * 100)}%
          </motion.span>
        </motion.div>
      </motion.div>
    </>
  );
};

export default ScrollIndicator;
