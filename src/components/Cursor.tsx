
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const Cursor = () => {
  const [position, setPosition] = useState({ x: -100, y: -100 });
  const [clicked, setClicked] = useState(false);
  const [linkHovered, setLinkHovered] = useState(false);
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const addEventListeners = () => {
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mousedown", onMouseDown);
      document.addEventListener("mouseup", onMouseUp);
      document.addEventListener("mouseenter", onMouseEnter);
      document.addEventListener("mouseleave", onMouseLeave);
    };

    const removeEventListeners = () => {
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("mouseup", onMouseUp);
      document.removeEventListener("mouseenter", onMouseEnter);
      document.removeEventListener("mouseleave", onMouseLeave);
    };

    const onMouseMove = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
      
      const hoveredElement = document.elementFromPoint(e.clientX, e.clientY);
      const isLinkOrButton = 
        hoveredElement instanceof HTMLAnchorElement || 
        hoveredElement instanceof HTMLButtonElement || 
        hoveredElement?.closest('a') !== null || 
        hoveredElement?.closest('button') !== null;
      
      setLinkHovered(isLinkOrButton);
    };

    const onMouseDown = () => {
      setClicked(true);
    };

    const onMouseUp = () => {
      setClicked(false);
    };

    const onMouseLeave = () => {
      setHidden(true);
    };

    const onMouseEnter = () => {
      setHidden(false);
    };

    addEventListeners();
    return () => removeEventListeners();
  }, []);

  const variants = {
    default: {
      x: position.x - 16,
      y: position.y - 16,
      opacity: hidden ? 0 : 0.5
    },
    clicked: {
      height: 12,
      width: 12,
      x: position.x - 6,
      y: position.y - 6,
      backgroundColor: "#3b82f6",
      opacity: hidden ? 0 : 0.8,
      mixBlendMode: "difference" as "difference"
    },
    hovered: {
      height: 48,
      width: 48,
      x: position.x - 24,
      y: position.y - 24,
      backgroundColor: "rgba(59, 130, 246, 0.3)",
      opacity: hidden ? 0 : 0.6,
      mixBlendMode: "difference" as "difference"
    }
  };

  const springConfig = { damping: 25, stiffness: 700 };

  // Only show on non-touch devices
  if (typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches) {
    return null;
  }

  return (
    <>
      <motion.div
        className="custom-cursor-dot fixed top-0 left-0 w-3 h-3 bg-blue-600 rounded-full pointer-events-none z-50"
        animate={clicked ? "clicked" : "default"}
        variants={variants}
        transition={{ type: "spring", ...springConfig }}
      />
      <motion.div
        className="custom-cursor-ring fixed top-0 left-0 w-8 h-8 rounded-full border-2 border-blue-500 pointer-events-none z-50"
        animate={linkHovered ? "hovered" : "default"}
        variants={variants}
        transition={{ type: "spring", ...springConfig, mass: 0.5 }}
      />
    </>
  );
};

export default Cursor;
