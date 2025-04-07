
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface ProgressiveImageProps {
  src: string;
  alt: string;
  className?: string;
  placeholderColor?: string;
}

const ProgressiveImage: React.FC<ProgressiveImageProps> = ({
  src,
  alt,
  className = '',
  placeholderColor = '#e2e8f0'
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isInView, setIsInView] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsInView(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    const img = new Image();
    
    if (isInView) {
      img.src = src;
      img.onload = () => {
        setIsLoaded(true);
      };
    }

    const currentRef = document.getElementById(`progressive-img-${src.replace(/[^a-zA-Z0-9]/g, '')}`);
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      observer.disconnect();
    };
  }, [src, isInView]);

  return (
    <div 
      id={`progressive-img-${src.replace(/[^a-zA-Z0-9]/g, '')}`}
      className={`relative overflow-hidden ${className}`}
    >
      {/* Placeholder */}
      <div
        className="absolute inset-0 transition-opacity duration-300"
        style={{ 
          backgroundColor: placeholderColor,
          opacity: isLoaded ? 0 : 1
        }}
      />
      
      {/* Blur-up effect */}
      <motion.div 
        className="absolute inset-0 blur-xl"
        animate={{ 
          opacity: isLoaded ? 0 : 1,
          scale: isLoaded ? 1.1 : 1
        }}
        initial={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        style={{
          backgroundImage: `url(${src})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center'
        }}
      />
      
      {/* Actual image */}
      <motion.img
        src={isInView ? src : ''}
        alt={alt}
        className="w-full h-full object-cover transition-opacity duration-500"
        initial={{ opacity: 0 }}
        animate={{ opacity: isLoaded ? 1 : 0 }}
        transition={{ duration: 0.5 }}
        onLoad={() => setIsLoaded(true)}
      />
      
      {/* Shimmer effect */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent"
        style={{ 
          opacity: isLoaded ? 0 : 0.2,
          backgroundSize: '200% 100%'
        }}
        animate={{
          backgroundPosition: ['100% 0%', '-100% 0%']
        }}
        transition={{
          repeat: Infinity,
          duration: 1.5,
          ease: "linear"
        }}
      />
    </div>
  );
};

export default ProgressiveImage;
