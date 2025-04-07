
import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

type Particle = {
  x: number;
  y: number;
  size: number;
  speedX: number;
  speedY: number;
  color: string;
};

const AnimatedBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particles = useRef<Particle[]>([]);
  const mousePosition = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const createParticles = () => {
      particles.current = [];
      const particleCount = Math.min(window.innerWidth * 0.05, 100);
      const colors = ["#3b82f6", "#14b8a6", "#8b5cf6", "#ec4899"];

      for (let i = 0; i < particleCount; i++) {
        particles.current.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: Math.random() * 2 + 1,
          speedX: (Math.random() - 0.5) * 0.5,
          speedY: (Math.random() - 0.5) * 0.5,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      mousePosition.current = {
        x: e.clientX,
        y: e.clientY,
      };
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      particles.current.forEach((particle, index) => {
        // Update position
        particle.x += particle.speedX;
        particle.y += particle.speedY;
        
        // Boundary check
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.y > canvas.height) particle.y = 0;
        if (particle.y < 0) particle.y = canvas.height;
        
        // Mouse interaction
        if (mousePosition.current) {
          const dx = mousePosition.current.x - particle.x;
          const dy = mousePosition.current.y - particle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 120) {
            const angle = Math.atan2(dy, dx);
            const force = (120 - distance) / 120;
            
            particle.speedX -= Math.cos(angle) * force * 0.02;
            particle.speedY -= Math.sin(angle) * force * 0.02;
          }
        }
        
        // Draw particle
        ctx.fillStyle = particle.color;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fill();
        
        // Connect nearby particles
        for (let j = index + 1; j < particles.current.length; j++) {
          const otherParticle = particles.current[j];
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          if (distance < 100) {
            ctx.beginPath();
            ctx.strokeStyle = particle.color;
            ctx.globalAlpha = 0.2 * (1 - distance / 100);
            ctx.lineWidth = 0.5;
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.stroke();
          }
        }
      });
      
      requestAnimationFrame(animate);
    };

    window.addEventListener("resize", resizeCanvas);
    window.addEventListener("mousemove", handleMouseMove);
    
    resizeCanvas();
    createParticles();
    animate();
    
    return () => {
      window.removeEventListener("resize", resizeCanvas);
      window.removeEventListener("mousemove", handleMouseMove);
    };
  }, []);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="fixed inset-0 z-0 pointer-events-none"
      />
      <motion.div 
        className="fixed inset-0 bg-gradient-to-br from-blue-50/30 to-teal-50/30 dark:from-blue-950/30 dark:to-teal-950/30 z-0 pointer-events-none"
        animate={{ 
          background: [
            "radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.03) 0%, rgba(20, 184, 166, 0.03) 100%)",
            "radial-gradient(circle at 80% 80%, rgba(59, 130, 246, 0.03) 0%, rgba(20, 184, 166, 0.03) 100%)",
            "radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.03) 0%, rgba(20, 184, 166, 0.03) 100%)",
            "radial-gradient(circle at 80% 20%, rgba(59, 130, 246, 0.03) 0%, rgba(20, 184, 166, 0.03) 100%)",
            "radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.03) 0%, rgba(20, 184, 166, 0.03) 100%)",
          ]
        }}
        transition={{ duration: 20, ease: "linear", repeat: Infinity }}
      />
    </>
  );
};

export default AnimatedBackground;
