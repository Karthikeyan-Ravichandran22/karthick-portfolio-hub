
import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				},
				blue: {
					50: '#f0f9ff',
					100: '#e0f2fe',
					200: '#bae6fd',
					300: '#7dd3fc',
					400: '#38bdf8',
					500: '#0ea5e9',
					600: '#0284c7',
					700: '#0369a1',
					800: '#075985',
					900: '#0c4a6e',
				},
				teal: {
					50: '#f0fdfa',
					100: '#ccfbf1',
					200: '#99f6e4',
					300: '#5eead4',
					400: '#2dd4bf',
					500: '#14b8a6',
					600: '#0d9488',
					700: '#0f766e',
					800: '#115e59',
					900: '#134e4a',
				},
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: { height: '0' },
					to: { height: 'var(--radix-accordion-content-height)' }
				},
				'accordion-up': {
					from: { height: 'var(--radix-accordion-content-height)' },
					to: { height: '0' }
				},
				'fade-in': {
					'0%': { opacity: '0' },
					'100%': { opacity: '1' }
				},
				'fade-in-up': {
					'0%': { 
						opacity: '0',
						transform: 'translateY(20px)'
					},
					'100%': { 
						opacity: '1',
						transform: 'translateY(0)'
					}
				},
				'float': {
					'0%, 100%': { transform: 'translateY(0)' },
					'50%': { transform: 'translateY(-10px)' }
				},
				'background-shine': {
					'0%': { backgroundPosition: '0% 0%' },
					'100%': { backgroundPosition: '-200% 0%' }
				},
				'shimmer': {
					'100%': { transform: 'translateX(100%)' }
				},
				'spin-slow': {
					'0%': { transform: 'rotate(0deg)' },
					'100%': { transform: 'rotate(360deg)' }
				},
				'bg-pan': {
					'0%': { backgroundPosition: '0% center' },
					'100%': { backgroundPosition: '-200% center' }
				},
				'glitch': {
					'0%': { transform: 'translate(0)' },
					'20%': { transform: 'translate(-2px, 2px)' },
					'40%': { transform: 'translate(-2px, -2px)' },
					'60%': { transform: 'translate(2px, 2px)' },
					'80%': { transform: 'translate(2px, -2px)' },
					'100%': { transform: 'translate(0)' }
				},
				'text-glow': {
					'0%, 100%': { 
						textShadow: '0 0 5px rgba(59, 130, 246, 0.5), 0 0 15px rgba(59, 130, 246, 0.3), 0 0 20px rgba(59, 130, 246, 0.2)' 
					},
					'50%': { 
						textShadow: '0 0 10px rgba(59, 130, 246, 0.8), 0 0 30px rgba(59, 130, 246, 0.5), 0 0 40px rgba(59, 130, 246, 0.3)' 
					}
				},
				'gradient-shift': {
					'0%': { backgroundPosition: '0% 50%' },
					'50%': { backgroundPosition: '100% 50%' },
					'100%': { backgroundPosition: '0% 50%' }
				},
				'rotate-3d': {
					'0%': { transform: 'perspective(1000px) rotateY(0deg)' },
					'100%': { transform: 'perspective(1000px) rotateY(360deg)' }
				},
				'particle-fade': {
					'0%': { transform: 'scale(0) translate(0, 0)', opacity: '1' },
					'100%': { transform: 'scale(1) translate(var(--tx), var(--ty))', opacity: '0' }
				},
				'border-pulse': {
					'0%, 100%': { borderColor: 'rgba(59, 130, 246, 0.5)' },
					'50%': { borderColor: 'rgba(20, 184, 166, 0.8)' }
				},
				'wave': {
					'0%': { backgroundPositionX: '0' },
					'100%': { backgroundPositionX: '100vw' }
				},
				'sparkle-animation': {
					'0%': { transform: 'scale(0) rotate(0deg)', opacity: '0' },
					'50%': { opacity: '1' },
					'100%': { transform: 'scale(1) rotate(180deg)', opacity: '0' }
				},
				'bounce-subtle': {
					'0%, 100%': { transform: 'translateY(0)' },
					'50%': { transform: 'translateY(-5px)' }
				},
				'scale-pulse': {
					'0%, 100%': { transform: 'scale(1)' },
					'50%': { transform: 'scale(1.05)' }
				},
				'move-left-right': {
					'0%, 100%': { transform: 'translateX(0)' },
					'50%': { transform: 'translateX(10px)' }
				},
				'rotate-center': {
					'0%': { transform: 'rotate(0)' },
					'100%': { transform: 'rotate(360deg)' }
				},
				'text-focus': {
					'0%': { filter: 'blur(4px)', opacity: '0' },
					'100%': { filter: 'blur(0)', opacity: '1' }
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'fade-in': 'fade-in 0.5s ease-out',
				'fade-in-up': 'fade-in-up 0.7s ease-out',
				'float': 'float 6s ease-in-out infinite',
				'background-shine': 'background-shine 8s linear infinite',
				'shimmer': 'shimmer 2s infinite',
				'spin-slow': 'spin-slow 8s linear infinite',
				'bg-pan': 'bg-pan 30s linear infinite',
				'glitch': 'glitch 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94) both infinite',
				'text-glow': 'text-glow 3s ease-in-out infinite alternate',
				'gradient-shift': 'gradient-shift 8s ease infinite',
				'rotate-3d': 'rotate-3d 20s linear infinite',
				'particle-fade': 'particle-fade 1s forwards ease-out',
				'border-pulse': 'border-pulse 3s infinite',
				'wave': 'wave 20s linear infinite',
				'sparkle-animation': 'sparkle-animation 1s forwards',
				'bounce-subtle': 'bounce-subtle 2s ease-in-out infinite',
				'scale-pulse': 'scale-pulse 2s ease-in-out infinite',
				'move-left-right': 'move-left-right 3s ease-in-out infinite',
				'rotate-center': 'rotate-center 8s linear infinite',
				'text-focus': 'text-focus 0.5s cubic-bezier(0.550, 0.085, 0.680, 0.530) forwards'
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
