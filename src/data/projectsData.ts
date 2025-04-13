
export const projects = [
  {
    title: "AI-Powered Financial Advisor",
    description: "Personal finance management system leveraging GPT-4 to provide customized investment advice, budget optimization, and financial goal planning.",
    image: "https://images.unsplash.com/photo-1579621970588-a35d0e7ab9b6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    technologies: ["Python", "TensorFlow", "GPT-4", "React", "Flask"],
    github: "https://github.com/user/ai-financial-advisor",
    link: "https://ai-finance-advisor.netlify.app",
    readme: "# AI-Powered Financial Advisor\n\n## Overview\nThis application helps users optimize their finances through AI-driven insights and recommendations.\n\n## Features\n- Personalized investment strategies\n- Budget optimization\n- Financial goal planning\n- Risk assessment\n- Market trend analysis\n\n## Technologies\n- Python backend with Flask\n- TensorFlow for predictive modeling\n- GPT-4 for natural language processing\n- React frontend\n- MongoDB for data storage"
  },
  {
    title: "Neural Style Transfer App",
    description: "Web application that uses deep learning to apply artistic styles to user-uploaded images, leveraging VGG19 architecture for high-quality style transfers.",
    image: "https://images.unsplash.com/photo-1579546929518-9e396f3cc809?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    technologies: ["Python", "PyTorch", "TensorFlow", "React", "FastAPI"],
    github: "https://github.com/user/neural-style-transfer",
    link: "https://neural-style-app.vercel.app",
    readme: "# Neural Style Transfer Application\n\n## Overview\nThis application allows users to transform their photos into artwork by applying the style of famous paintings.\n\n## Features\n- Upload custom images\n- Choose from a library of artistic styles\n- Adjust style transfer intensity\n- Download high-resolution outputs\n- Share directly to social media\n\n## Technical Implementation\n- PyTorch implementation of Neural Style Transfer algorithm\n- VGG19 pre-trained network for feature extraction\n- React frontend with canvas manipulation\n- FastAPI backend for processing"
  },
  {
    title: "Language Learning with GPT",
    description: "Adaptive language learning platform that creates personalized exercises, provides real-time feedback, and simulates conversations based on user proficiency.",
    image: "https://images.unsplash.com/photo-1546410531-bb4caa6b424d?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    technologies: ["JavaScript", "React", "Node.js", "GPT-3.5", "MongoDB"],
    github: "https://github.com/user/gpt-language-learner",
    link: "https://language-gpt.herokuapp.com",
    readme: "# Language Learning with GPT\n\n## Purpose\nTo create a personalized language learning experience that adapts to each user's proficiency level and learning style.\n\n## Core Functionality\n- Personalized vocabulary exercises\n- Grammar correction with explanations\n- Simulated conversations with virtual native speakers\n- Cultural context for idioms and expressions\n- Progress tracking with spaced repetition\n\n## Implementation Details\n- GPT-3.5 for generating exercises and conversations\n- React frontend with speech recognition\n- Node.js backend\n- MongoDB for user profiles and progress tracking"
  },
  {
    title: "Emotion Recognition System",
    description: "Real-time emotion detection from facial expressions using computer vision and deep learning, capable of identifying seven basic emotional states.",
    image: "https://images.unsplash.com/photo-1470790376778-a9fbc86d70e2?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    technologies: ["Python", "OpenCV", "TensorFlow", "Keras", "Flask"],
    github: "https://github.com/user/emotion-recognition",
    link: "https://emotion-detector.onrender.com",
    readme: "# Emotion Recognition System\n\n## Overview\nThis system uses computer vision and deep learning to detect and classify human emotions in real-time from facial expressions.\n\n## Features\n- Real-time emotion detection from webcam\n- Support for 7 emotional states (happy, sad, angry, surprised, fearful, disgusted, neutral)\n- Emotion tracking over time\n- Optional video recording with emotion timestamps\n\n## Technical Details\n- Custom CNN architecture\n- Trained on FER2013 and CK+ datasets\n- OpenCV for face detection\n- Flask web interface with JavaScript visualization\n- Model achieves 72% accuracy on test data"
  },
  {
    title: "Generative Music Composer",
    description: "AI-driven music composition tool that creates original melodies and harmonies based on user-defined parameters such as genre, mood, and instrumentation.",
    image: "https://images.unsplash.com/photo-1507838153414-b4b713384a76?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    technologies: ["Python", "TensorFlow", "Magenta", "React", "Web Audio API"],
    github: "https://github.com/user/ai-music-composer",
    link: "https://generative-composer.netlify.app",
    readme: "# Generative Music Composer\n\n## Description\nThis application uses deep learning to generate original musical compositions based on user preferences.\n\n## Core Features\n- Generate melodies based on genre, mood, and tempo\n- Create chord progressions and harmonies\n- Customize instrumentation\n- Export to MIDI and MP3 formats\n- Save and edit compositions\n\n## Implementation\n- LSTM neural networks for melody generation\n- Transformer models for harmonic structure\n- TensorFlow.js for client-side generation\n- Web Audio API for playback\n- Trained on 10,000+ MIDI files across various genres"
  },
  {
    title: "Medical Image Diagnosis",
    description: "Deep learning system for automated analysis of medical images, assisting in early detection of conditions through pattern recognition in X-rays and MRIs.",
    image: "https://images.unsplash.com/photo-1530497610245-94d3c16cda28?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
    technologies: ["Python", "PyTorch", "fastai", "scikit-learn", "Flask"],
    github: "https://github.com/user/medical-image-ai",
    link: "https://medimage-diagnosis.onrender.com",
    readme: "# Medical Image Diagnosis\n\n## Purpose\nTo assist healthcare professionals in diagnosing medical conditions through automated analysis of X-rays, CT scans, and MRIs.\n\n## Capabilities\n- Pneumonia detection from chest X-rays\n- Brain tumor classification from MRI scans\n- Diabetic retinopathy grading\n- Bone fracture detection\n- Detailed probability scores with visual explanations\n\n## Technical Implementation\n- Transfer learning with ResNet50 and DenseNet architectures\n- Class activation mapping for localization\n- Trained on NIH Chest X-ray, ISIC, and ChestX-ray14 datasets\n- Achieves 91% sensitivity and 87% specificity on test datasets\n- Built with PyTorch and fastai\n\n## Ethical Considerations\n- Designed as an assistive tool, not a replacement for medical professionals\n- Includes confidence metrics and uncertainty estimation\n- Transparent decision-making process"
  }
];
