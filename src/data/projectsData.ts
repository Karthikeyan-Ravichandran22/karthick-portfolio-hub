
// Project data for the portfolio site
export interface Project {
  id: number;
  title: string;
  description: string;
  image: string;
  technologies: string[];
  link?: string;
  github?: string;
  readme?: string;
}

export const projects: Project[] = [
  {
    id: 1,
    title: "Neural Style Transfer",
    description: "An implementation of neural style transfer algorithm that combines the content of one image with the style of another using deep neural networks.",
    image: "/project-neural-style.jpg",
    technologies: ["Python", "TensorFlow", "Computer Vision", "Deep Learning"],
    github: "https://github.com/yourusername/neural-style-transfer",
    readme: `# Neural Style Transfer

## Overview
This project implements the Neural Style Transfer algorithm using TensorFlow, allowing users to generate artistic images by combining the content of one image with the style of another.

## Features
- Content and style image upload
- Adjustable style weight and content weight parameters
- Real-time style transfer visualization
- Multiple pre-trained models for different artistic styles
- Batch processing for multiple images

## Technical Implementation
The implementation uses a pre-trained VGG19 network to extract content and style features from images. The algorithm then performs gradient descent to generate a new image that minimizes both content loss and style loss.

\`\`\`python
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import vgg19

def neural_style_transfer(content_image, style_image, num_iterations=1000):
    # Load VGG19 model
    model = vgg19.VGG19(weights='imagenet', include_top=False)
    
    # Extract layers for content and style representation
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    # Initialize generated image with content image
    generated_image = tf.Variable(content_image)
    
    # Optimization loop
    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            # Calculate content loss
            content_loss = compute_content_loss(model, generated_image, content_image, content_layers)
            
            # Calculate style loss
            style_loss = compute_style_loss(model, generated_image, style_image, style_layers)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Compute gradients and update image
        gradients = tape.gradient(total_loss, generated_image)
        optimizer.apply_gradients([(gradients, generated_image)])
        
        # Clip pixel values to valid range
        generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))
    
    return generated_image
\`\`\`

## Setup and Usage
1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Run the application: \`python style_transfer.py --content path/to/content.jpg --style path/to/style.jpg\`

## Results
The repository includes example images showing the results of applying different artistic styles to photographs.

## Future Improvements
- Implement real-time video style transfer
- Add support for custom layer weighting
- Create a web-based interface for easier use
- Optimize for faster processing on CPUs`
  },
  {
    id: 2,
    title: "Generative AI Chatbot",
    description: "A sophisticated chatbot built using large language models that can understand and generate human-like text responses for various domains.",
    image: "/project-chatbot.jpg",
    technologies: ["Python", "PyTorch", "NLP", "Transformers", "FastAPI"],
    github: "https://github.com/yourusername/generative-ai-chatbot",
    link: "https://ai-chatbot-demo.example.com",
    readme: `# Generative AI Chatbot

## Overview
This project implements a state-of-the-art generative AI chatbot using the latest transformer-based language models. The chatbot can understand context, maintain conversation history, and generate human-like responses across various domains.

## Features
- Context-aware conversations with memory of previous exchanges
- Domain-specific knowledge in technology, science, and general topics
- API endpoints for easy integration with websites and applications
- Customizable personality and response style
- Rate limiting and content filtering for safe deployment

## Technical Implementation
The chatbot is built using a fine-tuned transformer model based on the GPT architecture, with additional components for context management and response generation.

\`\`\`python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GenerativeAIChatbot:
    def __init__(self, model_name="gpt2-medium"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.conversation_history = []
        
    def add_message(self, message, is_user=True):
        self.conversation_history.append({"content": message, "is_user": is_user})
        
    def generate_response(self, max_length=100):
        # Format conversation history
        prompt = self._format_conversation()
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        output = self.model.generate(
            input_ids,
            max_length=max_length + input_ids.shape[1],
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Decode the response
        response = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Add response to history
        self.add_message(response, is_user=False)
        
        return response
        
    def _format_conversation(self):
        formatted = ""
        for message in self.conversation_history[-5:]:  # Keep last 5 messages for context
            prefix = "User: " if message["is_user"] else "AI: "
            formatted += prefix + message["content"] + "\\n"
        formatted += "AI: "
        return formatted
\`\`\`

## API Implementation
The chatbot is exposed through a FastAPI interface:

\`\`\`python
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()
chatbot = GenerativeAIChatbot()

class Message(BaseModel):
    content: str
    is_user: bool = True

class Conversation(BaseModel):
    messages: List[Message]

@app.post("/chat")
async def chat(message: Message):
    chatbot.add_message(message.content, is_user=True)
    response = chatbot.generate_response()
    return {"response": response}
\`\`\`

## Setup and Usage
1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Run the API server: \`uvicorn app:app --reload\`
4. Interact with the chatbot through the API or included web interface

## Deployment
The repository includes Dockerfile and deployment scripts for easy hosting on cloud platforms like AWS, GCP, or Azure.

## Future Improvements
- Implement voice input/output capabilities
- Add multi-language support
- Enhance domain-specific knowledge with retrieval augmentation
- Optimize for mobile deployment`
  },
  {
    id: 3,
    title: "Computer Vision Object Detector",
    description: "Real-time object detection system that can identify and track multiple objects in images and video streams with high accuracy.",
    image: "/project-object-detection.jpg",
    technologies: ["Python", "OpenCV", "TensorFlow", "YOLO", "Computer Vision"],
    github: "https://github.com/yourusername/object-detector",
    readme: `# Computer Vision Object Detector

## Overview
This project implements a real-time object detection system capable of identifying and tracking multiple objects in images and video streams. It utilizes state-of-the-art deep learning models for high accuracy detection across various lighting conditions and scenarios.

## Features
- Real-time object detection in video streams
- Support for multiple object classes (80+ categories)
- Bounding box visualization with confidence scores
- Person tracking across video frames
- Performance metrics and optimization options

## Technical Implementation
The system is built on the YOLO (You Only Look Once) architecture with TensorFlow backend, optimized for speed and accuracy:

\`\`\`python
import cv2
import numpy as np
import tensorflow as tf
from models.yolo import YOLO

class ObjectDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = self._load_class_names('data/coco.names')
        
    def _load_class_names(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def detect_image(self, image):
        # Preprocess image
        input_size = self.model.input_shape[1:3]
        image_data = self._preprocess_image(image, input_size)
        
        # Run detection
        boxes, scores, classes = self.model.predict(image_data)
        
        # Filter by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': [self.class_names[int(c)] for c in classes]
        }
    
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup output video writer if needed
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect objects
            results = self.detect_image(frame)
            
            # Draw bounding boxes
            self._draw_results(frame, results)
            
            # Write frame if output specified
            if output_path:
                out.write(frame)
            
            # Display frame
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def _preprocess_image(self, image, input_size):
        # Resize and normalize image
        image_resized = cv2.resize(image, input_size)
        image_normalized = image_resized / 255.0
        image_expanded = np.expand_dims(image_normalized, axis=0)
        return image_expanded
    
    def _draw_results(self, image, results):
        boxes, scores, classes = results['boxes'], results['scores'], results['classes']
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{classes[i]}: {scores[i]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
\`\`\`

## Setup and Usage
1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Download pre-trained weights: \`python download_weights.py\`
4. Run image detection: \`python detect.py --image path/to/image.jpg\`
5. Run video detection: \`python detect.py --video path/to/video.mp4 --output output.mp4\`

## Performance
The system achieves:
- 45 FPS on GPU (NVIDIA RTX 3080)
- 15 FPS on CPU (Intel i7)
- 73.5% mAP (mean Average Precision) on COCO dataset

## Applications
- Security and surveillance systems
- Autonomous vehicles
- Retail analytics
- Industrial quality control

## Future Improvements
- Implement instance segmentation
- Add multi-camera tracking capability
- Optimize for edge devices (Jetson Nano, Raspberry Pi)
- Integrate with streaming platforms for live detection`
  },
  {
    id: 4,
    title: "Recommendation Engine",
    description: "Advanced recommendation system using collaborative filtering and content-based approaches to provide personalized suggestions for users.",
    image: "/project-recommendations.jpg",
    technologies: ["Python", "TensorFlow", "Scikit-learn", "Pandas", "Flask"],
    github: "https://github.com/yourusername/recommendation-engine",
    readme: `# Recommendation Engine

## Overview
This project implements a hybrid recommendation system that combines collaborative filtering and content-based approaches to provide highly personalized recommendations for users. The system is designed to be scalable and adaptable to different domains such as products, movies, or articles.

## Features
- User-based and item-based collaborative filtering
- Content-based recommendations using item features
- Hybrid recommendation combining multiple approaches
- Cold start handling for new users and items
- A/B testing framework for recommendation strategies
- REST API for easy integration

## Technical Implementation
The recommendation engine uses matrix factorization with implicit feedback for collaborative filtering and a neural network for content-based recommendations:

\`\`\`python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

class HybridRecommender:
    def __init__(self, user_item_matrix, item_features):
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        self.user_factors = None
        self.item_factors = None
        self.content_model = None
        
    def train_collaborative_filtering(self, n_factors=50, n_epochs=20, reg=0.1):
        # Initialize factors randomly
        n_users, n_items = self.user_item_matrix.shape
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        
        # Alternating Least Squares (ALS) implementation
        for epoch in range(n_epochs):
            # Fix item factors and solve for user factors
            for u in range(n_users):
                # Get items rated by user
                item_indices = self.user_item_matrix[u].indices
                if len(item_indices) == 0:
                    continue
                    
                # Get ratings and corresponding item factors
                ratings = self.user_item_matrix[u].data
                item_factors_u = self.item_factors[item_indices]
                
                # Solve least squares problem
                A = item_factors_u.T @ item_factors_u + reg * np.eye(n_factors)
                b = item_factors_u.T @ ratings
                self.user_factors[u] = np.linalg.solve(A, b)
            
            # Fix user factors and solve for item factors
            for i in range(n_items):
                # Get users who rated this item
                user_indices = self.user_item_matrix[:, i].indices
                if len(user_indices) == 0:
                    continue
                    
                # Get ratings and corresponding user factors
                ratings = self.user_item_matrix[:, i].data
                user_factors_i = self.user_factors[user_indices]
                
                # Solve least squares problem
                A = user_factors_i.T @ user_factors_i + reg * np.eye(n_factors)
                b = user_factors_i.T @ ratings
                self.item_factors[i] = np.linalg.solve(A, b)
                
            # Compute RMSE for this epoch
            predicted = self.user_factors @ self.item_factors.T
            mask = self.user_item_matrix.toarray() > 0
            rmse = np.sqrt(np.mean(np.square(predicted[mask] - self.user_item_matrix.toarray()[mask])))
            print(f"Epoch {epoch+1}/{n_epochs}, RMSE: {rmse:.4f}")
    
    def train_content_based(self, hidden_layers=[64, 32]):
        # Create neural network for content-based recommendations
        n_features = self.item_features.shape[1]
        n_items = self.item_features.shape[0]
        
        self.content_model = Sequential()
        self.content_model.add(Dense(hidden_layers[0], activation='relu', input_shape=(n_features,)))
        for units in hidden_layers[1:]:
            self.content_model.add(Dense(units, activation='relu'))
        self.content_model.add(Dense(50, activation='linear'))  # Match collaborative filtering dimensions
        
        # Compile and train model
        self.content_model.compile(optimizer='adam', loss='mse')
        
        # Use item factors from collaborative filtering as target
        self.content_model.fit(
            self.item_features, 
            self.item_factors, 
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
    
    def get_recommendations(self, user_id, n=10, alpha=0.7):
        if user_id >= self.user_factors.shape[0]:
            # Cold start - new user
            return self._get_popular_items(n)
        
        # Collaborative filtering score
        cf_scores = self.user_factors[user_id] @ self.item_factors.T
        
        # Content-based score using user's history
        user_items = self.user_item_matrix[user_id].indices
        if len(user_items) > 0:
            # Get features of items the user has interacted with
            user_profile = np.mean(self.item_features[user_items], axis=0).reshape(1, -1)
            
            # Predict item factors from content model
            content_item_factors = self.content_model.predict(self.item_features)
            
            # Compute similarity
            cb_scores = user_profile @ content_item_factors.T
            cb_scores = cb_scores.flatten()
        else:
            # No history, use average
            cb_scores = np.zeros_like(cf_scores)
        
        # Combine scores
        hybrid_scores = alpha * cf_scores + (1 - alpha) * cb_scores
        
        # Get top N items
        already_seen = set(user_items)
        top_items = np.argsort(-hybrid_scores)
        recommendations = [item for item in top_items if item not in already_seen][:n]
        
        return recommendations
    
    def _get_popular_items(self, n=10):
        # For cold start: return most popular items
        item_popularity = np.asarray(self.user_item_matrix.sum(axis=0)).flatten()
        return np.argsort(-item_popularity)[:n]
\`\`\`

## API Implementation
The recommendation engine is exposed through a Flask API:

\`\`\`python
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)
recommender = None

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id', type=int)
    count = request.args.get('count', default=10, type=int)
    
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400
        
    try:
        recommended_items = recommender.get_recommendations(user_id, n=count)
        return jsonify({
            "user_id": user_id,
            "recommendations": recommended_items
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations/explain', methods=['GET'])
def explain_recommendations():
    user_id = request.args.get('user_id', type=int)
    item_id = request.args.get('item_id', type=int)
    
    if not user_id or not item_id:
        return jsonify({"error": "Missing parameters"}), 400
        
    try:
        explanation = recommender.explain_recommendation(user_id, item_id)
        return jsonify({
            "user_id": user_id,
            "item_id": item_id,
            "explanation": explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load data and initialize recommender
    ratings = pd.read_csv('data/ratings.csv')
    item_features = pd.read_csv('data/item_features.csv')
    
    # Create user-item matrix
    user_item = pd.pivot_table(
        ratings, 
        values='rating', 
        index='user_id', 
        columns='item_id'
    ).fillna(0)
    
    # Convert to sparse matrix
    from scipy.sparse import csr_matrix
    user_item_matrix = csr_matrix(user_item.values)
    
    # Initialize and train recommender
    recommender = HybridRecommender(user_item_matrix, item_features.values)
    recommender.train_collaborative_filtering()
    recommender.train_content_based()
    
    # Start API
    app.run(debug=True, host='0.0.0.0', port=5000)
\`\`\`

## Evaluation
The system is evaluated using:
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (NDCG)
- Diversity and coverage metrics
- A/B testing with real users

## Setup and Usage
1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Prepare your data in CSV format (ratings and item features)
4. Train the models: \`python train.py\`
5. Start the API server: \`python api.py\`

## Use Cases
- E-commerce product recommendations
- Movie and TV show suggestions
- News article personalization
- Music recommendation systems

## Future Improvements
- Implement contextual recommendations (time, location)
- Add sequence-aware recommendations for session-based data
- Scale to larger datasets using distributed computing
- Improve cold start handling with active learning approaches`
  },
  {
    id: 5,
    title: "Time Series Forecasting",
    description: "Advanced time series analysis and forecasting system for predicting future values in financial data, sales, and other sequential datasets.",
    image: "/project-time-series.jpg",
    technologies: ["Python", "Prophet", "LSTM", "Pandas", "Statsmodels", "Plotly"],
    github: "https://github.com/yourusername/time-series-forecasting",
    readme: `# Time Series Forecasting

## Overview
This project implements various time series forecasting models to predict future values in sequential data. It supports multiple algorithms and provides tools for analysis, modeling, and visualization of time series data. The system is designed for applications in finance, sales forecasting, demand planning, and other domains requiring future predictions.

## Features
- Multiple forecasting models (ARIMA, Prophet, LSTM, etc.)
- Automatic seasonality detection
- Anomaly detection in time series data
- Confidence intervals for predictions
- Interactive visualizations
- Model performance comparison
- Cross-validation framework
- Hyperparameter optimization

## Technical Implementation
The project implements several forecasting approaches including traditional statistical methods and deep learning:

\`\`\`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TimeSeriesForecaster:
    def __init__(self, data, date_column, target_column):
        self.raw_data = data
        self.date_column = date_column
        self.target_column = target_column
        self.data = self._prepare_data()
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _prepare_data(self):
        df = self.raw_data.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df = df.sort_values(by=self.date_column)
        df = df.set_index(self.date_column)
        return df
    
    def split_data(self, test_size=0.2):
        n = len(self.data)
        train_size = int(n * (1 - test_size))
        self.train_data = self.data.iloc[:train_size]
        self.test_data = self.data.iloc[train_size:]
        return self.train_data, self.test_data
    
    def train_arima(self, order=(1, 1, 1)):
        # Train ARIMA model
        model = ARIMA(self.train_data[self.target_column], order=order)
        self.arima_model = model.fit()
        return self.arima_model
    
    def train_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        # Train Seasonal ARIMA model
        model = SARIMAX(
            self.train_data[self.target_column],
            order=order,
            seasonal_order=seasonal_order
        )
        self.sarima_model = model.fit(disp=False)
        return self.sarima_model
    
    def train_prophet(self):
        # Prepare data for Prophet
        df = self.train_data.reset_index()
        df = df.rename(columns={self.date_column: 'ds', self.target_column: 'y'})
        
        # Create and train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df)
        self.prophet_model = model
        return model
    
    def train_lstm(self, look_back=60, epochs=50, batch_size=32, neurons=50):
        # Scale data
        data = self.train_data[self.target_column].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=neurons, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=neurons))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Compile and train
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        
        self.lstm_model = model
        self.look_back = look_back
        return model
    
    def forecast_arima(self, steps):
        if not hasattr(self, 'arima_model'):
            raise ValueError("ARIMA model has not been trained yet")
        
        forecast = self.arima_model.forecast(steps=steps)
        return forecast
    
    def forecast_sarima(self, steps):
        if not hasattr(self, 'sarima_model'):
            raise ValueError("SARIMA model has not been trained yet")
        
        forecast = self.sarima_model.forecast(steps=steps)
        return forecast
    
    def forecast_prophet(self, periods):
        if not hasattr(self, 'prophet_model'):
            raise ValueError("Prophet model has not been trained yet")
        
        future = self.prophet_model.make_future_dataframe(periods=periods, freq='D')
        forecast = self.prophet_model.predict(future)
        return forecast
    
    def forecast_lstm(self, steps):
        if not hasattr(self, 'lstm_model'):
            raise ValueError("LSTM model has not been trained yet")
        
        # Get last sequence from training data
        last_sequence = self.train_data[self.target_column].values[-self.look_back:].reshape(-1, 1)
        last_sequence = self.scaler.transform(last_sequence)
        
        # Generate predictions
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            current_batch = current_sequence.reshape(1, self.look_back, 1)
            
            # Get prediction (next time step)
            next_pred = self.lstm_model.predict(current_batch)[0, 0]
            
            # Add to predictions
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
        # Inverse transform to get original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def plot_results(self, forecast_data, model_name):
        plt.figure(figsize=(12, 6))
        
        # Plot training data
        plt.plot(self.train_data.index, self.train_data[self.target_column], label='Training Data')
        
        # Plot test data if available
        if self.test_data is not None:
            plt.plot(self.test_data.index, self.test_data[self.target_column], label='Test Data')
        
        # Plot forecast
        if model_name == 'prophet':
            forecast_dates = pd.date_range(
                start=self.train_data.index[-1], 
                periods=len(forecast_data['yhat']), 
                freq='D'
            )
            plt.plot(forecast_dates, forecast_data['yhat'], label=f'{model_name} Forecast')
            plt.fill_between(
                forecast_dates,
                forecast_data['yhat_lower'],
                forecast_data['yhat_upper'],
                color='gray',
                alpha=0.2,
                label='95% Confidence Interval'
            )
        else:
            forecast_dates = pd.date_range(
                start=self.train_data.index[-1], 
                periods=len(forecast_data) + 1, 
                freq='D'
            )[1:]  # Skip first date as it's the last training date
            plt.plot(forecast_dates, forecast_data, label=f'{model_name} Forecast')
        
        plt.title(f'Time Series Forecast with {model_name}')
        plt.xlabel('Date')
        plt.ylabel(self.target_column)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def evaluate_model(self, predictions, actual):
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
\`\`\`

## Usage Example
Here's how to use the forecaster with sample data:

\`\`\`python
# Load data
data = pd.read_csv('data/sales_data.csv')

# Initialize forecaster
forecaster = TimeSeriesForecaster(data, date_column='date', target_column='sales')

# Split data
train, test = forecaster.split_data(test_size=0.2)

# Train different models
forecaster.train_arima(order=(2, 1, 2))
forecaster.train_sarima(order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
forecaster.train_prophet()
forecaster.train_lstm(look_back=30, epochs=100)

# Make forecasts
arima_forecast = forecaster.forecast_arima(steps=len(test))
sarima_forecast = forecaster.forecast_sarima(steps=len(test))
prophet_forecast = forecaster.forecast_prophet(periods=len(test))
lstm_forecast = forecaster.forecast_lstm(steps=len(test))

# Evaluate models
print("ARIMA Evaluation:")
forecaster.evaluate_model(arima_forecast, test['sales'].values)

print("SARIMA Evaluation:")
forecaster.evaluate_model(sarima_forecast, test['sales'].values)

print("Prophet Evaluation:")
prophet_test_forecast = prophet_forecast.iloc[-len(test):]['yhat'].values
forecaster.evaluate_model(prophet_test_forecast, test['sales'].values)

print("LSTM Evaluation:")
forecaster.evaluate_model(lstm_forecast, test['sales'].values)

# Plot results
forecaster.plot_results(arima_forecast, 'ARIMA')
forecaster.plot_results(sarima_forecast, 'SARIMA')
forecaster.plot_results(prophet_forecast, 'Prophet')
forecaster.plot_results(lstm_forecast, 'LSTM')
\`\`\`

## Setup and Usage
1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Import your time series data in CSV format
4. Run the jupyter notebooks for examples and visualization
5. Use the forecaster class to implement your own forecasting application

## Applications
- Financial market prediction
- Sales and demand forecasting
- Energy consumption prediction
- Website traffic forecasting
- Inventory management
- Resource allocation planning

## Future Improvements
- Implement ensemble methods combining multiple forecasts
- Add automatic hyperparameter tuning
- Support multivariate time series forecasting
- Implement transformers for time series
- Add explainability features for predictions
- Create a web interface for interactive forecasting`
  },
  {
    id: 6,
    title: "NLP Sentiment Analyzer",
    description: "Natural language processing tool that analyzes and classifies text sentiment with high accuracy across multiple domains and languages.",
    image: "/project-sentiment.jpg",
    technologies: ["Python", "SpaCy", "BERT", "Hugging Face", "Flask"],
    github: "https://github.com/yourusername/sentiment-analyzer",
    link: "https://sentiment-analyzer-demo.example.com",
    readme: `# NLP Sentiment Analyzer

## Overview
This project implements a sophisticated sentiment analysis system that can detect and classify the emotional tone in text data. The analyzer works across multiple domains and supports several languages, providing both classification results and confidence scores.

## Features
- Sentiment classification (positive, negative, neutral)
- Emotion detection (joy, anger, sadness, fear, surprise)
- Aspect-based sentiment analysis for specific topics
- Multi-language support (English, Spanish, French, German)
- Pre-trained models for specific domains (social media, reviews, news)
- REST API for integration with other applications
- Batch processing for large datasets

## Technical Implementation
The system uses transformer-based models from Hugging Face's library with fine-tuning on domain-specific datasets:

\`\`\`python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def analyze(self, text):
        # Prepare input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get scores
        scores = softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        # Get label mapping from model config
        labels = self.model.config.id2label
        
        # Format results
        results = {
            "text": text,
            "sentiment": labels[np.argmax(scores)],
            "confidence": float(np.max(scores)),
            "scores": {labels[i]: float(score) for i, score in enumerate(scores)}
        }
        
        return results
    
    def batch_analyze(self, texts, batch_size=16):
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Prepare batch input
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get scores
            scores = softmax(outputs.logits, dim=1).cpu().numpy()
            
            # Get label mapping from model config
            labels = self.model.config.id2label
            
            # Format batch results
            for j, text in enumerate(batch):
                text_scores = scores[j]
                result = {
                    "text": text,
                    "sentiment": labels[np.argmax(text_scores)],
                    "confidence": float(np.max(text_scores)),
                    "scores": {labels[i]: float(score) for i, score in enumerate(text_scores)}
                }
                results.append(result)
        
        return results
    
class AspectBasedSentimentAnalyzer:
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def analyze(self, text, aspects):
        results = []
        
        for aspect in aspects:
            # Format input for aspect-based sentiment analysis
            input_text = f"{text} [SEP] {aspect}"
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get scores
            scores = softmax(outputs.logits, dim=1).cpu().numpy()[0]
            
            # Get label mapping from model config
            labels = self.model.config.id2label
            
            # Format result
            result = {
                "text": text,
                "aspect": aspect,
                "sentiment": labels[np.argmax(scores)],
                "confidence": float(np.max(scores)),
                "scores": {labels[i]: float(score) for i, score in enumerate(scores)}
            }
            
            results.append(result)
        
        return results

class EmotionDetector:
    def __init__(self, model_name="joeddav/distilbert-base-uncased-go-emotions-student"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def detect_emotions(self, text, threshold=0.3):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get scores using sigmoid for multi-label classification
        scores = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Get emotions above threshold
        emotions = []
        for i, score in enumerate(scores):
            if score >= threshold:
                emotion = {
                    "label": self.model.config.id2label[i],
                    "score": float(score)
                }
                emotions.append(emotion)
        
        # Sort by score
        emotions = sorted(emotions, key=lambda x: x["score"], reverse=True)
        
        return {
            "text": text,
            "emotions": emotions
        }
\`\`\`

## API Implementation
The sentiment analysis is exposed through a Flask API:

\`\`\`python
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

# Initialize models
sentiment_analyzer = SentimentAnalyzer()
aspect_analyzer = AspectBasedSentimentAnalyzer()
emotion_detector = EmotionDetector()

# Cache for models (to avoid reloading)
models_cache = {
    "general": sentiment_analyzer,
    "aspect": aspect_analyzer,
    "emotion": emotion_detector
}

# Add language-specific models
for lang in ["es", "fr", "de"]:
    models_cache[f"general_{lang}"] = SentimentAnalyzer(f"nlptown/bert-base-multilingual-uncased-sentiment")

# Add domain-specific models
models_cache["social"] = SentimentAnalyzer("cardiffnlp/twitter-roberta-base-sentiment")
models_cache["reviews"] = SentimentAnalyzer("nlptown/bert-base-multilingual-uncased-sentiment")

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
        
    text = data['text']
    model_type = data.get('model', 'general')
    
    if model_type not in models_cache:
        return jsonify({"error": f"Model '{model_type}' not available"}), 400
    
    analyzer = models_cache[model_type]
    result = analyzer.analyze(text)
    
    return jsonify(result)

@app.route('/api/aspect-sentiment', methods=['POST'])
def analyze_aspect_sentiment():
    data = request.json
    
    if not data or 'text' not in data or 'aspects' not in data:
        return jsonify({"error": "Missing 'text' or 'aspects' field"}), 400
        
    text = data['text']
    aspects = data['aspects']
    
    result = aspect_analyzer.analyze(text, aspects)
    
    return jsonify(result)

@app.route('/api/emotions', methods=['POST'])
def detect_emotions():
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
        
    text = data['text']
    threshold = data.get('threshold', 0.3)
    
    result = emotion_detector.detect_emotions(text, threshold)
    
    return jsonify(result)

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    data = request.json
    
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
        
    texts = data['texts']
    model_type = data.get('model', 'general')
    
    if model_type not in models_cache:
        return jsonify({"error": f"Model '{model_type}' not available"}), 400
    
    analyzer = models_cache[model_type]
    results = analyzer.batch_analyze(texts)
    
    return jsonify({"results": results})

if __name__ == '__main__':
    # Preload models in background
    def preload_models():
        for name, model in models_cache.items():
            print(f"Preloading model: {name}")
            # Force model to load by making a prediction
            if hasattr(model, 'analyze'):
                model.analyze("This is a test.")
            elif hasattr(model, 'detect_emotions'):
                model.detect_emotions("This is a test.")
    
    threading.Thread(target=preload_models).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
\`\`\`

## Setup and Usage
1. Clone the repository
2. Install dependencies: \`pip install -r requirements.txt\`
3. Download pre-trained models (will be done automatically on first run)
4. Run the API server: \`python app.py\`
5. Use the API to analyze text:
   - Send POST request to \`/api/sentiment\` with JSON: \`{"text": "I love this product!"}\`
   - Send POST request to \`/api/aspect-sentiment\` with JSON: \`{"text": "The screen is bright but the battery life is terrible", "aspects": ["screen", "battery"]}\`
   - Send POST request to \`/api/emotions\` with JSON: \`{"text": "I'm so happy to see you!"}\`

## Example Use Cases
- Social media monitoring
- Customer feedback analysis
- Brand reputation management
- Market research
- Customer support automation
- Content moderation

## Future Improvements
- Add support for more languages
- Improve fine-tuning on specific domains
- Add explainable AI features
- Implement real-time streaming analysis
- Add more emotion categories
- Integrate with popular platforms via plugins`
  }
];

