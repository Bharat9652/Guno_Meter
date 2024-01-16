from flask import Flask, render_template, request, flash
import joblib
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'  # Used for flashing messages

# Load the model and TF-IDF vectorizer
model = joblib.load('guna_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Sample pool of 30 questions for testing
all_questions = [
    "How do you handle stress and challenges in your life?",
    "What motivates you to achieve success?",
    "Describe your approach to learning new things.",
    "How do you respond to criticism?",
    "In a group setting, do you prefer taking the lead or observing?",
    "What brings you a sense of peace and joy?",
    "How do you handle unexpected changes in plans?",
    "Describe your reaction to success.",
    "How do you spend your leisure time?",
    "What is your attitude towards routine and stability?",
    "How do you deal with conflicts in relationships?",
    "What qualities do you value in others?",
    "How do you approach decision-making?",
    "Describe your communication style.",
    "What is your perspective on personal growth?",
    "How do you express gratitude?",
    "Describe your work ethic.",
    "What role does spirituality play in your life?",
    "How do you handle setbacks?",
    "What is your relationship with technology?",
    "Describe your ideal day.",
    "How do you approach goal-setting?",
    "What do you prioritize in your life?",
    "How do you recharge and relax?",
    "What role does humor play in your life?",
    "Describe your approach to time management.",
    "How do you handle criticism from others?",
    "What qualities do you look for in a leader?",
    "How do you contribute to a team environment?"
]

# Randomly select 10 questions for each session
selected_questions = random.sample(all_questions, 20)

@app.route('/')
def home():
    return render_template('index.html', questions=selected_questions)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_responses = [request.form[f'response_{i+1}'] for i in range(len(selected_questions))]
        
        # Check for empty answers
        if any(not response.strip() for response in user_responses):
            flash('Please answer all questions before submitting.', 'warning')
            return render_template('index.html', questions=selected_questions)

        user_tfidf = vectorizer.transform(user_responses)
        prediction = model.predict(user_tfidf)
        return render_template('result.html', prediction=prediction)

#if __name__ == '__main__':
    #app.run(debug=True)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)