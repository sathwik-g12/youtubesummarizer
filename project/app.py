from flask import Flask, render_template, request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from heapq import nlargest
import nltk
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
from pytube import YouTube  

app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Function to get the video title using BeautifulSoup
def get_video_title(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find('title').text.strip()
    return title

# Function to get YouTube video cover page
def get_youtube_thumbnail_url(video_url):
    try:
        yt = YouTube(video_url)
        thumbnail_url = yt.thumbnail_url
        return thumbnail_url
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Step 1: Data Collection through YouTube Transcript API
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([d['text'] for d in transcript_list])
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

# Step 2: Preprocessing for Noise Reduction
def clean_transcript(transcript):
    cleaned_transcript = ' '.join([word.lower() for word in word_tokenize(transcript) if word.isalpha()])
    return cleaned_transcript

# Step 3: NLP Analysis for Context Understanding
def analyze_content(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

    # Frequency distribution
    freq_dist = FreqDist(filtered_tokens)

    # Identify important keywords
    important_keywords = [word for word, freq in freq_dist.items() if freq > 1]

    return important_keywords

# Step 4: Text Summarization
def generate_summary(transcript):
    summarizer = pipeline('summarization')
    summary = ''
    for i in range(0, (len(transcript)//1000)+1):
        summary_text = summarizer(transcript[i*1000:(i+1)*1000], max_length=50, min_length=10, length_penalty=2.0)[0]['summary_text']
        summary = summary + summary_text + ' '
    return summary

# Function to translate text using chunks
def translate_text_chunks(text, target_language):
    translator = Translator()

    # Split the text into chunks of 5000 characters
    chunk_size = 5000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Translate each chunk and concatenate the results
    translated_text = ""
    for chunk in chunks:
        translation = translator.translate(chunk, dest=target_language)
        translated_text += translation.text

    return translated_text

@app.route('/', methods=['GET', 'POST'])
def index():
    # Define a list of supported languages
    supported_languages = ["es", "fr", "de", "te", "hi", "mr", "ta"]

    if request.method == 'POST':
        video_id = request.form['video_id']
        target_language = request.form['target_language']

        video_title = get_video_title(video_id)
        transcript = get_transcript(video_id)

        if transcript is not None:
            cleaned_transcript = clean_transcript(transcript)
            keywords = analyze_content(cleaned_transcript)

            # Use NLTK-based summarization
            summary = generate_summary(cleaned_transcript)

            # Translate the summary to the selected target language using chunks
            translated_summary = translate_text_chunks(summary, target_language)

            # Get YouTube video thumbnail URL
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            thumbnail_url = get_youtube_thumbnail_url(video_url)

            has_output = any([keywords, summary, translated_summary])
            if not has_output:
                error_message = "No output received. Please check the video ID or try again later."
                return render_template('index.html', error_message=error_message, supported_languages=supported_languages)

            return render_template('results.html', keywords=keywords, original_summary=summary, translated_summary=translated_summary, video_title=video_title, target_language=target_language, thumbnail_url=thumbnail_url)

        error_message = "An error occurred while fetching the transcript. Please check the video ID."
        return render_template('index.html', error_message=error_message, supported_languages=supported_languages)

    return render_template('index.html', supported_languages=supported_languages)

if __name__ == '__main__':
    app.run(debug=True)
