from django.shortcuts import render,redirect,get_object_or_404
from django.core.mail import send_mail
import urllib.request
import urllib.parse
import random 
import time
from django.utils.datastructures import MultiValueDictKeyError
from django.contrib import messages
from django.conf import settings
from app.models import *
import os
from django.shortcuts import render, redirect
from .forms import VideoForm
from .models import Video, TranscriptSegment
from django.contrib import messages
from .utils import (
    download_youtube_video,
    extract_audio,
    transcribe_audio_with_timestamps,
    download_google_drive_file,
    download_dropbox_file,
    reencode_video,
)
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from django.http import JsonResponse
import openai 
import json
import requests
from django.http import JsonResponse
from django.middleware.csrf import get_token

from googletrans import Translator


def generate_otp(length=4):
    otp = "".join(random.choices("0123456789", k=length))
    return otp



EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')



def index(request):
    feedbacks = Feedback.objects.all()
    print("DEBUG: Number of feedbacks:", feedbacks.count())
    for fb in feedbacks:
        # Print out details for each feedback record.
        print("DEBUG: Feedback:", fb.user_name, fb.user.full_name, fb.additional_comments)
    return render(request, "index.html", {'feedbacks': feedbacks})

def about(request):
    return render(request,"about.html")

def user_login(request):
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        try:
            user = User.objects.get(email=email)
            if user.password != password:
                messages.error(request, "Incorrect password.")
                return redirect("user_login")
            if 12 == 12:
                if user.otp_status == "Verified":
                    request.session["user_id_after_login"] = user.pk
                    user.save()
                    messages.success(request, "Login successful!")
                    return redirect("user_dashboard")
                else:
                    new_otp = generate_otp()
                    user.otp = new_otp
                    user.otp_status = "Not Verified"
                    user.save()
                    subject = "New OTP for Verification"
                    message = f"Your new OTP for verification is: {new_otp}"
                    from_email = settings.EMAIL_HOST_USER
                    recipient_list = [user.email]
                    send_mail(
                        subject, message, from_email, recipient_list, fail_silently=False
                    )
                    request.session["id_for_otp_verification_user"] = user.pk
                    return redirect("user_otp")
            else:
                messages.info(request, "Your Account is Not Accepted by Admin Yet")
                return redirect("user_login")
        except User.DoesNotExist:
            messages.error(request, "No User Found.")
            return redirect("user_login")
    
    return render(request, "user_login.html")


def user_register(request): 
    if request.method == "POST":
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password') 
        phone_number = request.POST.get('phone_number')
        age = request.POST.get('age')
        address = request.POST.get('address')
        photo = request.FILES.get('photo')
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('user_register') 
        user = User(
            full_name=full_name,
            email=email,
            password=password, 
            phone_number=phone_number,
            age=age,
            address=address,
            photo=photo
        )
        otp = generate_otp()
        user.otp = otp
        user.save()
        subject = "OTP Verification for Account Activation"
        message = f"Hello {full_name},\n\nYour OTP for account activation is: {otp}\n\nIf you did not request this OTP, please ignore this email."
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [email]
        request.session["id_for_otp_verification_user"] = user.pk
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)
        messages.success(request, "Otp is sent your mail and phonenumber !")
        return redirect("user_otp")
    return render(request,"user_register.html")







def user_otp(request):
    otp_user_id = request.session.get("id_for_otp_verification_user")
    if not otp_user_id:
        messages.error(request, "No OTP session found. Please try again.")
        return redirect("user_register")
    if request.method == "POST":
        entered_otp = "".join(
            [
                request.POST["first"],
                request.POST["second"],
                request.POST["third"],
                request.POST["fourth"],
            ]
        )
        try:
            user = User.objects.get(id=otp_user_id)
        except User.DoesNotExist:
            messages.error(request, "User not found. Please try again.")
            return redirect("user_register")
        if user.otp == entered_otp:
            user.otp_status = "Verified"
            user.save()
            messages.success(request, "OTP verification successful!")
            return redirect("user_login")
        else:
            messages.error(request, "Incorrect OTP. Please try again.")
            return redirect("user_otp")
    return render(request,"user_otp.html")




def user_dashboard(request):
    return render(request,"user_dashboard.html")





def contact(request):
    return render(request,"contact.html")




def logout(request):
    request.session.flush()
    messages.info(request, "Logout Successfully")
    return redirect("user_login")




def translate_text(request):
    if request.method == "POST":
        transcript = request.POST.get("transcript", "")
        target_language = request.POST.get("language", "en")  # Default to English

        translator = Translator()
        try:
            translation = translator.translate(transcript, dest=target_language)
            return JsonResponse({"translated_text": translation.text}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
# Set your Perplexity.ai or OpenAI API key here
# Set your Perplexity.ai API key here
PERPLEXITY_API_KEY = "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"

def call_perplexity_api(system_content, user_content):
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
    }
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()



@csrf_exempt
def ask_question(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')
        question = data.get('question', '')
        response = call_perplexity_api("Be precise and concise.", f"Answer the following question based on this text:\n\n{text}\n\nQuestion: {question}")
        result = response['choices'][0]['message']['content']
        return JsonResponse({'result': result.strip()})

@csrf_exempt
def summarize_text(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')
        response = call_perplexity_api("Be precise and concise.", f"Summarize the following text:\n\n{text}")
        result = response['choices'][0]['message']['content']
        return JsonResponse({'result': result.strip()})







def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            title = form.cleaned_data['title']
            description = form.cleaned_data['description']
            url = form.cleaned_data.get('url')
            video_file = request.FILES.get('video_file')

            # Handle URL input
            if url:
                if "youtube.com" in url or "youtu.be" in url:
                    source = "YouTube"
                    video_path = download_youtube_video(url, title)
                elif "drive.google.com" in url:
                    source = "Google Drive"
                    video_path = download_google_drive_file(url, title)
                elif "dropbox.com" in url:
                    source = "Dropbox"
                    video_path = download_dropbox_file(url, title)
                else:
                    messages.error(request, 'Unsupported URL.')
                    return redirect('upload_video')

            # Handle file upload input
            elif video_file:
                source = "Uploaded File"
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'videos')
                os.makedirs(upload_dir, exist_ok=True)
                # Sanitize title: replace non-alphanumeric characters (including spaces) with underscores.
                sanitized_title = "".join(c if c.isalnum() else "_" for c in title)
                # Append a timestamp to ensure uniqueness
                unique_filename = f"{sanitized_title}_{int(time.time())}.mp4"
                video_path = os.path.join(upload_dir, unique_filename)
                with open(video_path, 'wb') as f:
                    for chunk in video_file.chunks():
                        f.write(chunk)
            else:
                messages.error(request, 'Please provide either a URL or upload a file.')
                return redirect('upload_video')

            # Re-encode video to ensure proper metadata placement.
            # This produces a file with "_fixed.mp4" appended.
            video_path = reencode_video(video_path)

            # Extract audio and transcribe it with timestamps.
            audio_path = extract_audio(video_path)
            full_transcript, timestamped_segments = transcribe_audio_with_timestamps(audio_path)

            # Retrieve current logged-in user from the session.
            user_id = request.session.get('user_id_after_login')
            current_user = User.objects.filter(id=user_id).first() if user_id else None

            # Save video data with the current user.
            # Now we store the re-encoded video file name (which ends with "_fixed.mp4")
            video_obj = Video.objects.create(
                user=current_user,
                title=title,
                description=description,
                url=url if url else "",
                source=source,
                transcript=full_transcript,
                file_name=os.path.basename(video_path),  # This now should be "sanitized_title_timestamp_fixed.mp4"
            )

            # Save each transcript segment.
            for segment in timestamped_segments:
                TranscriptSegment.objects.create(
                    video=video_obj,
                    start_time=segment["start"],
                    end_time=segment["end"],
                    text=segment["text"],
                )

            return redirect('detection_history')
    else:
        form = VideoForm()

    return render(request, 'upload_video.html', {'form': form})



def video_detail(request, pk):
    """
    Displays details of a specific video and its transcript.
    """
    video = Video.objects.get(pk=pk)
    return render(request, 'video_detail.html', {'video': video})

def detection_history(request):
    user_id = request.session.get('user_id_after_login')
    videos = Video.objects.filter(user_id=user_id)
    return render(request, "detection_history.html", {'videos': videos})



import requests
from django.shortcuts import render, redirect
from django.contrib import messages
from app.models import Video





def translate_video(request, video_id):
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        messages.error(request, "Video not found.")
        return redirect("detection_history")

    translated_text = None
    selected_language = None

    PERPLEXITY_API_KEY = "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"

    def call_perplexity_api_translate(text, target_language):
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a translation assistant. Translate the following text to {target_language}. Be precise and concise."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            try:
                # Assuming the API returns a structure with a choices list.
                translation = data["choices"][0]["message"]["content"]
                return translation
            except Exception as e:
                return "Translation extraction error"
        else:
            return f"Translation Error: {response.status_code}"

    if request.method == "POST":
        selected_language = request.POST.get("target_language")
        if not selected_language:
            messages.error(request, "Please select a language.")
        else:
            translated_text = call_perplexity_api_translate(video.transcript, selected_language)
            if translated_text.startswith("Translation Error") or translated_text == "Translation extraction error":
                messages.error(request, translated_text)
            else:
                messages.success(request, f"Translation to {selected_language} successful.")

    return render(request, "video_translate.html", {
        "video": video,
        "translated_text": translated_text,
        "selected_language": selected_language,
    })





def ask_question(request, video_id):
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        messages.error(request, "Video not found.")
        return redirect("detection_history")

    answer = None
    question = None

    PERPLEXITY_API_KEY = "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"

    def call_perplexity_api_for_question(transcript, question):
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a factual question answering assistant. "
                        "Return only the exact answer with no extra commentary, explanation, or formatting. Do not include any "
                        "Do not use any external references, links, markdown formatting, or any special symbols. "
                        "Answer only based solely on the provided transcript. Do not use any external references or include any links. "
                        "Do not add any extra commentary, stylistic formatting, or additional remarks. "
                        "Provide a direct, concise answer to the question. "
                        "If the question is not directly related to the transcript, respond exactly with: 'I am not getting, can you repeat the question?'"
                    )
                },
                {
                    "role": "user",
                    "content": f"Transcript: {transcript}\n\nQuestion: {question}"
                }
            ],
            "max_tokens": 500,
            "temperature": 0.2,
            "top_p": 0.9,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            try:
                # Strip extra whitespace from the answer
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                return "Error extracting answer"
        else:
            return f"API Error: {response.status_code}"

    if request.method == "POST":
        question = request.POST.get("question")
        if not question:
            messages.error(request, "Please enter a question.")
        else:
            answer = call_perplexity_api_for_question(video.transcript, question)
            # If the answer is not exactly the expected phrase, you can check or log it.
            if answer.startswith("API Error") or answer == "Error extracting answer":
                messages.error(request, answer)
            else:
                messages.success(request, "Question answered successfully.")

    return render(request, "video_ask_question.html", {
        "video": video,
        "question": question,
        "answer": answer,
    })


import os, nltk
os.environ['NLTK_DATA'] = 'C:/nltk_data'
nltk.download('punkt_tab')

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')



from django.shortcuts import render, redirect
from django.contrib import messages
from app.models import Video
import nltk
import numpy as np

# (Make sure these are downloaded once, e.g., in your startup or a separate setup script)
nltk.download('punkt')
nltk.download('stopwords')

def summarize_video(request, video_id):
    try:
        video = Video.objects.get(pk=video_id)
    except Video.DoesNotExist:
        messages.error(request, "Video not found.")
        return redirect("detection_history")
    
    summary = None

    # All helper functions are defined inside this view.
    def preprocess_text(text):
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        sentences = sent_tokenize(text)
        word_tokens = [word_tokenize(sentence.lower()) for sentence in sentences]
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [
            [word for word in words if word not in stop_words and word.isalnum()]
            for words in word_tokens
        ]
        stemmer = PorterStemmer()
        stemmed_tokens = [
            [stemmer.stem(word) for word in words]
            for words in filtered_tokens
        ]
        return stemmed_tokens

    def calculate_similarity_matrix(sentences):
        vocab = set()
        for words in sentences:
            vocab.update(words)
        vocab = list(vocab)
        vocab_index = {word: i for i, word in enumerate(vocab)}
        similarity_matrix = np.zeros((len(vocab), len(vocab)))
        for words in sentences:
            for i in range(len(words)):
                for j in range(len(words)):
                    if i != j:
                        word1_index = vocab_index[words[i]]
                        word2_index = vocab_index[words[j]]
                        similarity_matrix[word1_index][word2_index] += 1
        return similarity_matrix

    def textrank_summarize(text, num_sentences=2):
        from nltk.tokenize import sent_tokenize
        preprocessed = preprocess_text(text)
        similarity_matrix = calculate_similarity_matrix(preprocessed)
        # Sum the similarities for each token (each word) to score the sentence
        sentence_scores = np.sum(similarity_matrix, axis=1)
        # Pair each sentence (from the original text) with its score
        original_sentences = sent_tokenize(text)
        scored_sentences = list(zip(sentence_scores, original_sentences))
        # Sort sentences by score (highest first)
        scored_sentences.sort(key=lambda pair: pair[0], reverse=True)
        # Pick the top num_sentences
        top_sentences = [sentence for _, sentence in scored_sentences[:num_sentences]]
        # Preserve original order from transcript
        top_sentences.sort(key=lambda s: text.index(s))
        return ' '.join(top_sentences)
    
    if request.method == "POST":
        summary = textrank_summarize(video.transcript, num_sentences=2)
        messages.success(request, "Summarization successful.")

    return render(request, "video_summary.html", {
        "video": video,
        "summary": summary,
    })



def play_video(request, pk):
    print(f"[DEBUG] Fetching video with pk: {pk}")
    video = get_object_or_404(Video, pk=pk)
    
    # Construct the full file path for server-side checking
    video_path = os.path.join(settings.MEDIA_ROOT, "videos", video.file_name)
    print(f"[DEBUG] Constructed video file path: {video_path}")

    if not os.path.exists(video_path):
        print("[ERROR] Video file does not exist on disk.")
    else:
        print("[DEBUG] Video file exists on disk.")

    # For extra debugging, also output the media URL that will be used
    video_url = os.path.join(settings.MEDIA_URL, "videos", video.file_name)
    print(f"[DEBUG] Video URL (for browser): {video_url}")

    return render(request, 'play_video.html', {'video': video})


def play_audio(request, pk):
    print(f"[DEBUG] Fetching video with pk: {pk}")
    video = get_object_or_404(Video, pk=pk)
    
    # Construct the full file path for the video and then derive the audio path
    video_path = os.path.join(settings.MEDIA_ROOT, "videos", video.file_name)
    audio_file_name = video.file_name.replace(".mp4", ".wav")
    audio_path = os.path.join(settings.MEDIA_ROOT, "videos", audio_file_name)
    
    print(f"[DEBUG] Constructed video file path: {video_path}")
    print(f"[DEBUG] Constructed audio file path: {audio_path}")
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"[ERROR] Audio file not found on disk: {audio_path}")
    else:
        print(f"[DEBUG] Audio file exists on disk: {audio_path}")

    # Also log the URL that will be used in the template
    audio_url = os.path.join(settings.MEDIA_URL, "videos", audio_file_name)
    print(f"[DEBUG] Audio URL (for browser): {audio_url}")
    
    return render(request, 'play_audio.html', {'video': video})


def detect_themes(request, pk):
    import requests  # Import within the view as requested
    from django.shortcuts import render, get_object_or_404
    from .models import Video

    # Use your API key
    PERPLEXITY_API_KEY = "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"

    video = get_object_or_404(Video, pk=pk)
    transcript = video.transcript or "No transcript available."
    detected_themes = None

    if request.method == "POST" and "detect_api" in request.POST:
        print("DEBUG: Starting API call to detect themes for video:", video.title)
        # Prepare payload with instructions to extract themes (e.g., educational, technological, etc.)
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in analyzing transcripts. Based on the transcript provided, list the main themes as concise bullet points (for example: Educational, Technology, Health, etc.)."
                },
                {
                    "role": "user",
                    "content": f"Extract the main themes from the following transcript:\n\n{transcript}"
                }
            ],
            "max_tokens": 123,
            "temperature": 0.2,
            "top_p": 0.9,
            "search_domain_filter": None,
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
            "response_format": None
        }
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        url = "https://api.perplexity.ai/chat/completions"
        print("DEBUG: Sending API request with payload:", payload)
        try:
            response = requests.post(url, json=payload, headers=headers)
            print("DEBUG: API response status code:", response.status_code)
            if response.status_code == 200:
                api_json = response.json()
                # Assume API returns a structure where the detected themes are in the content
                detected_text = api_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                print("DEBUG: Detected text from API:", detected_text)
                # Parse the text into individual themes.
                # For example, if the API returns bullet points separated by newlines or dashes.
                themes = [line.strip() for line in detected_text.replace('-', '\n').split('\n') if line.strip()]
                detected_themes = themes
                print("DEBUG: Parsed detected themes:", detected_themes)
            else:
                print("ERROR: API request failed with status code", response.status_code)
                detected_themes = [f"API error: {response.status_code}"]
        except Exception as e:
            print("ERROR: Exception during API call:", str(e))
            detected_themes = [f"Exception occurred: {str(e)}"]

    return render(
        request,
        "detect_themes.html",
        {
            "video": video,
            "transcript": transcript,
            "detected_themes": detected_themes,
        },
    )





























def user_profile(request):
    user_id = request.session.get('user_id_after_login')
    print(user_id)
    user = User.objects.get(pk=user_id)
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        try:
            profile = request.FILES['profile']
            user.photo = profile
        except MultiValueDictKeyError:
            pass  
        password = request.POST.get('password')
        location = request.POST.get('location')
        user.full_name = name
        user.email = email
        user.phone_number = phone
        user.password = password
        user.address = location
        user.save()
        messages.success(request, 'Updated successfully!')
        return redirect('user_profile')
    return render(request, "user_profile.html", {'i': user})








def user_feedbacks(request):
    if request.method == 'POST':
        user_id = request.session.get('user_id_after_login')
        user = User.objects.filter(pk=user_id).first()
        user_name = request.POST.get('user_name')
        user_email = request.POST.get('user_email')
        rating = request.POST.get('rating')
        additional_comments = request.POST.get('additional_comments')

        feedback_instance = Feedback.objects.create(
            user=user,
            user_name=user_name,
            user_email=user_email,
            rating=rating,
            additional_comments=additional_comments,
        )
        feedback_instance.save()
        messages.success(request, "Feedback Submitted Successfully!")
        return redirect('user_feedbacks')
    return render(request, "user_feedbacks.html")














import re
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.shortcuts import render, get_object_or_404
from .models import Video

# Ensure necessary downloads
nltk.download("stopwords")

# Load the saved model
loaded_model = tf.keras.models.load_model("ag_news_model.keras")

# Label mapping for predictions
class_labels = ["World", "Sports", "Business", "Sci/Tech"]
voc_size = 5000  # Example vocabulary size
sent_len = 50  # Max length of sentences

def clean_new_text(text):
    """Cleans the input text by removing special characters, stopwords, and applying stemming."""
    stem = PorterStemmer()
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.split()
    text = [word for word in text if word not in stopwords.words("english")]
    text = [stem.stem(word) for word in text]
    return " ".join(text)

def prepare_new_text(text):
    """Encodes and pads the text for model input."""
    onehot_repr = one_hot(text, voc_size)
    return pad_sequences([onehot_repr], padding="post", maxlen=sent_len)

def predict(text):
    """Predicts the theme of the given text using the trained model."""
    cleaned_text = clean_new_text(text)
    prepared_text = prepare_new_text(cleaned_text)
    
    # Make prediction
    prediction = loaded_model.predict(prepared_text)
    predicted_class_index = np.argmax(prediction[0])
    
    return class_labels[predicted_class_index], round(np.max(prediction[0]) * 100, 2)

def detect_themes_model(request, pk):
    """Handles the detection of themes using the trained model."""
    video = get_object_or_404(Video, pk=pk)
    transcript = video.transcript  # Assuming transcript is a field in Video model

    detected_model_theme = None
    detected_model_score = None

    if request.method == "POST" and "detect_model" in request.POST:
        print("DEBUG: Running model detection for video:", video.title)
        detected_model_theme, detected_model_score = predict(transcript)
        print("DEBUG: Model predicted theme:", detected_model_theme, "Score:", detected_model_score)

    context = {
        "video": video,
        "transcript": transcript,
        "detected_themes": None,  # API-based detection placeholder
        "detected_model_theme": detected_model_theme,
        "detected_model_score": detected_model_score,
    }
    return render(request, "detect_themes.html", context)
