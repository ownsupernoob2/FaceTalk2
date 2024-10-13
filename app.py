import os
import shutil
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from openai import AzureOpenAI
import azure.cognitiveservices.speech as speechsdk
import uuid
from markupsafe import escape
from wav2lip.wav2lip import FaceVideoMaker
import pytz

# Azure OpenAI setup
client = AzureOpenAI(
    azure_endpoint="",
    api_key="",
    api_version="2023-05-15"
)

# Azure Text-to-Speech setup
speech_config = speechsdk.SpeechConfig(subscription="", region="eastus")
speech_config.speech_synthesis_voice_name = "en-GB-SoniaNeural"

work_dir = 'temp'
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
files_to_delete = []

print('Server initialization...')
app = Flask(__name__)
faceVideoMaker = FaceVideoMaker(audio_dir=work_dir, video_dir=work_dir)
start_time = time.time()

# Token usage statistics
site_api_key_completion_tokens = 0
site_api_key_prompt_tokens = 0
site_api_key_total_tokens = 0
site_api_key_requests = 0
custom_api_key_completion_tokens = 0
custom_api_key_prompt_tokens = 0
custom_api_key_total_tokens = 0
custom_api_key_requests = 0

def print_log():
    running_time = time.time() - start_time
    folder = './temp'

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    print('Running duration:' + str(int(running_time / 3600)) + 'hours' + str(
        int(running_time % 3600 / 60)) + 'minutes' + str(int(running_time % 60)) + ' Second')
    print('------------------------------------')
    print('The total number of conversations of this sites api-key: ' + str(site_api_key_requests))
    print('This site api-key completion tokens: ' + str(site_api_key_completion_tokens))
    print('This site api-key prompt tokens: ' + str(site_api_key_prompt_tokens))
    print('This site api-key total tokens: ' + str(site_api_key_total_tokens))
    print('------------------------------------')
    print('External api-key total conversations: ' + str(custom_api_key_requests))
    print('External api-key completion tokens: ' + str(custom_api_key_completion_tokens))
    print('External api-key prompt tokens: ' + str(custom_api_key_prompt_tokens))
    print('External api-key total tokens: ' + str(custom_api_key_total_tokens))

# API-KEY access restrictions
API_LIMIT_PER_HOUR = 1000
counter = 0

def reset_counter():
    global counter
    counter = 0

def add_counter():
    global counter
    counter += 1

def is_limit_reached():
    global counter
    return counter >= API_LIMIT_PER_HOUR

# Timed tasks
def hourly_maintain():
    reset_counter()
    print_log()

    for file in files_to_delete:
        os.remove(file)
    files_to_delete.clear()
    for filename in os.listdir(work_dir):
        file = os.path.join(work_dir, filename)
        if os.path.isfile(file):
            files_to_delete.append(file)

scheduler = BackgroundScheduler(timezone=pytz.utc)
next_hour_time = datetime.fromtimestamp(time.time() + 3600 - time.time() % 3600)
scheduler.add_job(hourly_maintain, 'interval', minutes=60, next_run_time=next_hour_time)
scheduler.start()

# Azure OpenAI API calls
def fetch_chat_response_v1(text, api_key):
    return fetch_chat_response([{"role": "user", "content": text}], api_key)

def fetch_chat_response(messages, api_key):
    print(messages)
    return client.chat.completions.create(model="gpt-4",
    messages=messages,
    max_tokens=100)

def parse_chat_response(response, useSiteApiKey):
    message = response['choices'][0]['message']['content']
    if useSiteApiKey:
        global site_api_key_completion_tokens, site_api_key_prompt_tokens, site_api_key_total_tokens, site_api_key_requests
        site_api_key_completion_tokens += response['usage']['completion_tokens']
        site_api_key_prompt_tokens += response['usage']['prompt_tokens']
        site_api_key_total_tokens += response['usage']['total_tokens']
        site_api_key_requests += 1
    else:
        global custom_api_key_completion_tokens, custom_api_key_prompt_tokens, custom_api_key_total_tokens, custom_api_key_requests
        custom_api_key_completion_tokens += response['usage']['completion_tokens']
        custom_api_key_prompt_tokens += response['usage']['prompt_tokens']
        custom_api_key_total_tokens += response['usage']['total_tokens']
        custom_api_key_requests += 1
    return message

def remove_code_block(message):
    if message.find('```') == -1:
        return message
    else:
        return ''.join([words for i, words in enumerate(message.split('```')) if i % 2 == 0])

# Azure Text-to-Speech
def text_to_speech(text, filename):
    audio_path = os.path.join(work_dir, f'{filename}.wav')
    audio_config = speechsdk.audio.AudioOutputConfig(filename=audio_path)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

# Flask routing
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/message', methods=['POST'])
def message_deprecate():
    data = request.json
    useSiteApiKey = data.get('key_type') == 'default'
    if useSiteApiKey:
        if is_limit_reached():
            return jsonify({'error_code': 1}), 200
    try:
        if useSiteApiKey:
            response = fetch_chat_response_v1(data.get('message'), openai.api_key)
            add_counter()
        else:
            api_key = data.get('api_key')
            if not api_key:
                return jsonify({'error_code': 2}), 200
            response = fetch_chat_response_v1(data.get('message'), api_key)
    except openai.AuthenticationError as e:
        return jsonify({'error_code': 1 if useSiteApiKey else 2}), 200
    except openai.RateLimitError as e:
        return jsonify({'error_code': 3}), 200
    except (openai.APIError, openai.Timeout, openai.APIConnectionError) as e:
        return jsonify({'error_code': 4}), 200
    except Exception as e:
        return jsonify({'error_code': 0}), 200

    message = parse_chat_response(response, useSiteApiKey)

    is_video_mode = data.get('video')
    if not is_video_mode:
        return jsonify({'message': escape(message)}), 200

    message_no_code_block = remove_code_block(message)
    id = str(uuid.uuid4())[:8]
    text_to_speech(message_no_code_block, id)
    faceVideoMaker.makeVideo(id)

    return jsonify({'message': escape(message), 'video_url': f'/{work_dir}/{id}.mp4'}), 200


def fetch_chat_response(messages, client):
    print(messages)
    return client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=100
    )

def parse_chat_response(response, useSiteApiKey):
    message = response.choices[0].message.content
    if useSiteApiKey:
        global site_api_key_completion_tokens, site_api_key_prompt_tokens, site_api_key_total_tokens, site_api_key_requests
        site_api_key_completion_tokens += response.usage.completion_tokens
        site_api_key_prompt_tokens += response.usage.prompt_tokens
        site_api_key_total_tokens += response.usage.total_tokens
        site_api_key_requests += 1
    else:
        global custom_api_key_completion_tokens, custom_api_key_prompt_tokens, custom_api_key_total_tokens, custom_api_key_requests
        custom_api_key_completion_tokens += response.usage.completion_tokens
        custom_api_key_prompt_tokens += response.usage.prompt_tokens
        custom_api_key_total_tokens += response.usage.total_tokens
        custom_api_key_requests += 1
    return message

@app.route('/api/messagev2', methods=['POST'])
def message_v2():
    data = request.json
    useSiteApiKey = data.get('key_type') == 'default'
    if useSiteApiKey and is_limit_reached():
        return jsonify({'error_code': 1}), 200

    try:
        if useSiteApiKey:
            response = fetch_chat_response(data.get('messages'), client)
            add_counter()
        else:
            api_key = data.get('api_key')
            if not api_key:
                return jsonify({'error_code': 2}), 200
            custom_client = AzureOpenAI(
                azure_endpoint="https://facetalk.openai.azure.com/",
                api_key=api_key,
                api_version="2023-05-15"
            )
            response = fetch_chat_response(data.get('messages'), custom_client)
    except Exception as e:
        error_code = handle_openai_error(e, useSiteApiKey)
        return jsonify({'error_code': error_code}), 200

    message = parse_chat_response(response, useSiteApiKey)

    is_video_mode = data.get('video')
    if not is_video_mode:
        return jsonify({'message': escape(message)}), 200

    message_no_code_block = remove_code_block(message)
    id = str(uuid.uuid4())[:8]
    text_to_speech(message_no_code_block, id)
    faceVideoMaker.makeVideo(id)

    return jsonify({'message': escape(message), 'video_url': f'/{work_dir}/{id}.mp4'}), 200

def handle_openai_error(e, useSiteApiKey):
    if isinstance(e, AzureOpenAI.AuthenticationError):
        return 1 if useSiteApiKey else 2
    elif isinstance(e, AzureOpenAI.RateLimitError):
        return 3
    elif isinstance(e, (AzureOpenAI.APIError, AzureOpenAI.Timeout, AzureOpenAI.APIConnectionError)):
        return 4
    else:
        print(f"Unexpected error: {str(e)}")
        return 0

@app.route(f'/{work_dir}/<path:filename>')
def get_video_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), work_dir), filename)

@app.route('/api/face_img')
def face_img():
    return send_from_directory(os.path.join(os.getcwd(), 'assets'), 'face_2.jpg')

if __name__ == '__main__':
    app.run(debug=True)