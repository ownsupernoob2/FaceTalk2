import os, shutil
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
import openai
from google.cloud import texttospeech
import uuid
from markupsafe import escape
from wav2lip.wav2lip import FaceVideoMaker
import pytz

work_dir = 'temp'
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
files_to_delete = []

print('Server initialization...')
app = Flask(__name__)
faceVideoMaker = FaceVideoMaker(audio_dir=work_dir, video_dir=work_dir)
start_time = time.time()

# OPENAI Related
# OPENAI token usage statistics
openai.api_key = os.getenv("OPENAI_API_KEY")

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


# OPENAI API-KEY access restrictions
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

    print_log()

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


# OPENAI API calls
def fetch_chat_response_v1(text, api_key):
    return fetch_chat_response([{"role": "user", "content": text}], api_key)


def fetch_chat_response(messages, api_key):
    print(messages)
    return openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )


def parse_chat_response(response, useSiteApiKey):
    # print('parse_chat_response')
    message = response['choices'][0]['message']['content']
    # print(response['usage'])
    if useSiteApiKey:
        global site_api_key_completion_tokens
        global site_api_key_prompt_tokens
        global site_api_key_total_tokens
        global site_api_key_requests
        site_api_key_completion_tokens += response['usage']['completion_tokens']
        site_api_key_prompt_tokens += response['usage']['prompt_tokens']
        site_api_key_total_tokens += response['usage']['total_tokens']
        site_api_key_requests += 1
    else:
        global custom_api_key_completion_tokens
        global custom_api_key_prompt_tokens
        global custom_api_key_total_tokens
        global custom_api_key_requests
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


# GOOGLE Text-to-Speech related
textToSpeechClient = texttospeech.TextToSpeechClient()
voice = texttospeech.VoiceSelectionParams(
    language_code="en-GB",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    speaking_rate=1.25
)


def text_to_speech(text, filename):
    # test_time = time. time()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = textToSpeechClient.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config)
    audio_path = os.path.join(work_dir, f'{filename}.wav')
    with open(audio_path, "wb") as out:
        out.write(response.audio_content)
        # print('TTS test time:' + str(time.time() - test_time))


# Flask routing
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/message', methods=['POST'])
def message_deprecate():
    data = request.json
    # print(data)
    useSiteApiKey = data.get('key_type') == 'default'
    if useSiteApiKey:
        if is_limit_reached():
            # Error code 1: The api-key of this site exceeds the usage limit
            return jsonify({'error_code': 1}), 200
    # print("before fetch_chat_response")
    try:
        if useSiteApiKey:
            response = fetch_chat_response_v1(data.get('message'), openai.api_key)
            add_counter()
        else:
            api_key = data.get('api_key')
            if not api_key:
                # Error code 2: The external api-key is empty or invalid
                return jsonify({'error_code': 2}), 200
            response = fetch_chat_response_v1(data.get('message'), api_key)
    except openai.error.AuthenticationError as e:
        # Error code 1: The api-key of this site is in arrears or invalid, and it will be handled as exceeding the usage limit
        # (Because any information of the api-key on this site should be avoided from being exposed, so related errors are uniformly returned exceeding the usage limit)
        # Error code 2: The external api-key is empty or invalid
        return jsonify({'error_code': 1 if useSiteApiKey else 2}), 200
    except openai.error.RateLimitError as e:
        # Error code 3: api-key is used too frequently within a certain period of time
        return jsonify({'error_code': 3}), 200
    except (openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError) as e:
        # Error code 4: OPENAI API service exception
        return jsonify({'error_code': 4}), 200
    except Exception as e:
        # Error code 0: unknown error
        return jsonify({'error_code': 0}), 200

    message = parse_chat_response(response, useSiteApiKey)

    is_video_mode = data.get('video')
    # print(f'is_video_mode: {is_video_mode}')
    if not is_video_mode:
        return jsonify({'message': escape(message)}), 200

    message_no_code_block = remove_code_block(message)
    id = str(uuid.uuid4())[:8]
    text_to_speech(message_no_code_block, id)
    faceVideoMaker.makeVideo(id)

    return jsonify({'message': escape(message), 'video_url': f'/{work_dir}/{id}.mp4'}), 200


@app.route('/api/messagev2', methods=['POST'])
def message_v2():
    data = request.json
    # print(data)
    useSiteApiKey = data.get('key_type') == 'default'
    if useSiteApiKey:
        if is_limit_reached():
            # Error code 1: The api-key of this site exceeds the usage limit
            return jsonify({'error_code': 1}), 200
    # print("before fetch_chat_response")
    try:
        if useSiteApiKey:
            response = fetch_chat_response(data.get('messages'), openai.api_key)
            add_counter()
        else:
            api_key = data.get('api_key')
            if not api_key:
                # Error code 2: The external api-key is empty or invalid
                return jsonify({'error_code': 2}), 200
            response = fetch_chat_response(data.get('messages'), api_key)
    except openai.error.AuthenticationError as e:
        # Error code 1: The api-key of this site is in arrears or invalid, and it will be handled as exceeding the usage limit
        # (Because any information of the api-key on this site should be avoided from being exposed, so related errors are uniformly returned exceeding the usage limit)
        # Error code 2: The external api-key is empty or invalid
        return jsonify({'error_code': 1 if useSiteApiKey else 2}), 200
    except openai.error.RateLimitError as e:
        # Error code 3: api-key is used too frequently within a certain period of time
        return jsonify({'error_code': 3}), 200
    except (openai.error.APIError, openai.error.Timeout, openai.error.APIConnectionError) as e:
        # Error code 4: OPENAI API service exception
        return jsonify({'error_code': 4}), 200
    except Exception as e:
        # Error code 0: unknown error
        return jsonify({'error_code': 0}), 200

    message = parse_chat_response(response, useSiteApiKey)

    is_video_mode = data.get('video')
    # print(f'is_video_mode: {is_video_mode}')
    if not is_video_mode:
        return jsonify({'message': escape(message)}), 200

    message_no_code_block = remove_code_block(message)
    id = str(uuid.uuid4())[:8]
    text_to_speech(message_no_code_block, id)
    faceVideoMaker.makeVideo(id)

    return jsonify({'message': escape(message), 'video_url': f'/{work_dir}/{id}.mp4'}), 200


@app.route(f'/{work_dir}/<path:filename>')
def get_video_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), work_dir), filename)


@app.route('/api/face_img')
def face_img():
    return send_from_directory(os.path.join(os.getcwd(), 'assets'), 'face_2.jpg')
