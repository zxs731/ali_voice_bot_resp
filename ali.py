import pyaudio
import dashscope
from dashscope.audio.asr import TranslationRecognizerRealtime, TranslationRecognizerCallback,TranscriptionResult,TranslationResult
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback,AudioFormat
from dashscope import Generation
from http import HTTPStatus
import threading
import time
import os
from dotenv import load_dotenv
load_dotenv("ali.env")

# 设置 DashScope API Key
dashscope.api_key = os.environ["key"]

# 音频参数
ASR_FORMAT = "pcm"
ASR_SAMPLE_RATE = 16000
TTS_FORMAT = AudioFormat.PCM_22050HZ_MONO_16BIT
TTS_RATE = 22050

# 全局状态变量
#mic = None
#stream = None
asr_callback = None
recognizer = None
user_input_ready = threading.Event()
user_input_text = ""

# 回调类 - ASR
class ASRCallback(TranslationRecognizerCallback):
    def __init__(self):
        self.transcription_buffer = ""
        self.timer = None
        self.is_listening = True
        self.mic=None
        self.stream=None

    def on_open(self):
        #global mic, stream
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=ASR_SAMPLE_RATE,
            input=True
        )
        print("ASR: 语音识别已启动，请开始说话...")

    def on_close(self):
        #global mic, stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream=None
        if self.mic:
            self.mic.terminate()
            self.mic=None
        print("ASR: 语音识别已关闭。")

    def on_event(self, request_id, transcription_result: TranscriptionResult, translation_result: TranslationResult, usage):
        global user_input_text, user_input_ready

        if transcription_result:
            current_text = transcription_result.text.strip()
            if current_text:
                self.update_buffer(current_text)

    def update_buffer(self, text):
        global user_input_text
        self.transcription_buffer = text
        self.reset_timer()

    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(1, self.on_timeout)
        self.timer.start()

    def on_timeout(self):
        global user_input_text, user_input_ready
        user_input_text = self.transcription_buffer.strip()
        if user_input_text:
            print("检测到停顿，用户输入完成：", user_input_text)
            self.is_listening = False
            user_input_ready.set()

# 回调类 - TTS
class TTSCallback(ResultCallback):
    def __init__(self):
        self._player = None
        self._stream = None

    def on_open(self):
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTS_RATE,
            output=True
        )
        print("TTS: 语音合成已启动。")

    def on_close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream=None
        if self._player:
            self._player.terminate()
            self._player=None
        print("TTS: 语音合成已关闭。")

    def on_data(self, data: bytes):
        if self._stream:
            self._stream.write(data)

messages=[]
# 处理用户输入，调用大模型生成回复
def process_input(user_input):
    global messages
    print("处理用户输入：", user_input)
    messages += [{"role": "user", "content": user_input}]
    responses = Generation.call(
        model="qwen-turbo",
        messages=messages,
        result_format="message",
        stream=True,
        incremental_output=True
    )

    # 初始化 TTS
    tts_callback = TTSCallback()
    synthesizer = SpeechSynthesizer(
        model="cosyvoice-v1",
        voice="longxiaochun",
        format=TTS_FORMAT,
        callback=tts_callback
    )

    reply = ""
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]["message"]["content"]
            reply += content
            print(content,end="",flush=True)
            synthesizer.streaming_call(content)  # 不需要 start()
        else:
            print("生成回复失败：", response.message)

    messages += [{"role": "assistant", "content": reply}]

    synthesizer.streaming_complete()  # 仍需调用
    print("1111111")
    #tts_callback.on_close()
    print('回复播放完成，重新进入监听状态。')

# 主循环：持续监听并处理语音输入
def run_assistant():
    while True:
        print("等待用户输入...")
        global asr_callback, recognizer
        # 重置旧的 ASR 实例
        if asr_callback:
            asr_callback = None
        if recognizer:
            recognizer = None

        asr_callback = ASRCallback()
        recognizer = TranslationRecognizerRealtime(
            model="gummy-realtime-v1",
            format=ASR_FORMAT,
            sample_rate=ASR_SAMPLE_RATE,
            transcription_enabled=True,
            translation_enabled=False,
            callback=asr_callback
        )
        recognizer.start()
        asr_callback.is_listening = True

        while asr_callback.is_listening:
            if asr_callback.stream:
                try:
                    data = asr_callback.stream.read(3200, exception_on_overflow=False)
                    recognizer.send_audio_frame(data)
                except Exception as e:
                    print("录音出错：", e)
                    break
            else:
                break

        recognizer.stop()
        #asr_callback.on_close()

        if user_input_ready.is_set():
            process_input(user_input_text)
            user_input_ready.clear()

# 启动语音助手
if __name__ == "__main__":
    user_input_ready.clear()
    run_assistant()
