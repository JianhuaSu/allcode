import streamlit as st
import os
import subprocess
import argparse
import moviepy.editor as mp
import matplotlib.pyplot as plt
from io import BytesIO
from collections import Counter

# from audio_preprocess import AudioFeature
# from video_preprocess import VideoFeature

from kimi_api import chat_with_intent


def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_video_path', type=str, default='/root/autodl-tmp/demo/datasets/video_data/raw_video', help="The directory of the raw video path.")
    parser.add_argument('--raw_audio_path', type=str, default='/root/autodl-tmp/demo/datasets/audio_data/raw_audio', help="The directory of the raw audio path.")

    parser.add_argument('--video_data_path', type=str, default='/root/autodl-tmp/demo/datasets/video_data', help="The directory of the audio data path.")
    parser.add_argument("--video_feats", type=str, default='swin_feats.pkl', help="The directory of audio features.")
    
    parser.add_argument('--audio_data_path', type=str, default='/root/autodl-tmp/demo/datasets/audio_data', help="The directory of the audio data path.")
    parser.add_argument("--audio_feats", type=str, default='wavlm_feats.pkl', help="The directory of audio features.")
    
    parser.add_argument("--audio_model_path", type=str, default='/root/autodl-tmp/demo/mode/wavlm-libri-clean-100h-base-plus', help="The directory of audio model.")
    parser.add_argument("--audio_sr", type=int, default=16000, help="")

    args = parser.parse_args()

    return args



args = parse_arguments()

# audio_data = AudioFeature(args, get_raw_audio = True)
# video_data = VideoFeature(args)

# 路径设置
ori_video_path = '/root/autodl-tmp/demo/datasets/video_data/raw_video'
ori_text_path = '/root/autodl-tmp/demo/datasets/text.tsv'
prediction_results_path = "/root/autodl-tmp/demo/datasets/结果.txt"

# 确保视频文件夹存在
if not os.path.exists(ori_video_path):
    os.makedirs(ori_video_path)

# 读取预测结果并显示
def read_prediction_results(file_path):
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                # 假设每行的格式是 "method"-"prediction"，例如 "msz"-"Introduce"
                parts = line.strip().split('-')
                if len(parts) == 2:
                    method = parts[0].strip('"')  # 去除双引号
                    prediction = parts[1].strip('"')  # 去除双引号
                    results.append((method, prediction))
    except Exception as e:
        st.error(f"读取文件时出错: {e}")
    return results

# 处理视频和文本上传的函数
def handle_video_and_text_upload(uploaded_video, input_text):
    # 处理视频保存
    if uploaded_video is not None:
        file_extension = uploaded_video.name.split('.')[-1]
        new_video_name = f"video.{file_extension}"
        video_save_path = os.path.join(ori_video_path, new_video_name)
        # 保存视频文件
        with open(video_save_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.session_state.video_path = video_save_path  # 存储视频路径
        
        yinbotu = get_yinbo(video_save_path)
        st.session_state.yinbotu = yinbotu  # 存储视频路径

    # 处理文本保存
    if input_text:
        with open(ori_text_path, "w") as f:
            f.write(input_text)
        st.session_state.text_content = input_text  # 存储文本内容

# 处理预测逻辑的函数
def handle_prediction(prediction_results_path):
    try:

        # 切换到脚本需要的工作目录
        os.chdir('/root/autodl-tmp/demo/OOD_MSZ')
        # 执行 shell 脚本
        result = subprocess.run(['bash', '/root/autodl-tmp/demo/OOD_MSZ/examples/run.sh'], capture_output=True, text=True)
        # 如果文件中有内容，展示预测结果
        if os.path.exists(prediction_results_path):
            prediction_results = read_prediction_results(prediction_results_path)
            if prediction_results:
                st.session_state.prediction_results = prediction_results  # 将结果存储在session_state中
            else:
                st.warning("没有找到预测结果。")
    except Exception as e:
        st.error(f"执行脚本时出错: {str(e)}")



def get_yinbo(video_path):
    # 读取视频文件
    video = mp.VideoFileClip(video_path)
    # 提取音频
    audio = video.audio
    # 将音频转换为波形
    audio_waveform = audio.to_soundarray()
    # 提取音频参数
    sample_rate = audio.fps
    duration = audio.duration
    # 计算时间轴
    time = [i / sample_rate for i in range(len(audio_waveform))]
    
    # 绘制波形图（去除网格、标签、标题等）
    plt.figure(figsize=(10, 4))  # 设置图形的大小
    plt.plot(time, audio_waveform, color='blue', linewidth=0.5)  # 设置线条颜色和宽度
    
    # 不再显示xlabel、ylabel、title 和 grid
    plt.axis('off')  # 关闭坐标轴显示

    # 将图像保存到内存中
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)  # bbox_inches='tight'去除多余的空白
    img_buf.seek(0)  # 将指针回到图像的开始位置
    
    # 返回图像对象以供Streamlit显示
    return img_buf

def get_most_common_prediction(prediction_results):
    # 统计所有 prediction 的频率
    predictions = [prediction for _, prediction in prediction_results]
    prediction_counts = Counter(predictions)
    
    # 获取出现最多的 prediction
    most_common_prediction, count = prediction_counts.most_common(1)[0]
    
    return most_common_prediction, count



st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;  /* 增加页面容器宽度 */
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
    """, 
    unsafe_allow_html=True
)

# 设置页面标题
st.title("***")

# 使用st.columns创建两个列，左侧用于提交，右侧用于预测
col1, col2 = st.columns([1, 1])  # 左侧列宽2倍，右侧列宽1倍

# 左侧列 - 输入内容和按钮
with col1:
    with st.expander("输入框:", expanded=True):
        # 输入文本句子
        input_text = st.text_input("输入文本句子", key="input_text_key")

        # 上传视频文件
        uploaded_video = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov", "mkv", "flv"], key="upload_video_key")

        # 提交按钮逻辑
        if st.button("提交", key="submit_button"):
            handle_video_and_text_upload(uploaded_video, input_text)


# 右侧列 - 预测操作
with col2:
    with st.container():
        if st.button("预测", key="predict_button"):
            progress_bar = st.progress(0)
            handle_prediction(prediction_results_path)

        # 显示预测结果
        if 'prediction_results' in st.session_state:
            
            prediction_results = st.session_state.prediction_results
            with st.expander("预测详细结果:"):
                st.markdown(prediction_results)
                
            most_common_prediction, count = get_most_common_prediction(prediction_results)
            
            st.markdown(f'最终预测结果为：**"{most_common_prediction}"**')
            progress_bar.progress(0.5)  # 设置为100%完成

            if 'text_content' in st.session_state and most_common_prediction is not None:

                assistant_response, additional_suggestion = chat_with_intent(st.session_state.text_content, most_common_prediction)
                
                with st.expander("Assistant Response:"):
                    st.markdown(assistant_response)
                    
                with st.expander("Additional Suggestion:"):
                    st.markdown(additional_suggestion)
            progress_bar.progress(1)  # 设置为100%完成
            progress_bar.empty()
        else:
            st.info("点击预测按钮以查看结果。")
        
with st.container():
    # 创建左右两列布局
    col1, col2 = st.columns([1, 1])  # 1:1 列布局，你可以根据需要调整比例

    # 左侧列 - 显示文本和音频内容
    with col1:
        if 'text_content' in st.session_state:
            
            with st.expander("已上传的文本内容：", expanded=True):
                st.text(st.session_state.text_content)  
            st.empty()

        if 'yinbotu' in st.session_state:
            with st.expander("已上传的音频波形：", expanded=True):
                st.image(st.session_state.yinbotu) 
            st.empty()

    # 右侧列 - 显示视频内容
    with col2:
        if 'video_path' in st.session_state:
            
            with st.expander("已上传的视频内容：", expanded=True):
                st.video(st.session_state.video_path)        

        # 添加空白占位符来保证 col2 高度与 col1 一致
        st.empty()