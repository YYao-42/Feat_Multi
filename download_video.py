import yt_dlp
import subprocess
import os

def download_and_process_video(url, start_time, end_time, output_name, fps=30):
    # check if the video has been downloaded
    if os.path.exists('videos/ORI/' + output_name):
        print('Video already exists: ' + output_name)
        return
    start_time = str(int(start_time/3600)).zfill(2) + ':' + str(int(start_time/60)).zfill(2) + ':' + str(int(start_time%60)).zfill(2)
    end_time = str(int(end_time/3600)).zfill(2) + ':' + str(int(end_time/60)).zfill(2) + ':' + str(int(end_time%60)).zfill(2)
    output_path = 'videos/ORI/' + output_name
    # Options for yt-dlp
    options = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Choose best quality
        'outtmpl': 'video_dl.mp4',  # Output filename
    }
    # Download video using yt-dlp
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])
    # Trim the downloaded video to the specified start and end times using ffmpeg
    subprocess.run(['ffmpeg', '-ss', start_time, '-to', end_time, '-i', 'video_dl.mp4', 'video_trimmed.mp4'])
    # Resample the trimmed video to the specified fps using ffmpeg
    subprocess.run(['ffmpeg', '-i', 'video_trimmed.mp4', '-filter:v', 'fps='+str(fps), 'video_resampled.mp4'])
    # Resize the resampled video to 1920x1080 using ffmpeg (https://creatomate.com/blog/how-to-change-the-resolution-of-a-video-using-ffmpeg)
    subprocess.run(['ffmpeg', '-i', 'video_resampled.mp4', '-vf', 'scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1:color=black', output_path])
    # Optionally, remove the original downloaded video
    os.remove('video_dl.mp4')
    os.remove('video_trimmed.mp4')
    os.remove('video_resampled.mp4')


if not os.path.exists('videos/ORI'):
    os.makedirs('videos/ORI')

videos_info = [
    {'url': 'https://youtu.be/uOUVE5rGmhM', 'start_time': 9, 'end_time': 231, 'output_name': '01_Dance_1.mp4'},
    {'url': 'https://youtu.be/36fRhtJdWQ4', 'start_time': 17, 'end_time': 479, 'output_name': '02_Mime_1.mp4'},
    {'url': 'https://youtu.be/DjihbYg6F2Y', 'start_time': 5, 'end_time': 231, 'output_name': '03_Acrob_1.mp4'},
    {'url': 'https://youtu.be/CvzMqIQLiXE', 'start_time': 4, 'end_time': 348, 'output_name': '04_Magic_1.mp4'},
    {'url': 'https://youtu.be/f4DZp0OEkK4', 'start_time': 6, 'end_time': 228, 'output_name': '05_Dance_2.mp4'},
    {'url': 'https://youtu.be/u9wJUTnBdrs', 'start_time': 6, 'end_time': 347, 'output_name': '06_Mime_2.mp4'},
    {'url': 'https://youtu.be/kRqdxGPLajs', 'start_time': 184, 'end_time': 519, 'output_name': '07_Acrob_2.mp4'},
    {'url': 'https://youtu.be/FUv-Q6EgEFI', 'start_time': 4, 'end_time': 270, 'output_name': '08_Magic_2.mp4'},
    {'url': 'https://youtu.be/LXO-jKksQkM', 'start_time': 6, 'end_time': 294, 'output_name': '09_Dance_3.mp4'},
    {'url': 'https://youtu.be/S84AoWdTq3E', 'start_time': 2, 'end_time': 426, 'output_name': '12_Magic_3.mp4'},
    {'url': 'https://youtu.be/0wc60tA1klw', 'start_time': 15, 'end_time': 217, 'output_name': '13_Dance_4.mp4'},
    {'url': 'https://youtu.be/0Ala3ypPM3M', 'start_time': 22, 'end_time': 386, 'output_name': '14_Mime_3.mp4'},
    {'url': 'https://youtu.be/mg6-SnUl0A0', 'start_time': 16, 'end_time': 233, 'output_name': '15_Dance_5.mp4'},
    {'url': 'https://youtu.be/8V7rhAJF6Gc', 'start_time': 32, 'end_time': 388, 'output_name': '16_Mime_6.mp4'},
    # Add more videos as needed
]

for video_info in videos_info:
    download_and_process_video(video_info['url'], video_info['start_time'], video_info['end_time'], video_info['output_name'])
