'''
This script is used to present the video stimuli and send triggers to EEG recorder and NEON (as a backup synchronization method).
The video stimuli are presented in a full screen mode.
The trigger to the EEG recorder is sent via a serial port, and the event to NEON is sent via NEON's real-time API.
The trigger and event are scheduled to be sent once per minute.
        
Author: yuanyuan.yao@kuleuven.be
'''

import nest_asyncio
import socket
import time
import vlc
import numpy as np
import keyboard
from threading import Thread
from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.realtime_api.simple import Device
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.background import BackgroundScheduler
import serial.tools.list_ports
import serial


def send_trigger_per_min(device, port, offset_ms):
    port.write(serial_trigger.to_bytes(2, 'big'))
    timestamp_ns = time.time_ns()
    companion_app_time = timestamp_ns - offset_ms*1e6
    print(device.send_event("Start of 1 min (NEON)"))
    print(device.send_event("Start of 1 min (PC offset corrected)", event_timestamp_unix_ns=companion_app_time))


def callback(player, vlc_instance, device, port, sched):
    print('FPS:', player.get_fps())
    print('Time(ms):', player.get_time())
    print('Frame:', .001 * player.get_time() * player.get_fps())
    player.stop()
    player.release()
    vlc_instance.release()
    device.recording_stop_and_save()
    port.close()
    sched.shutdown()


# Set up serial connection
# ports = serial.tools.list_ports.comports()
# for port, desc, hwid in sorted(ports):
#     print("{}: {} [{}]".format(port, desc, hwid))
with serial.Serial('COM3', 115200, serial.EIGHTBITS,timeout=0,parity=serial.PARITY_NONE, 
rtscts=1) as ser:
  print(ser.is_open)
  ser.close()
port = serial.Serial("COM3", baudrate=115200)
serial_trigger = int(1)


# Set up NEON
nest_asyncio.apply()
# device = discover_one_device()
# If doesn't work, try to set the IP manually
ip = "192.168.137.207"
device = Device(address=ip, port="8080")
if device is None:
    print("No device found.")
    port.close()
    exit()
print(f"Phone IP address: {device.phone_ip}")
print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.module_serial}")
# Check time offset between NEON and PC
# https://docs.pupil-labs.com/neon/how-tos/data-collection/achieve-super-precise-time-sync.html
estimate = device.estimate_time_offset()
offset_ms = estimate.time_offset_ms.mean
print(f"Estimated time offset: {offset_ms:.3f} ms")
# Start NEON
device.recording_start()
# wait for NEON to start functioning
time.sleep(3)

# creating vlc media player object
vlc_instance = vlc.Instance(['--video-on-top'])
media_player = vlc_instance.media_player_new()
# media = vlc.Media("videos/blink_wgb.avi")
media = vlc.Media("videos/video_test_size_120_fps_30_len_7.avi")
media_player.set_media(media)
media_player.set_fullscreen(True)


# Align NEON and Biosemi at the start of each minute.
# sched = BlockingScheduler()
# sched.add_job(send_trigger_per_min, trigger=CronTrigger(second=00), args=[device, port, offset_ms])
# sched.start()

sched = BackgroundScheduler()
sched.add_job(send_trigger_per_min, trigger='interval', seconds=60, args=[device, port, offset_ms])  # Example: Run every 60 seconds


# Quit when Esc is pressed
keyboard.add_hotkey("Esc", callback, args=[media_player, vlc_instance, device, port, sched])


# start playing video
# print(device.send_event("Play (before execution)", event_timestamp_unix_ns=time.time_ns()))
media_player.play() 
# The idea now is to align two event signals (one from stimulus pc to EEG recorder via serial port, and the other from stimulus pc to NEON via socket). The exact timepoint of media_player.play() is unimportant because the video does not present on the screen immediately anyway.
port.write(serial_trigger.to_bytes(1, 'big'))
timestamp_port = time.time_ns()
print(device.send_event("Play (after execution)", event_timestamp_unix_ns=(timestamp_port - offset_ms*1e6)))
sched.start()
while True:
    if media_player.get_state() == vlc.State.Ended:
        port.write(serial_trigger.to_bytes(1, 'big'))
        timestamp_end = time.time_ns()
        print(device.send_event("End of video", event_timestamp_unix_ns=(timestamp_end - offset_ms*1e6)))
        callback(media_player, vlc_instance, device, port, sched)
        break
