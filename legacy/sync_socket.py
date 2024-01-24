import nest_asyncio
import socket
import time
import vlc
import numpy as np
import keyboard
from threading import Thread
from pupil_labs.realtime_api.simple import discover_one_device
from pupil_labs.realtime_api.simple import Device
import serial.tools.list_ports
import serial


def read_one_package(con, n_channels, sample_per_channel, bytes_per_sample, sensor_id):
    BUFFER_SIZE = n_channels * sample_per_channel * bytes_per_sample
    data = con.recv(BUFFER_SIZE)
    data_row_sample = np.reshape(list(data), (n_channels*sample_per_channel, bytes_per_sample))
    data_sensor = data_row_sample[sample_per_channel*sensor_id:sample_per_channel*(sensor_id+1), :]
    data_int_24bit = (data_sensor[:, 2] * 65536) + (data_sensor[:, 1] * 256) + data_sensor[:, 0]
    # If MSB is set: turn negative and reduce MSB value
    msb_set = data_int_24bit >= 8388608
    data_int_24bit[msb_set] = -(data_int_24bit[msb_set] - 8388608)
    # Convert result to microvolt
    # data_mv = (data_int_24bit * 31.25) / 1000 # for EEG channels
    return data_int_24bit

def callback(player, vlc_instance, device, socket):
    print('FPS:', player.get_fps())
    print('Time(ms):', player.get_time())
    print('Frame:', .001 * player.get_time() * player.get_fps())
    player.stop()
    player.release()
    vlc_instance.release()
    device.recording_stop_and_save()
    socket.close()


# Set up serial connection
ports = serial.tools.list_ports.comports()
for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))
port = serial.Serial("COM3", baudrate=115200)
serial_trigger = 42
serial_trigger = serial_trigger.to_bytes(1, byteorder='big')


# Set up TCP connection
# TCP_IP = '192.168.137.33'
TCP_IP = '127.0.0.1'
TCP_PORT = 8888
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
sensor_id = 2
n_channels = 7
sample_per_channel = 16
bytes_per_sample = 3


# Set up NEON
nest_asyncio.apply()
# device = discover_one_device()
# If doesn't work, try to set the IP manually
ip = "192.168.137.207"
device = Device(address=ip, port="8080")
print(f"Phone IP address: {device.phone_ip}")
print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.module_serial}")
# Start NEON
device.recording_start()


# creating vlc media player object
vlc_instance = vlc.Instance(['--video-on-top'])
media_player = vlc_instance.media_player_new()
# media = vlc.Media("videos/Cello.mp4")
media = vlc.Media("../../../../Videos/Exp/exp_part1.avi")
media_player.set_media(media)
media_player.set_fullscreen(True)


keyboard.add_hotkey("Esc", callback, args=[media_player, vlc_instance, device, s])


# start playing video
# print(device.send_event("Play (before execution)", event_timestamp_unix_ns=time.time_ns()))
media_player.play()
port.write(serial_trigger)
print(device.send_event("Play (after execution)", event_timestamp_unix_ns=time.time_ns()))
while True:
    data_one_package = read_one_package(s, n_channels, sample_per_channel, bytes_per_sample, sensor_id)
    data_mean = np.mean(data_one_package)
    if data_mean > -1570000:
        print(device.send_event("Not black", event_timestamp_unix_ns=time.time_ns()))
    if media_player.get_state() == vlc.State.Ended:
        callback(media_player, vlc_instance, device, s)
        print(device.send_event("End of video", event_timestamp_unix_ns=time.time_ns()))
        break
