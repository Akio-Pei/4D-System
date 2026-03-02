import socket
import struct
import cv2
import numpy as np
import threading
import time

HOST = '0.0.0.0'
PORT_VIDEO = 8888
PORT_THERMAL = 8889

# Tiny1-C 固定分辨率
THERMAL_W = 256
THERMAL_H = 192
EXPECTED_SIZE = THERMAL_W * THERMAL_H * 2  # 98304


def recv_all(sock, count):
    buf = b''
    while count:
        try:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        except:
            return None
    return buf


def video_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_VIDEO))
    s.listen(1)
    print(f">>> [Video] Listening on {PORT_VIDEO}...")

    while True:
        try:
            conn, addr = s.accept()
            print(f">>> [Video] Connected: {addr}")
            fps_count = 0
            start_time = time.time()

            while True:
                header = recv_all(conn, 16)
                if not header: break
                ts, size, fid = struct.unpack("=QII", header)

                data = recv_all(conn, size)
                if not data: break

                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    fps_count += 1
                    if time.time() - start_time > 1.0:
                        print(f"[Video 9281] FPS: {fps_count} | TS: {ts}")
                        fps_count = 0
                        start_time = time.time()
                    cv2.imshow("OV9281", frame)
                    if cv2.waitKey(1) == 27: return
        except Exception as e:
            print(f"[Video Error] {e}")
        finally:
            if 'conn' in locals(): conn.close()


def thermal_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT_THERMAL))
    s.listen(1)
    print(f">>> [Thermal] Listening on {PORT_THERMAL}...")

    while True:
        try:
            conn, addr = s.accept()
            print(f">>> [Thermal] Connected: {addr}")

            while True:
                header = recv_all(conn, 16)
                if not header: break
                ts, size, fid = struct.unpack("=QII", header)

                # 严格检查大小
                if size != EXPECTED_SIZE:
                    print(f"[Thermal] Error: Size mismatch! Recv {size}, Expect {EXPECTED_SIZE}")
                    # 如果大小不对，可能是流乱了，尝试读掉数据但不处理
                    recv_all(conn, size)
                    continue

                raw_data = recv_all(conn, size)
                if not raw_data: break

                # 解析
                try:
                    temp_array = np.frombuffer(raw_data, dtype=np.uint16).reshape((THERMAL_H, THERMAL_W))

                    # 中心点温度
                    center_val = temp_array[THERMAL_H // 2, THERMAL_W // 2]
                    center_temp = (float(center_val) / 64.0) - 273.15

                    # 伪彩显示
                    norm_img = cv2.normalize(temp_array, None, 0, 255, cv2.NORM_MINMAX)
                    color_img = cv2.applyColorMap(norm_img.astype(np.uint8), cv2.COLORMAP_JET)

                    # 文字信息
                    text = f"Temp: {center_temp:.1f} C"
                    cv2.putText(color_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    cv2.imshow("Thermal", color_img)
                    if cv2.waitKey(1) == 27: return

                except Exception as e:
                    print(f"[Thermal Data Error] {e}")

        except Exception as e:
            print(f"[Thermal Socket Error] {e}")
        finally:
            if 'conn' in locals(): conn.close()
            print(">>> [Thermal] Connection lost, waiting...")


if __name__ == '__main__':
    t1 = threading.Thread(target=video_server)
    t2 = threading.Thread(target=thermal_server)
    t1.start()
    t2.start()
    t1.join()
    t2.join()