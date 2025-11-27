import socket, struct, cv2, numpy as np

HOST = "192.168.4.1"
PORT = 2000

def recvall(sock,n):
    buf = b''
    while len(buf)< n:
        pkt = sock.recv(n- len(buf))
        if not pkt:
            print("Vedio length is wrong!")
            return None
        buf += pkt
    return buf


with socket.create_connection((HOST,PORT)) as s:
    print("Successfully create connections")
    while True:
        hdr = recvall(s,4)
        if hdr is None: break
        (length,) = struct.unpack('>I', hdr)
        data = recvall(s, length)
        if data is None: break
        img = cv2.imdecode(np.frombuffer(data,dtype = np.uint8),cv2.IMREAD_COLOR)
        cv2.imwrite("temp/test_frame.jpg", img)
        if img is None: continue
        #cv2.imshow('MJPEG TCP', img)
        #if cv2.waitKey(1) == 27: break
cv2.destroyAllWindows()
