#!/usr/bin/python
'''
	Streaming the openCV to a server.
	Adapted from: https://gist.github.com/n3wtron/4624820
'''
import cv2
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
import time

capture=None

class CamHandler(BaseHTTPRequestHandler):
	def do_GET(self):
		print self.path
		if self.path.endswith('.mjpg'):
			self.send_response(20)
			self.send_header('Content-type','multipart/x-mixed-replace; boundary=--jpgboundary')
			self.end_headers()
			while True:
				try:
					rc,img = capture.read()
					if not rc:
						continue
					imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
					r, buf = cv2.imencode(".jpg",imgRGB)
					self.wfile.write("--jpgboundary\r\n")
					self.send_header('Content-type','image/jpeg')
					self.send_header('Content-length',str(len(buf)))
					self.end_headers()
					self.wfile.write(bytearray(buf))
					self.wfile.write('\r\n')
				except KeyboardInterrupt:
					break
			return
		if self.path.endswith('.html') or self.path=="/":
			self.send_response(200)
			self.send_header('Content-type','text/html')
			self.end_headers()
			self.wfile.write('<html><head></head><body style="overflow-x:hidden;overflow-y:hidden">')
			self.wfile.write('<img src="http://127.0.0.1:9090/cam.mjpg" height="780px" width="1366px"/>')
			self.wfile.write('</body></html>')
			return

def main():
	global capture
	capture = cv2.VideoCapture(0)
	capture.set(1366, 320); 
	capture.set(768, 240);
	try:
		server = HTTPServer(('',9090),CamHandler)
		print "server started"
		server.serve_forever()
	except KeyboardInterrupt:
		capture.release()
		server.socket.close()

if __name__ == '__main__':
	main()
