# stream_oakd.py
import time
import cv2
import depthai as dai

# --- Config ---
PREVIEW_W, PREVIEW_H = 640, 480
FPS = 30

def main():
    pipeline = dai.Pipeline()

    cam = pipeline.create(dai.node.ColorCamera)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(FPS)
    cam.setPreviewSize(PREVIEW_W, PREVIEW_H)
    cam.setInterleaved(False)
    cam.setPreviewKeepAspectRatio(True)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    with dai.Device(pipeline) as dev:
        q = dev.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        cv2.namedWindow("OAK-D RGB", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("OAK-D RGB", PREVIEW_W, PREVIEW_H)

        last_t = time.time()
        fps = 0.0

        while True:
            pkt = q.tryGet()
            if pkt is None:
                # no new frame yet; keep UI responsive
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            frame = pkt.getCvFrame()
            now = time.time()
            dt = max(1e-9, now - last_t)
            last_t = now
            fps = 0.9*fps + 0.1*(1.0/dt)

            disp = frame.copy()
            cv2.putText(disp, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("OAK-D RGB", disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
