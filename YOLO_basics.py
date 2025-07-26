from ultralytics import YOLO
import cv2
import numpy as np

def sharpen_frame(frame):
    blurred   = cv2.GaussianBlur(frame, (0, 0), sigmaX=3, sigmaY=3)
    return cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

def main(input_path: str,
         output_path: str,
         threshold: int = 30,
         conf_thresh: float = 0.1,
         img_size: int = 1024,
         up_scale: float = 1.2):
    # load
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path}")
        return

    cv2.namedWindow('Counting People', cv2.WINDOW_NORMAL)
    writer = None

    print(f"Running with conf>{conf_thresh}, size={img_size}, up‑scale={up_scale}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # sharpen & optionally upscale
        frame_sharp = sharpen_frame(frame)
        if up_scale != 1.0:
            frame_sharp = cv2.resize(frame_sharp, None, fx=up_scale, fy=up_scale, interpolation=cv2.INTER_LINEAR)

        # run detection with lower conf, larger imgsz, augment
        results = model(frame_sharp,
                        conf=conf_thresh,
                        iou=0.45,
                        imgsz=img_size,
                        augment=True)

        # prepare writer once
        h, w = frame.shape[:2]
        if writer is None:
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # count & draw
        count = 0
        for res in results:
            for box in res.boxes:
                if int(box.cls) == 0:
                    # YOLO returned coords on the *up‑scaled* frame if you upscaled it
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if up_scale != 1.0:
                        x1, y1, x2, y2 = [coord / up_scale for coord in (x1, y1, x2, y2)]
                    count += 1
                    cv2.rectangle(frame,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)),
                                  (0, 255, 0), 2)

        # overlay count + threshold warning
        cv2.putText(frame, f'Count: {count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        if count > threshold:
            cv2.putText(frame, '!!! UNSAFE !!!', (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0,0,255), 3)

        writer.write(frame)
        cv2.resizeWindow('Counting People', w, h)
        cv2.imshow('Counting People', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done – saved to {output_path}")

if __name__ == '__main__':
    main(
      input_path   = r"C:\Users\heram\PycharmProjects\PythonProject1\Running YOLO\Saved Pictures\videoplayback.mp4",
      output_path  = 'output_improved.mp4',
      threshold    = 30,
      conf_thresh  = 0.1,
      img_size     = 1024,
      up_scale     = 1.2
    )
