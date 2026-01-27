import cv2
import mediapipe as mp
import time
import math
import os
import pygame
import numpy as np  # layar hitam
from pathlib import Path


def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def main():
    # Layar belah boy
    TIMER_DURATION = 0.1
    HEAD_PITCH_THRESHOLD = 2.0
    GAZE_DOWN_THRESHOLD = 0.55
    EYE_CLOSED_THRESHOLD = 0.15

    # Setup assets
    current_dir = Path(__file__).parent
    assets_dir = current_dir / "assets"
    video_path = (assets_dir / "sahurVid.mp4").resolve()
    audio_path = (assets_dir / "sahur.mp3").resolve()

    # Cek Video
    if not video_path.exists():
        print(f"ERROR: Video tidak ketemu di {video_path}")
        return

    # Setup Audio
    has_audio = False
    if audio_path.exists():
        pygame.mixer.init()
        try:
            pygame.mixer.music.load(str(audio_path))
            has_audio = True
            print("AUDIO: Siap (skyrim-skeleton.mp3 loaded)")
        except Exception as e:
            print(f"AUDIO ERROR: {e}")
    else:
        print(f"WARNING: File audio tidak ditemukan di {audio_path}")

    # Setup Video Player
    skyrim_cap = cv2.VideoCapture(str(video_path))

    # Setup AI & Kamera
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    cam = cv2.VideoCapture(0)
    # Paksa resolusi 640x480 (Standar)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    doomscroll_start_time = None
    is_playing_video = False

    print("(SPLIT SCREEN)===")

    while True:
        # 1.Pantau webcam
        ret_cam, cam_frame = cam.read()
        if not ret_cam:
            break

        cam_frame = cv2.flip(cam_frame, 1)
        h, w, _ = cam_frame.shape

        # Proses AI di Webcam mu
        rgb_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status_text = "FOKUS"
        color_status = (0, 255, 0)

        # Deteksi Logika
        is_sleeping = False
        is_head_down = False
        is_gazing_down = False
        head_ratio = 0.0

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Hitung parameter
            forehead = landmarks[10]
            nose = landmarks[4]
            chin = landmarks[152]
            head_ratio = calculate_distance(nose, forehead) / (
                calculate_distance(nose, chin) + 1e-6
            )

            v_eye = calculate_distance(landmarks[159], landmarks[145])
            h_eye = calculate_distance(landmarks[33], landmarks[133])
            ear = v_eye / (h_eye + 1e-6)

            l_iris_y = landmarks[468].y
            top_y = landmarks[159].y
            bot_y = landmarks[145].y
            gaze_ratio = (l_iris_y - top_y) / (bot_y - top_y + 1e-6)

            # Cek Status
            is_sleeping = ear < EYE_CLOSED_THRESHOLD
            is_head_down = head_ratio > HEAD_PITCH_THRESHOLD
            is_gazing_down = gaze_ratio > GAZE_DOWN_THRESHOLD

            # Visualisasi Head Ratio
            cv2.putText(
                cam_frame,
                f"Head: {head_ratio:.2f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # 2.LOGIKA VIDEO & AUDIO
        if is_playing_video:
            # SYARAT STOP: Kepala Tegak & Mata Melek
            if (not is_sleeping) and (head_ratio < HEAD_PITCH_THRESHOLD - 0.1):
                print("FOKUS KEMBALI -> STOP!")
                is_playing_video = False
                if has_audio:
                    pygame.mixer.music.stop()
                skyrim_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                doomscroll_start_time = None
            else:
                cv2.putText(
                    cam_frame,
                    "TERDETEKSI!",
                    (30, 120),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

        else:
            # SYARAT START
            if is_sleeping or is_head_down or is_gazing_down:
                if doomscroll_start_time is None:
                    doomscroll_start_time = time.time()

                elapsed = time.time() - doomscroll_start_time
                if is_sleeping:
                    status_text = "TIDUR!"
                elif is_head_down:
                    status_text = "NUNDUK!"
                else:
                    status_text = "LIRIK!"
                color_status = (0, 0, 255)

                if elapsed >= TIMER_DURATION:
                    is_playing_video = True
                    if has_audio:
                        pygame.mixer.music.play(-1)
            else:
                doomscroll_start_time = None
                status_text = "FOKUS"

        # Tampilkan Status di Webcam
        cv2.putText(
            cam_frame,
            status_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color_status,
            3,
        )

        # 3.SIAPKAN FRAME KANAN (VIDEO / HITAM)
        if is_playing_video:
            ret_vid, video_frame = skyrim_cap.read()
            if not ret_vid:
                skyrim_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_vid, video_frame = skyrim_cap.read()

            # Resize video biar sama tingginya dengan webcam (480px)
            video_frame = cv2.resize(video_frame, (w, h))
        else:
            # Kalau gak main video, tampilkan layar hitam kosong
            video_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                video_frame,
                "Pantau aja dulu..",
                (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (100, 100, 100),
                2,
            )

        # 4. GABUNGKAN KIRI (WEBCAM) & KANAN (VIDEO)
        # cv2.hconcat = Horizontal Concatenation (Tempel Samping)
        final_display = cv2.hconcat([cam_frame, video_frame])

        cv2.imshow("Doomscroll Detector V7 (Split Screen)", final_display)
        if cv2.waitKey(1) == 27:
            break

    if has_audio:
        pygame.mixer.quit()
    skyrim_cap.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
