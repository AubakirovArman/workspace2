# testinference — поэтапный стенд инференса

Цель: запускать каждый шаг пайплайна lip-sync отдельно, чтобы точечно мерить и ускорять.

Структура шагов:
- extract_mel.py — извлечение mel-спектрограммы из аудио (torchaudio).
- chunk_mel.py — разбиение mel на окна (`mel_step_size`) по FPS.
- load_frames.py — загрузка кадров аватара (видео/изображение), resize/crop/rotate.
- detect_faces.py — батчевая детекция лиц и координаты (SFD через face_detection).
- run_wav2lip.py — инференс Wav2Lip: подготовка батчей, forward, предсказанные патчи.
- paste_patches.py — вставка патчей в соответствующие регионы кадров.
- encode_video.py — запись кадров → ffmpeg pipe, кодек автоподбор.

Минимальный формат данных между шагами:
- mel.npy — массив формы (n_mels, n_frames)
- mel_chunks.npy — список/массив чанков формы (n_mels, mel_step_size)
- frames_dir/ — PNG кадры исходного видео (или один кадр для статики)
- coords.npy — массив координат [(y1,y2,x1,x2), ...] на каждый кадр
- patches_dir/ — PNG патчи от Wav2Lip на каждый кадр

Примеры запуска:
1) Извлечь mel: `python3 testinference/extract_mel.py --audio input.wav --out mel.npy`
2) Разбить на чанки: `python3 testinference/chunk_mel.py --mel mel.npy --fps 25 --step 16 --out mel_chunks.npy`
3) Загрузить кадры: `python3 testinference/load_frames.py --face avatar.mp4 --fps 25 --out frames_dir`
4) Детекция лиц: `python3 testinference/detect_faces.py --frames frames_dir --pads 0 10 0 0 --out coords.npy`
5) Инференс Wav2Lip: `python3 testinference/run_wav2lip.py --checkpoint /workspace/Wav2Lip-SD-GAN.pt --frames frames_dir --mel_chunks mel_chunks.npy --coords coords.npy --out patches_dir`
6) Вставить патчи: `python3 testinference/paste_patches.py --frames frames_dir --patches patches_dir --coords coords.npy --out merged_frames_dir`
7) Закодировать: `python3 testinference/encode_video.py --frames merged_frames_dir --audio input.wav --fps 25 --out output.mp4`

Замечания:
- Все шаги фиксируют время выполнения и печатают метрики.
- Кодеки: сначала пробуем NVENC, при недоступности — libx264 с yuv420p и even dimensions.
- Любой шаг можно заменить/ускорить независимо (например, оптимизация записи кадров).

