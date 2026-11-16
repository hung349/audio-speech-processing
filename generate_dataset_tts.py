"""Generate the Lab 2 word WAVs with Vietnamese neural TTS (no microphone).

Requires internet access to Microsoft Edge TTS.
Install: pip install edge-tts imageio-ffmpeg
Run:     python generate_dataset_tts.py --out dataset --repetitions 5
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from edge_tts.exceptions import NoAudioReceived

FS = 16_000
VOICE = "vi-VN-HoaiMyNeural"  # Vietnamese female voice; change to vi-VN-NamMinhNeural if desired
FALLBACK_VOICE = "vi-VN-NamMinhNeural"
WORDS = [("khong", "không"), ("mot", "một"), ("hai", "hai"), ("ba", "ba"), ("bon", "bốn")]
# Small speaking-rate and pitch changes give each utterance slight variation.
VARIATIONS = [(-8, -1), (-4, 0), (0, 0), (4, 1), (8, 0)]


async def synthesize(text: str, mp3_path: Path, rate: int, pitch: int) -> None:
    # The online Edge TTS endpoint can intermittently return an empty stream.
    # Retry with neutral prosody, then try another Vietnamese neural voice.
    attempts = [
        (VOICE, rate, pitch),
        (VOICE, 0, 0),
        (FALLBACK_VOICE, 0, 0),
    ]
    errors = []
    for voice, attempt_rate, attempt_pitch in attempts:
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=f"{attempt_rate:+d}%",
                pitch=f"{attempt_pitch:+d}Hz",
            )
            await communicate.save(str(mp3_path))
            if voice != VOICE:
                print(f"  Dùng giọng dự phòng {voice} cho mẫu này.")
            return
        except Exception as exc:
            errors.append(f"{voice} ({attempt_rate:+d}%, {attempt_pitch:+d}Hz): {exc}")
            mp3_path.unlink(missing_ok=True)
    raise RuntimeError(
        "Dịch vụ TTS không trả âm thanh sau 3 lần thử. Kiểm tra kết nối Internet/VPN "
        "hoặc thử lại sau. Lỗi: " + " | ".join(errors)
    )


async def main_async(args) -> None:
    if args.repetitions < 5:
        raise SystemExit("Lab yêu cầu tối thiểu 5 mẫu mỗi từ; --repetitions phải >= 5")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lab2_tts_") as temp:
        temp_dir = Path(temp)
        for label, word in WORDS:
            folder = args.out / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(args.repetitions):
                rate, pitch = VARIATIONS[i % len(VARIATIONS)]
                mp3_path = temp_dir / f"{label}_{i + 1:02d}.mp3"
                wav_path = folder / f"{label}_{i + 1:02d}.wav"
                print(f"Đang tạo {wav_path} — giọng={VOICE}, rate={rate:+d}%, pitch={pitch:+d}Hz")
                await synthesize(word, mp3_path, rate, pitch)
                # Convert to mono 16 kHz PCM and add 300 ms silence at both ends.
                subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3_path),
                    "-af", "adelay=300|300,apad=pad_dur=0.3",
                    "-ac", "1", "-ar", str(FS), "-c:a", "pcm_s16le", str(wav_path),
                ], check=True)
    print(f"\nĐã tạo xong dataset tại: {args.out.resolve()}")
    print("Dùng với pipeline: python lab2_speech_dtw.py --data", args.out, "--out results")
    print("Lưu ý: đây là dữ liệu tổng hợp từ một giọng TTS; kết quả không đại diện cho giọng người thật.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
