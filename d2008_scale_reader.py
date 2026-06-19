"""
D2008 Electronic Weighing Indicator - Data Reader
==================================================
Giao thức: TF=0 (Continuous mode)
Cổng:      RS232 (9600 baud, 8N1)

Cài đặt thư viện cần thiết:
    pip install pyserial

Cấu hình trên cân D2008:
    - Vào Setting > PSt 01
    - Đặt bt = 9600 (hoặc tốc độ bạn chọn)
    - Đặt TF = 0
    - Đặt Jn = nonE (không parity)
"""

import serial
import time
import sqlite3
import threading
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable


# ─────────────────────────────────────────────
#  CẤU HÌNH — chỉnh sửa theo thực tế
# ─────────────────────────────────────────────
SERIAL_PORT   = "/dev/ttyS6"       # Windows: "COM3", Linux: "/dev/ttyUSB0"
BAUD_RATE     = 9600
DB_FILE       = "scale_data.db"
SERIAL_DUMP_FILE = None
LOG_INTERVAL  = 1.0          # Ghi DB mỗi N giây (0 = ghi mọi frame)

# Stability detection
STABLE_COUNT     = 5         # Number of consecutive readings to check
STABLE_TOLERANCE = 0.5       # kg — max spread to be considered stable
# ─────────────────────────────────────────────


@dataclass
class WeightFrame:
    """Dữ liệu một frame cân D2008 (TF=0, 12 byte)"""
    raw_bytes:    bytes
    sign:         str        # '+' hoặc '-'
    weight:       float      # Giá trị cân (đã có dấu thập phân)
    decimal_pos:  int        # Số chữ số thập phân (0~4)
    checksum_ok:  bool
    digits_str:   str        # Raw 6-digit string (for overload detection)
    status:       str = "UNSTABLE"  # STABLE / UNSTABLE / OVERLOAD
    timestamp:    datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        return self.checksum_ok and self.status != "OVERLOAD"

    def __str__(self):
        sign_str = "-" if self.sign == '-' else ""
        return (f"[{self.timestamp.strftime('%H:%M:%S.%f')[:12]}] "
                f"Weight: {sign_str}{self.weight:.{self.decimal_pos}f} kg  "
                f"| {self.status:<10} "
                f"| Checksum: {'OK' if self.checksum_ok else 'FAIL'}")


class D2008Parser:
    """
    Parser cho giao thức TF=0 của D2008

    Cấu trúc frame 12 byte:
      Byte 1:  0x02 (XON - start)
      Byte 2:  0x2B(+) hoặc 0x2D(-) (dấu)
      Byte 3-8: 6 chữ số ASCII (giá trị cân, không có dấu thập phân)
      Byte 9:  Vị trí thập phân (0x30=0, 0x31=1, ... 0x34=4)
      Byte 10: Verify high nibble (ASCII)
      Byte 11: Verify low nibble  (ASCII)
      Byte 12: 0x03 (XOFF - end)

    Verify = XOR(byte2, byte3, ..., byte9)
      - Nếu nibble <= 9: gửi nibble + 0x30
      - Nếu nibble >= A: gửi nibble + 0x37
    """

    FRAME_SIZE  = 12
    START_BYTE  = 0x02
    END_BYTE    = 0x03

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes) -> list[WeightFrame]:
        """Nạp bytes mới, trả về danh sách các frame hoàn chỉnh đã parse."""
        self._buffer.extend(data)
        frames = []

        while len(self._buffer) >= self.FRAME_SIZE:
            # Tìm byte START
            start_idx = self._buffer.find(self.START_BYTE)
            if start_idx == -1:
                self._buffer.clear()
                break

            # Bỏ rác trước start byte
            if start_idx > 0:
                self._buffer = self._buffer[start_idx:]

            # Đủ dữ liệu cho 1 frame chưa?
            if len(self._buffer) < self.FRAME_SIZE:
                break

            # Kiểm tra END byte đúng vị trí
            candidate = self._buffer[:self.FRAME_SIZE]
            if candidate[-1] != self.END_BYTE:
                # Frame lỗi, bỏ START byte, tìm tiếp
                self._buffer = self._buffer[1:]
                continue

            frame = self._parse_frame(bytes(candidate))
            if frame:
                frames.append(frame)

            self._buffer = self._buffer[self.FRAME_SIZE:]

        return frames

    def _parse_frame(self, raw: bytes) -> Optional[WeightFrame]:
        """Parse 12 byte thành WeightFrame."""
        try:
            # Byte 2: dấu
            sign = '+' if raw[1] == 0x2B else '-'

            # Byte 3-8: 6 chữ số ASCII (không thập phân)
            digits_str = raw[2:8].decode('ascii')
            if not digits_str.isdigit():
                return None
            raw_value = int(digits_str)

            # Byte 9: vị trí thập phân
            decimal_pos = raw[8] - 0x30
            if not (0 <= decimal_pos <= 4):
                return None
            weight = raw_value / (10 ** decimal_pos)

            # Byte 10-11: verify
            checksum_ok = self._verify(raw)

            return WeightFrame(
                raw_bytes=raw,
                sign=sign,
                weight=weight if sign == '+' else -weight,
                decimal_pos=decimal_pos,
                checksum_ok=checksum_ok,
                digits_str=digits_str,
            )
        except Exception:
            return None

    @staticmethod
    def _verify(raw: bytes) -> bool:
        """Kiểm tra XOR checksum theo tài liệu D2008."""
        # XOR byte 2..9 (index 1..8)
        xor_val = 0
        for b in raw[1:9]:
            xor_val ^= b

        high_nibble = (xor_val >> 4) & 0x0F
        low_nibble  = xor_val & 0x0F

        # Encode nibble thành ASCII
        def encode_nibble(n):
            return n + 0x30 if n <= 9 else n + 0x37

        expected_high = encode_nibble(high_nibble)
        expected_low  = encode_nibble(low_nibble)

        return raw[9] == expected_high and raw[10] == expected_low


class ScaleDatabase:
    """Lưu dữ liệu cân vào SQLite."""

    def __init__(self, db_file: str):
        self.db_file = db_file
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS weight_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    weight_kg   REAL    NOT NULL,
                    sign        TEXT    NOT NULL,
                    decimal_pos INTEGER NOT NULL,
                    checksum_ok INTEGER NOT NULL,
                    status      TEXT    NOT NULL DEFAULT 'UNSTABLE'
                )
            """)
            self._conn.commit()
        print(f"[DB] Database sẵn sàng: {self.db_file}")

    def save(self, frame: WeightFrame):
        with self._lock:
            self._conn.execute("""
                INSERT INTO weight_log
                    (timestamp, weight_kg, sign, decimal_pos, checksum_ok, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                frame.timestamp.isoformat(),
                frame.weight,
                frame.sign,
                frame.decimal_pos,
                int(frame.checksum_ok),
                frame.status,
            ))
            self._conn.commit()

    def get_recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            old_row_factory = self._conn.row_factory
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute("""
                SELECT * FROM weight_log
                ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
            self._conn.row_factory = old_row_factory
        return [dict(r) for r in rows]

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


class D2008Reader:
    """
    Đọc dữ liệu từ cân D2008 qua RS232.

    Ví dụ sử dụng:
        reader = D2008Reader(port="COM3", baud=9600)
        reader.on_weight = lambda f: print(f)
        reader.start()
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = BAUD_RATE,
        db_file: str = DB_FILE,
        dump_file: Optional[str] = SERIAL_DUMP_FILE,
        log_interval: float = LOG_INTERVAL,
    ):
        self.port         = port
        self.baud         = baud
        self.dump_file    = dump_file
        self.log_interval = log_interval

        self._parser   = D2008Parser()
        self._db       = ScaleDatabase(db_file)
        self._serial   = None
        self._running  = False
        self._thread   = None
        self._last_log   = 0.0
        self._last_print = 0.0
        self._recent_weights = deque(maxlen=STABLE_COUNT)
        self._prev_status = "UNSTABLE"

        # Callback — fires every ~1s for logging/display
        self.on_weight: Optional[Callable[[WeightFrame], None]] = None
        # Callback — fires on every frame for session logic
        self.on_frame: Optional[Callable[[WeightFrame], None]] = None
        # Callback — fires only on status transitions (STABLE↔UNSTABLE, OVERLOAD)
        self.on_status_change: Optional[Callable[[WeightFrame, str, str], None]] = None

        # Giá trị cân mới nhất (thread-safe read)
        self.latest: Optional[WeightFrame] = None

    def start(self):
        """Bắt đầu đọc (non-blocking, chạy background thread)."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[READER] Đang kết nối {self.port} @ {self.baud} baud...")

    def stop(self):
        """Dừng đọc."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._db.close()
        print("[READER] Đã dừng.")

    def _run(self):
        try:
            if self.dump_file:
                self._run_from_dump_file(self.dump_file)
                return

            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=2,
            )
            print(f"[READER] Kết nối thành công: {self.port}")

            while self._running:
                raw = self._serial.read(self._serial.in_waiting or 1)
                if not raw:
                    continue

                frames = self._parser.feed(raw)
                for frame in frames:
                    self._handle_frame(frame)

        except (serial.SerialException, TypeError, OSError) as e:
            if self._running:
                print(f"[ERROR] Lỗi cổng serial: {e}")
        finally:
            if self._serial and self._serial.is_open:
                self._serial.close()

    def _run_from_dump_file(self, file_path: str):
        """Giả lập đọc serial bằng dữ liệu dump từ file nhị phân."""
        print(f"[READER] Đang đọc dữ liệu giả lập từ file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                dump_data = f.read()
        except FileNotFoundError:
            print(f"[ERROR] Không tìm thấy file dump: {file_path}")
            return

        if not dump_data:
            print("[ERROR] File dump rỗng, không có dữ liệu để đọc.")
            return

        print(f"[READER] Nạp thành công {len(dump_data)} bytes dữ liệu dump.")

        chunk_size = 64
        while self._running:
            for idx in range(0, len(dump_data), chunk_size):
                if not self._running:
                    break

                raw = dump_data[idx: idx + chunk_size]
                frames = self._parser.feed(raw)
                for frame in frames:
                    self._handle_frame(frame)

                time.sleep(0.02)

    def _get_status(self, frame: WeightFrame) -> str:
        """Determine scale status: OVERLOAD, STABLE, or UNSTABLE."""
        if frame.digits_str == "999999":
            return "OVERLOAD"

        self._recent_weights.append(frame.weight)

        if len(self._recent_weights) >= STABLE_COUNT:
            w_min = min(self._recent_weights)
            w_max = max(self._recent_weights)
            if (w_max - w_min) <= STABLE_TOLERANCE:
                return "STABLE"

        return "UNSTABLE"

    def _handle_frame(self, frame: WeightFrame):
        """Xử lý mỗi frame nhận được."""
        frame.status = self._get_status(frame)
        self.latest = frame

        # Fire on every frame for session logic (stable counting etc.)
        if self.on_frame:
            self.on_frame(frame)

        # Detect status transitions — fires on every change
        if frame.status != self._prev_status:
            old_status = self._prev_status
            self._prev_status = frame.status
            if self.on_status_change:
                self.on_status_change(frame, old_status, frame.status)

        now = time.time()

        # Throttled callback for logging/display (~1s)
        if now - self._last_print >= 1.0:
            if self.on_weight:
                self.on_weight(frame)
            self._last_print = now

        # Ghi DB theo interval
        if frame.checksum_ok and (now - self._last_log >= self.log_interval):
            self._db.save(frame)
            self._last_log = now


# ─────────────────────────────────────────────
#  DEMO — chạy trực tiếp
# ─────────────────────────────────────────────
def demo_console():
    """Demo đơn giản: in dữ liệu ra console và lưu DB."""

    reader = D2008Reader(
        port=SERIAL_PORT,
        baud=BAUD_RATE,
        db_file=DB_FILE,
        dump_file=SERIAL_DUMP_FILE,
        log_interval=LOG_INTERVAL,
    )

    # Gán callback in ra console
    def on_weight(frame: WeightFrame):
        ck = "OK" if frame.checksum_ok else "FAIL"
        print(f"  {frame.status:<10} {frame.weight:>10.{frame.decimal_pos}f} kg   "
              f"[{frame.timestamp.strftime('%H:%M:%S')}]  cksum:{ck}")

    reader.on_weight = on_weight
    reader.start()

    print("\n=== D2008 Scale Reader ===")
    print(f"Stability: {STABLE_COUNT} readings within {STABLE_TOLERANCE} kg")
    print("Nhấn Ctrl+C để dừng\n")
    print(f"{'Status':<10} {'Weight':>12}     {'Time'}       {'Checksum'}")
    print("-" * 55)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        reader.stop()
        print("\n--- 20 bản ghi gần nhất trong DB ---")
        db = ScaleDatabase(DB_FILE)
        for row in db.get_recent(20):
            status = row.get('status', '?')
            print(f"  [{row['timestamp']}]  {row['weight_kg']:>10.3f} kg  {status}")


# ─────────────────────────────────────────────
#  TEST PARSER (không cần cân thật)
# ─────────────────────────────────────────────
def test_parser():
    """
    Test parser với dữ liệu giả lập.
    Theo tài liệu: truyền 20.00 → frame:
      02 2B 30 30 32 30 30 30 32 31 1B 03
    """
    print("=== Test Parser (không cần cân thật) ===\n")

    # Tạo frame test theo đúng tài liệu (giá trị 20.00 kg)
    def make_frame(value_str: str, decimal: int, sign: str = '+') -> bytes:
        """Tạo frame hợp lệ để test."""
        sign_byte = 0x2B if sign == '+' else 0x2D
        digits = value_str.zfill(6).encode('ascii')
        dec_byte = 0x30 + decimal

        payload = bytes([sign_byte]) + digits + bytes([dec_byte])

        # Tính XOR verify
        xor_val = 0
        for b in payload:
            xor_val ^= b
        h = (xor_val >> 4) & 0x0F
        l = xor_val & 0x0F
        encode = lambda n: n + 0x30 if n <= 9 else n + 0x37

        return bytes([0x02]) + payload + bytes([encode(h), encode(l), 0x03])

    parser = D2008Parser()

    tests = [
        ("002000", 2, '+'),   # 20.00 kg
        ("050000", 0, '+'),   # 50000 kg
        ("001500", 3, '+'),   # 1.500 kg
        ("000000", 2, '+'),   # 0.00 kg (zero)
        ("010230", 1, '-'),   # -1023.0 kg (âm)
        ("999999", 0, '+'),   # OVERLOAD
    ]

    for val, dec, sign in tests:
        frame_bytes = make_frame(val, dec, sign)
        frames = parser.feed(frame_bytes)
        if frames:
            f = frames[0]
            print(f"  Input: {val} dec={dec} sign={sign}")
            print(f"  Output: weight={f.weight} | digits={f.digits_str} | checksum={'OK' if f.checksum_ok else 'FAIL'}")
            print()

    print("Test hoàn thành!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_parser()
    else:
        demo_console()
