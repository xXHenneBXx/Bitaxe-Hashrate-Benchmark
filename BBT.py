import os
import sys
import time
import json
import asyncio
import aiohttp
import logging
import threading
from tkinter import *
from tkinter import ttk, messagebox
from banner import banner_top, banner_bottom
from colorama import Fore, Style, init

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

init(autoreset=True)


# Custom formatter for colored logs
class ColoredLevelFormatter(logging.Formatter):
    def format(self, record):
        levelno = record.levelno
        if levelno >= logging.CRITICAL:
            color = Style.BRIGHT + Fore.MAGENTA
        elif levelno >= logging.ERROR:
            color = Style.BRIGHT + Fore.RED
        elif levelno >= logging.WARNING:
            color = Style.BRIGHT + Fore.YELLOW
        elif levelno >= logging.INFO:
            color = Style.BRIGHT + Fore.GREEN
        elif levelno >= logging.DEBUG:
            color = Fore.CYAN
        else:
            color = ''
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredLevelFormatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CYAN = Style.BRIGHT + Fore.CYAN
GREEN = Style.BRIGHT + Fore.GREEN
YELLOW = Style.BRIGHT + Fore.YELLOW
RED = Fore.RED
MAGENTA = Style.BRIGHT + Fore.MAGENTA
RESET = Style.RESET_ALL

VOLTAGE_INCREMENT = 5
FREQUENCY_INCREMENT = 25
PEAK_FREQUENCY_INCREMENT = 10
BENCHMARK_TIME_DEFAULT = 600
SAMPLE_INTERVAL = 15
MAX_TEMP_DEFAULT = 68
MIN_ALLOWED_VOLTAGE = 1000
MAX_ALLOWED_VOLTAGE = 1400
MIN_ALLOWED_FREQUENCY = 400
MAX_ALLOWED_FREQUENCY = 1300
MAX_VR_TEMP = 86
MIN_INPUT_VOLTAGE = 4600
MAX_INPUT_VOLTAGE = 5500
MAX_POWER_DEFAULT = 40

LOG_BG_WIDTH = 820
LOG_BG_HEIGHT = 520

BACKGROUND_IMAGE = 'Img/BGIMG.png'
ICON_IMAGE = 'Img/Icon.png'
SPLASH_IMAGE = 'Img/Startup.png'

class CanvasLogger(Frame):
    def __init__(self, parent, image_path=None, width=LOG_BG_WIDTH, height=LOG_BG_HEIGHT):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.canvas = Canvas(self, width=width, height=height, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollbar = ttk.Scrollbar(self, orient=VERTICAL, command=self._on_scroll)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=self._update_scrollbar)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.lines = []
        self.max_lines = 800
        self.bg_photo = None
        self.text_y = 12
        self.text_id = None
        self.offset = 0
        if image_path and os.path.exists(image_path):
            self._load_background(image_path)
        self.text_id = self.canvas.create_text(
            12, self.text_y,
            anchor=NW,
            text='',
            font=('Consolas', 10),
            fill='black',
            width=width - 28
        )
        self.canvas.bind('<Configure>', self._on_resize)

    def _load_background(self, path):
        try:
            if PIL_AVAILABLE:
                image = Image.open(path)
                image = image.resize((self.width, self.height), Image.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=NW, image=self.bg_photo, tags='bg_image')
            else:
                self.bg_photo = PhotoImage(file=path)
                self.canvas.create_image(0, 0, anchor=NW, image=self.bg_photo, tags='bg_image')
            self.canvas.lower('bg_image')
        except Exception:
            self.bg_photo = None

    def _on_resize(self, event):
        self.width = event.width
        self.height = event.height
        self.canvas.config(width=self.width, height=self.height)
        if os.path.exists(BACKGROUND_IMAGE):
            self.canvas.delete('bg_image')
            try:
                if PIL_AVAILABLE:
                    image = Image.open(BACKGROUND_IMAGE)
                    image = image.resize((self.width, self.height), Image.LANCZOS)
                    self.bg_photo = ImageTk.PhotoImage(image)
                    self.canvas.create_image(0, 0, anchor=NW, image=self.bg_photo, tags='bg_image')
                else:
                    self.bg_photo = PhotoImage(file=BACKGROUND_IMAGE)
                    self.canvas.create_image(0, 0, anchor=NW, image=self.bg_photo, tags='bg_image')
                self.canvas.lower('bg_image')
            except Exception:
                self.bg_photo = None
        self.canvas.coords(self.text_id, 12, 12 - self.offset)
        self._update_scrollregion()

    def _on_scroll(self, *args):
        if not self.text_id:
            return
        if args[0] == 'moveto':
            fraction = float(args[1])
            self._set_offset(fraction)
        elif args[0] == 'scroll':
            amount = int(args[1])
            if args[2] == 'units':
                self._set_offset(self.offset + amount * 15)
            elif args[2] == 'pages':
                self._set_offset(self.offset + amount * self.height)
        self._update_scrollbar(self.scrollbar.get())

    def _set_offset(self, value):
        bbox = self.canvas.bbox(self.text_id)
        if not bbox:
            return
        text_height = bbox[3] - bbox[1]
        max_offset = max(0, text_height - self.height + 24)
        self.offset = min(max(value, 0), max_offset)
        self.canvas.coords(self.text_id, 12, 12 - self.offset)
        self._update_scrollbar()

    def _update_scrollbar(self, *args):
        bbox = self.canvas.bbox(self.text_id)
        if not bbox:
            self.scrollbar.set(0.0, 1.0)
            return
        text_height = bbox[3] - bbox[1]
        if text_height <= self.height - 24:
            self.scrollbar.set(0.0, 1.0)
        else:
            top = self.offset / max(1, text_height - self.height + 24)
            bottom = min(1.0, top + (self.height - 24) / max(1, text_height))
            self.scrollbar.set(top, bottom)

    def _update_scrollregion(self):
        bbox = self.canvas.bbox(self.text_id)
        if bbox:
            self.canvas.configure(scrollregion=(0, 0, self.width, max(self.height, bbox[3] + 12)))
        else:
            self.canvas.configure(scrollregion=(0, 0, self.width, self.height))

    def add_line(self, message):
        self.lines.append(message)
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines:]
        self.canvas.itemconfigure(self.text_id, text='\n'.join(self.lines))
        bbox = self.canvas.bbox(self.text_id)
        if bbox:
            text_height = bbox[3] - bbox[1]
            self.offset = max(0, text_height - self.height + 24)
        else:
            self.offset = 0
        self._set_offset(self.offset)
        self._update_scrollregion()
        self._update_scrollbar()

    def clear(self):
        self.lines = []
        self.canvas.itemconfigure(self.text_id, text='')
        self.offset = 0
        self.canvas.coords(self.text_id, 12, 12)
        self._update_scrollregion()
        self._update_scrollbar()

class TkLogger(logging.Handler):
    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget
        self.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.log_widget.add_line(msg)

class SplashScreen:
    def __init__(self, root, logo_path, duration=8000):
        self.root = root
        self.duration = duration
        self.elapsed = 0
        self.step = 100
        self.window = Toplevel(root)
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.configure(bg='black')

        self.frame = Frame(self.window, bg='black', padx=16, pady=16)
        self.frame.pack(fill=BOTH, expand=True)

        self.logo_photo = None
        if os.path.exists(logo_path):
            try:
                if PIL_AVAILABLE:
                    image = Image.open(logo_path)
                    image = image.resize((320, 160), Image.LANCZOS)
                    self.logo_photo = ImageTk.PhotoImage(image)
                else:
                    self.logo_photo = PhotoImage(file=logo_path)
            except Exception:
                self.logo_photo = None

        if self.logo_photo:
            Label(self.frame, image=self.logo_photo, bg='black').pack(pady=(0, 12))

        Label(self.frame, text=' V1.0 By ~ xXHenneBXx', fg='white', bg='black', font=('Segoe UI', 14, 'bold')).pack(pady=(0, 8))
        self.progress = ttk.Progressbar(self.frame, orient=HORIZONTAL, mode='determinate', maximum=100, length=420)
        self.progress.pack(pady=(0, 8))
        Label(self.frame, text='Loading...', fg='white', bg='black', font=('Segoe UI', 11)).pack()

    def show(self):
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() - width) // 2
        y = (self.window.winfo_screenheight() - height) // 2
        self.window.geometry(f'{width}x{height}+{x}+{y}')
        self.root.withdraw()
        self.window.deiconify()
        self._tick()

    def _tick(self):
        self.elapsed += self.step
        progress = min(100, int((self.elapsed / self.duration) * 100))
        self.progress['value'] = progress
        if self.elapsed < self.duration:
            self.root.after(self.step, self._tick)
        else:
            self.window.destroy()
            self.root.deiconify()

class BenchmarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title('BBT v1.0 ~ xXHenneBXx')
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.running = False
        self.stop_requested = False
        self.benchmark_thread = None
        self.results = []
        self.default_voltage = 1150
        self.default_frequency = 550
        self.small_core_count = None
        self.asic_count = None
        self._load_icon()
        self._build_ui()
        self._setup_logger()
        self.theme = 'light'
        self.style = ttk.Style(self.root)
        self.apply_theme()
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def _load_icon(self):
        icon_path = os.path.join(self.base_dir, ICON_IMAGE)
        if os.path.exists(icon_path):
            try:
                if PIL_AVAILABLE:
                    image = Image.open(icon_path)
                    image = image.resize((64, 64), Image.LANCZOS)
                    self.icon_photo = ImageTk.PhotoImage(image)
                else:
                    self.icon_photo = PhotoImage(file=icon_path)
                self.root.iconphoto(True, self.icon_photo)
            except Exception:
                self.icon_photo = None

    def _build_ui(self):
        self.root.geometry('1180x820')
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=12, pady=12)

        left_frame = ttk.LabelFrame(main_frame, text='Configuration', padding=10)
        left_frame.pack(side=LEFT, fill=Y, padx=(0, 12), pady=2)

        ttk.Label(left_frame, text='Bitaxe IP(s):').grid(row=0, column=0, sticky=W, pady=5)
        self.ip_var = StringVar(value='192.168.0.1')
        ttk.Entry(left_frame, textvariable=self.ip_var, width=24).grid(row=0, column=1, sticky=EW, pady=5)

        ttk.Label(left_frame, text='Voltage (mV):').grid(row=1, column=0, sticky=W, pady=5)
        self.voltage_var = StringVar(value='1150')
        ttk.Entry(left_frame, textvariable=self.voltage_var, width=24).grid(row=1, column=1, sticky=EW, pady=5)

        ttk.Label(left_frame, text='Frequency (MHz):').grid(row=2, column=0, sticky=W, pady=5)
        self.frequency_var = StringVar(value='500')
        ttk.Entry(left_frame, textvariable=self.frequency_var, width=24).grid(row=2, column=1, sticky=EW, pady=5)

        ttk.Label(left_frame, text='Benchmark Mode:').grid(row=3, column=0, sticky=W, pady=5)
        self.mode_var = StringVar(value='Normal')
        self.mode_selector = ttk.Combobox(left_frame, textvariable=self.mode_var, values=['Normal', 'Peak Performance', 'Single'], state='readonly', width=22)
        self.mode_selector.grid(row=3, column=1, sticky=EW, pady=5)

        ttk.Label(left_frame, text='Benchmark Time (s):').grid(row=4, column=0, sticky=W, pady=5)
        self.bench_time_var = StringVar(value=str(BENCHMARK_TIME_DEFAULT))
        ttk.Entry(left_frame, textvariable=self.bench_time_var, width=24).grid(row=4, column=1, sticky=EW, pady=5)

        ttk.Label(left_frame, text='Max Temp (°C):').grid(row=5, column=0, sticky=W, pady=5)
        self.max_temp_var = StringVar(value=str(MAX_TEMP_DEFAULT))
        ttk.Entry(left_frame, textvariable=self.max_temp_var, width=24).grid(row=5, column=1, sticky=EW, pady=5)

        ttk.Label(left_frame, text='Max Power (W):').grid(row=6, column=0, sticky=W, pady=5)
        self.max_power_var = StringVar(value=str(MAX_POWER_DEFAULT))
        ttk.Entry(left_frame, textvariable=self.max_power_var, width=24).grid(row=6, column=1, sticky=EW, pady=5)

        self.apply_settings_only = BooleanVar(value=False)
        ttk.Checkbutton(left_frame, text='Apply settings only (no benchmark)', variable=self.apply_settings_only).grid(row=7, column=0, columnspan=2, sticky=W, pady=10)

        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=8, column=0, columnspan=2, sticky=EW, pady=12)
        self.start_btn = ttk.Button(button_frame, text='Start', command=self.start_benchmark)
        self.start_btn.pack(side=LEFT, padx=4)
        self.stop_btn = ttk.Button(button_frame, text='Stop', command=self.stop_benchmark, state=DISABLED)
        self.stop_btn.pack(side=LEFT, padx=4)
        ttk.Button(button_frame, text='Clear Log', command=self.clear_log).pack(side=LEFT, padx=4)
        self.theme_btn = ttk.Button(button_frame, text='Dark Mode', command=self.toggle_theme)
        self.theme_btn.pack(side=LEFT, padx=4)

        ttk.Label(left_frame, text='Status:').grid(row=9, column=0, sticky=W, pady=5)
        self.status_var = StringVar(value='Ready')
        ttk.Label(left_frame, textvariable=self.status_var, foreground='green').grid(row=9, column=1, sticky=W, pady=5)

        ttk.Label(left_frame, text='Usage:').grid(row=10, column=0, sticky=NW, pady=5)
        self.usage_var = StringVar(value='Normal = Default Tuning\nPeak Performance = Precise Tuning(May Take Hours)\nSingle = one pass only')
        ttk.Label(left_frame, textvariable=self.usage_var, wraplength=180, justify=LEFT).grid(row=10, column=1, sticky=W, pady=5)
        ttk.Label(left_frame, text='DISCLAIMER:').grid(row=11, column=0, sticky=W, pady=(6, 2))
        self.disclaimer_var = StringVar(value=' This tool Stress Tests your Bitaxe. Use at your own risk!!!')
        ttk.Label(left_frame, foreground='red', textvariable=self.disclaimer_var, wraplength=180, justify=LEFT).grid(row=11, column=1, sticky=W, pady=(6, 2))
        ttk.Label(left_frame, text='Authors:').grid(row=12, column=0, sticky=W, pady=(0, 6))
        self.authors_var = StringVar(value='mrv777, xXHenneBXx')
        ttk.Label(left_frame, textvariable=self.authors_var, wraplength=180, justify=LEFT).grid(row=12, column=1, sticky=W, pady=(0, 6))
        left_frame.columnconfigure(1, weight=1)

        right_frame = ttk.LabelFrame(main_frame, text='Benchmark Log', padding=10)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        self.log_canvas = CanvasLogger(right_frame, image_path=os.path.join(self.base_dir, BACKGROUND_IMAGE))
        self.log_canvas.pack(fill=BOTH, expand=True)

        results_frame = ttk.LabelFrame(self.root, text='Top 5 Results', padding=10)
        results_frame.pack(fill=BOTH, expand=False, padx=12, pady=(8, 12))

        columns = ('IP', 'Voltage', 'Frequency', 'Hashrate', 'Temp', 'Power', 'Fan', 'IntVolt')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=6)
        headings = {
            'IP': 'IP',
            'Voltage': 'Voltage (mV)',
            'Frequency': 'Frequency (MHz)',
            'Hashrate': 'Hashrate (GH/s)',
            'Temp': 'Avg Temp (°C)',
            'Power': 'Avg Power (W)',
            'Fan': 'Avg Fan (%)',
            'IntVolt': 'Avg Int Volt (mV)'
        }
        for col in columns:
            self.results_tree.heading(col, text=headings[col])
            self.results_tree.column(col, width=120, anchor=W)
        self.results_tree.column('Hashrate', width=140)
        self.results_tree.column('IntVolt', width=130)
        self.results_tree.pack(fill=BOTH, expand=True)

    def _setup_logger(self):
        handler = TkLogger(self.log_canvas)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    def toggle_theme(self):
        self.theme = 'dark' if self.theme == 'light' else 'light'
        self.apply_theme()

    def apply_theme(self):
        if self.theme == 'dark':
            bg = '#121212'
            frame_bg = '#1e1e1e'
            entry_bg = '#2b2b2b'
            fg = "#0770D3"
            button_text = 'Light Mode'
        else:
            bg = '#f2f2f2'
            frame_bg = '#f8f8f8'
            entry_bg = 'white'
            fg = 'black'
            button_text = 'Dark Mode'

        self.root.configure(bg=bg)
        self.style.configure('TFrame', background=frame_bg)
        self.style.configure('TLabelframe', background=frame_bg, foreground=fg)
        self.style.configure('TLabel', background=frame_bg, foreground=fg)
        self.style.configure('TButton', background=entry_bg, foreground=fg)
        self.style.configure('TEntry', fieldbackground=entry_bg, foreground=fg)
        self.style.configure('TCombobox', fieldbackground=entry_bg, foreground=fg)
        self.style.configure('Vertical.TScrollbar', background=frame_bg, troughcolor=bg)

        self.log_canvas.canvas.configure(bg=frame_bg)
        self.theme_btn.config(text=button_text)

        text_color = "#0FD33A" if self.theme == 'dark' else 'black'
        self.log_canvas.canvas.itemconfigure(self.log_canvas.text_id, fill=text_color)

    def start_benchmark(self):
        if self.running:
            return
        if not self.validate_inputs():
            return
        self.running = True
        self.stop_requested = False
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.status_var.set('Running')
        self.results = []
        self.update_results_table()
        self.log_canvas.add_line('Benchmark started.')
        self.benchmark_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.benchmark_thread.start()

    def stop_benchmark(self):
        if not self.running:
            return
        self.stop_requested = True
        self.status_var.set('Stopping...')
        logger.info('Stop requested by user.')

    def clear_log(self):
        self.log_canvas.clear()

    def validate_inputs(self):
        ip_text = self.ip_var.get().strip()
        if not ip_text:
            messagebox.showerror('Input Error', 'Please enter one or more Bitaxe IP addresses.')
            return False
        try:
            voltage = int(self.voltage_var.get())
            frequency = int(self.frequency_var.get())
            bench_time = int(self.bench_time_var.get())
            max_temp = int(self.max_temp_var.get())
            max_power = int(self.max_power_var.get())
        except ValueError:
            messagebox.showerror('Input Error', 'Voltage, frequency, benchmark time, max temperature, and power must be integers.')
            return False
        if voltage < MIN_ALLOWED_VOLTAGE or voltage > MAX_ALLOWED_VOLTAGE:
            messagebox.showerror('Input Error', f'Voltage must be between {MIN_ALLOWED_VOLTAGE} and {MAX_ALLOWED_VOLTAGE}.')
            return False
        if frequency < MIN_ALLOWED_FREQUENCY or frequency > MAX_ALLOWED_FREQUENCY:
            messagebox.showerror('Input Error', f'Frequency must be between {MIN_ALLOWED_FREQUENCY} and {MAX_ALLOWED_FREQUENCY}.')
            return False
        if bench_time // SAMPLE_INTERVAL < 7:
            messagebox.showerror('Input Error', 'Benchmark time is too short. Increase it or lower the sample interval.')
            return False
        if max_temp <= 0:
            messagebox.showerror('Input Error', 'Max temperature must be positive.')
            return False
        if max_power <= 0:
            messagebox.showerror('Input Error', 'Max power must be positive.')
            return False
        return True

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.benchmark_main())
        except Exception as exc:
            logger.exception(f'Benchmark loop error: {exc}')
        finally:
            loop.close()
            self.root.after(0, self._benchmark_finished)

    def _benchmark_finished(self):
        self.running = False
        self.stop_requested = False
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.status_var.set('Ready')
        self.log_canvas.add_line('Benchmark finished.')

    async def benchmark_main(self):
        ip_list = [ip.strip() for ip in self.ip_var.get().replace(',', ' ').split() if ip.strip()]
        voltage = int(self.voltage_var.get())
        frequency = int(self.frequency_var.get())
        mode_raw = self.mode_var.get().strip()
        mode = mode_raw.lower().replace(' ', '_')
        if mode not in ('normal', 'peak_performance', 'single'):
            mode = 'normal'
        bench_time = int(self.bench_time_var.get())
        max_temp = int(self.max_temp_var.get())
        max_power = int(self.max_power_var.get())
        set_only = self.apply_settings_only.get()
        self.log_canvas.add_line(f'Benchmark mode selected: {mode_raw}')
        frequency_increment = PEAK_FREQUENCY_INCREMENT if mode == 'peak_performance' else FREQUENCY_INCREMENT

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for raw_ip in ip_list:
                if self.stop_requested:
                    break
                bitaxe_ip = f'http://{raw_ip}'
                self.log_canvas.add_line(f'Starting device {raw_ip}')
                ok = await self.fetch_default_settings(bitaxe_ip, session)
                if not ok:
                    logger.error(f'Unable to read defaults for {raw_ip}. Skipping.')
                    continue
                if set_only:
                    await self.apply_settings_only_mode(bitaxe_ip, voltage, frequency, session, raw_ip)
                    continue
                current_voltage = voltage
                current_frequency = frequency
                while (current_voltage <= MAX_ALLOWED_VOLTAGE and
                       current_frequency <= MAX_ALLOWED_FREQUENCY and
                       not self.stop_requested):
                    success = await self.set_system_settings(bitaxe_ip, current_voltage, current_frequency, session, raw_ip)
                    if not success:
                        break
                    if mode == 'single' and current_voltage == voltage and current_frequency == frequency:
                        logger.info(f'{raw_ip}: running single benchmark mode at {current_voltage}mV/{current_frequency}MHz')
                    benchmark_result = await self.benchmark_iteration(bitaxe_ip, current_voltage, current_frequency, session, bench_time, max_temp, max_power)
                    if benchmark_result[0] is not None:
                        avg_hashrate, avg_temp, efficiency_jth, hashrate_ok, avg_vr_temp, avg_power, avg_fan_speed, avg_internal_voltage, error_reason = benchmark_result
                        record = {
                            'ip': raw_ip,
                            'coreVoltage': current_voltage,
                            'frequency': current_frequency,
                            'averageHashRate': avg_hashrate,
                            'averageTemperature': avg_temp,
                            'efficiencyJTH': efficiency_jth,
                            'averagePower': avg_power,
                            'errorReason': error_reason
                        }
                        if avg_vr_temp is not None:
                            record['averageVRTemp'] = avg_vr_temp
                        if avg_fan_speed is not None:
                            record['averageFanSpeed'] = avg_fan_speed
                        if avg_internal_voltage is not None:
                            record['averageInternalVoltage'] = avg_internal_voltage
                        self.results.append(record)
                        self.root.after(0, self.update_results_table)
                        if mode == 'single':
                            break
                        if mode == 'peak_performance':
                            if hashrate_ok:
                                if current_frequency + frequency_increment <= MAX_ALLOWED_FREQUENCY:
                                    current_frequency += frequency_increment
                                elif current_voltage + VOLTAGE_INCREMENT <= MAX_ALLOWED_VOLTAGE:
                                    current_voltage += VOLTAGE_INCREMENT
                                    current_frequency = max(current_frequency - frequency_increment, MIN_ALLOWED_FREQUENCY)
                                else:
                                    break
                            else:
                                if current_voltage + VOLTAGE_INCREMENT <= MAX_ALLOWED_VOLTAGE:
                                    current_voltage += VOLTAGE_INCREMENT
                                    current_frequency = max(current_frequency - frequency_increment, MIN_ALLOWED_FREQUENCY)
                                else:
                                    break
                        elif hashrate_ok:
                            if current_frequency + FREQUENCY_INCREMENT <= MAX_ALLOWED_FREQUENCY:
                                current_frequency += FREQUENCY_INCREMENT
                            else:
                                break
                        else:
                            if current_voltage + VOLTAGE_INCREMENT <= MAX_ALLOWED_VOLTAGE:
                                current_voltage += VOLTAGE_INCREMENT
                                current_frequency = max(current_frequency - FREQUENCY_INCREMENT, MIN_ALLOWED_FREQUENCY)
                            else:
                                break
                    else:
                        logger.info('Stopping device benchmark due to limit or error.')
                        break
                    if self.stop_requested:
                        break
                if self.results:
                    await self.reset_to_best_setting(raw_ip, session)
                else:
                    logger.info(f'No valid results for {raw_ip}. Applying default settings.')
                    await self.set_system_settings(bitaxe_ip, self.default_voltage, self.default_frequency, session, raw_ip)
                self.save_results(raw_ip)

    async def fetch_default_settings(self, bitaxe_ip, session):
        try:
            async with session.get(f'{bitaxe_ip}/api/system/info', timeout=10) as response:
                if response.status != 200:
                    logger.error(f'Failed default fetch {response.status} for {bitaxe_ip}')
                    return False
                data = await response.json()
                self.default_voltage = data.get('coreVoltage', self.default_voltage)
                self.default_frequency = data.get('frequency', self.default_frequency)
                self.small_core_count = data.get('smallCoreCount', 0)
                self.asic_count = data.get('asicCount', 1)
                logger.info(f'{bitaxe_ip} defaults: {self.default_voltage}mV, {self.default_frequency}MHz, cores={self.small_core_count * self.asic_count}')
                return True
        except Exception as exc:
            logger.error(f'Error fetching defaults for {bitaxe_ip}: {exc}')
            return False

    async def apply_settings_only_mode(self, bitaxe_ip, voltage, frequency, session, raw_ip):
        logger.info(f'Applying settings only to {raw_ip}')
        await self.set_system_settings(bitaxe_ip, voltage, frequency, session, raw_ip)
        self.save_results(raw_ip)

    async def get_system_info(self, bitaxe_ip, session):
        retries = 5
        for attempt in range(retries):
            if self.stop_requested:
                return None
            try:
                async with session.get(f'{bitaxe_ip}/api/system/info', timeout=10) as response:
                    if response.status == 200:
                        return await response.json()
                    logger.warning(f'System info HTTP {response.status} from {bitaxe_ip}')
            except asyncio.TimeoutError:
                logger.warning(f'Timeout fetching system info {attempt + 1}/5 from {bitaxe_ip}')
            except aiohttp.ClientError as exc:
                logger.error(f'Connection error for {bitaxe_ip}: {exc}')
            await asyncio.sleep(5)
        return None

    async def set_system_settings(self, bitaxe_ip, core_voltage, frequency, session, raw_ip):
        payload = {'coreVoltage': core_voltage, 'frequency': frequency}
        try:
            async with session.patch(f'{bitaxe_ip}/api/system', json=payload, timeout=10) as response:
                if response.status != 200:
                    logger.error(f'Failed set settings for {raw_ip}: HTTP {response.status}')
                    return False
                logger.info(f'{raw_ip}: set {core_voltage}mV @ {frequency}MHz')
                await asyncio.sleep(2)
                await self.restart_system(bitaxe_ip, session, raw_ip)
                return True
        except Exception as exc:
            logger.error(f'Set system error for {raw_ip}: {exc}')
            return False

    async def restart_system(self, bitaxe_ip, session, raw_ip):
        try:
            async with session.post(f'{bitaxe_ip}/api/system/restart', timeout=10) as response:
                if response.status != 200:
                    logger.error(f'Restart failed for {raw_ip}: HTTP {response.status}')
                    return
            logger.info(f'{raw_ip}: restart issued, waiting 90s')
            await asyncio.sleep(90)
        except Exception as exc:
            logger.error(f'Restart error for {raw_ip}: {exc}')

    async def benchmark_iteration(self, bitaxe_ip, core_voltage, frequency, session, bench_time, max_temp, max_power):
        total_samples = bench_time // SAMPLE_INTERVAL
        expected_hashrate = frequency * ((self.small_core_count or 1) * (self.asic_count or 1) / 1000)
        hash_rates = []
        temperatures = []
        power_consumptions = []
        vr_temps = []
        fan_speeds = []
        internal_voltages = []

        for sample in range(total_samples):
            if self.stop_requested:
                return None, None, None, False, None, None, None, 'STOP_REQUESTED'
            info = await self.get_system_info(bitaxe_ip, session)
            if info is None:
                logger.info(f'Failed system info for {bitaxe_ip}')
                return None, None, None, False, None, None, None, 'SYSTEM_INFO_FAILURE'

            temp = info.get('temp')
            vr_temp = info.get('vrTemp')
            voltage = info.get('voltage')
            hash_rate = info.get('hashRate')
            power_consumption = info.get('power')
            fan_speed = info.get('fanspeed')

            if temp is None or temp < 5:
                logger.warning(f'{bitaxe_ip}: temp data missing or low')
                return None, None, None, False, None, None, None, 'TEMPERATURE_DATA_FAILURE'
            if temp >= max_temp:
                logger.warning(f'{bitaxe_ip}: chip temp exceeded {max_temp}°C')
                return None, None, None, False, None, None, None, 'CHIP_TEMP_EXCEEDED'
            if vr_temp is not None and vr_temp >= MAX_VR_TEMP:
                logger.warning(f'{bitaxe_ip}: VR temp exceeded {MAX_VR_TEMP}°C')
                return None, None, None, False, None, None, None, 'VR_TEMP_EXCEEDED'
            if voltage is not None and (voltage < MIN_INPUT_VOLTAGE or voltage > MAX_INPUT_VOLTAGE):
                logger.warning(f'{bitaxe_ip}: input voltage out of range')
                return None, None, None, False, None, None, None, 'INPUT_VOLTAGE_OUT_OF_RANGE'
            if hash_rate is None or power_consumption is None:
                logger.warning(f'{bitaxe_ip}: hashrate or power data missing')
                return None, None, None, False, None, None, None, 'DATA_MISSING'
            if power_consumption > max_power:
                logger.warning(f'{bitaxe_ip}: power exceeded {max_power}W')
                return None, None, None, False, None, None, None, 'POWER_EXCEEDED'

            hash_rates.append(hash_rate)
            temperatures.append(temp)
            power_consumptions.append(power_consumption)
            if voltage is not None:
                internal_voltages.append(voltage)
            if vr_temp is not None and vr_temp > 0:
                vr_temps.append(vr_temp)
            if fan_speed is not None:
                fan_speeds.append(fan_speed)

            percentage = ((sample + 1) / total_samples) * 100
            logger.info(
                f'[{sample + 1}/{total_samples}] {percentage:5.1f}% | CV:{core_voltage}mV | F:{frequency}MHz | H:{int(hash_rate)} GH/s | T:{int(temp)}°C | P:{int(power_consumption)}W'
                + (f' | VR:{int(vr_temp)}°C' if vr_temp is not None else '')
                + (f' | FAN:{int(fan_speed)}%' if fan_speed is not None else '')
            )
            if sample < total_samples - 1:
                await asyncio.sleep(SAMPLE_INTERVAL)

        if not hash_rates or not temperatures or not power_consumptions:
            return None, None, None, False, None, None, None, 'NO_DATA'

        sorted_hashrates = sorted(hash_rates)
        trimmed_hashrates = sorted_hashrates[3:-3] if len(sorted_hashrates) > 6 else sorted_hashrates
        average_hashrate = sum(trimmed_hashrates) / len(trimmed_hashrates)
        sorted_temps = sorted(temperatures)
        trimmed_temps = sorted_temps[6:] if len(sorted_temps) > 6 else sorted_temps
        average_temperature = sum(trimmed_temps) / len(trimmed_temps)
        average_vr_temp = None
        if vr_temps:
            sorted_vr = sorted(vr_temps)
            trimmed_vr = sorted_vr[6:] if len(sorted_vr) > 6 else sorted_vr
            average_vr_temp = sum(trimmed_vr) / len(trimmed_vr)
        average_power = sum(power_consumptions) / len(power_consumptions)
        average_fan_speed = sum(fan_speeds) / len(fan_speeds) if fan_speeds else None
        average_internal_voltage = sum(internal_voltages) / len(internal_voltages) if internal_voltages else None
        efficiency_jth = average_power / (average_hashrate / 1000) if average_hashrate > 0 else 0
        hashrate_ok = average_hashrate >= expected_hashrate * 0.94
        logger.info(f'{bitaxe_ip}: avg {average_hashrate:.2f} GH/s, temp {average_temperature:.2f}°C, eff {efficiency_jth:.2f} J/TH')
        return average_hashrate, average_temperature, efficiency_jth, hashrate_ok, average_vr_temp, average_power, average_fan_speed, average_internal_voltage, None

    async def reset_to_best_setting(self, raw_ip, session):
        bitaxe_ip = f'http://{raw_ip}'
        ip_results = [r for r in self.results if r.get('ip') == raw_ip]
        if not ip_results:
            logger.info(f'{raw_ip}: no results, applying defaults')
            await self.set_system_settings(bitaxe_ip, self.default_voltage, self.default_frequency, session, raw_ip)
        else:
            best = max(ip_results, key=lambda r: r.get('averageHashRate', 0))
            await self.set_system_settings(bitaxe_ip, best['coreVoltage'], best['frequency'], session, raw_ip)

    def save_results(self, raw_ip):
        if not self.results:
            return
        first_ip = raw_ip or self.results[0].get('ip', 'multi')
        filename = f'Benchmark@{first_ip}.json'
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=4)
            logger.info(f'Results saved to {filename}')
        except Exception as exc:
            logger.error(f'Failed saving results: {exc}')

    def update_results_table(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        top5 = sorted(self.results, key=lambda r: r.get('averageHashRate', 0), reverse=True)[:5]
        for row in top5:
            self.results_tree.insert('', END, values=(
                row.get('ip', ''),
                row.get('coreVoltage', ''),
                row.get('frequency', ''),
                f"{row.get('averageHashRate', 0):.2f}",
                f"{row.get('averageTemperature', 0):.2f}",
                f"{row.get('averagePower', 0):.2f}",
                f"{row.get('averageFanSpeed', 0):.2f}" if row.get('averageFanSpeed') is not None else '',
                f"{row.get('averageInternalVoltage', 0):.2f}" if row.get('averageInternalVoltage') is not None else ''
            ))

    def on_close(self):
        if self.running:
            if messagebox.askyesno('Exit', 'Benchmark is running. Stop and exit?'):
                self.stop_requested = True
                self.root.after(100, self._wait_close)
        else:
            self.root.destroy()

    def _wait_close(self):
        if self.benchmark_thread and self.benchmark_thread.is_alive():
            self.root.after(100, self._wait_close)
        else:
            self.root.destroy()


def main():
    root = Tk()
    print(Fore.CYAN + Style.BRIGHT + banner_top + RESET)
    print(Fore.GREEN + banner_bottom + RESET)
    app = BenchmarkApp(root)
    splash = SplashScreen(root, os.path.join(app.base_dir, SPLASH_IMAGE), duration=8000)
    splash.show()
    root.mainloop()

if __name__ == '__main__':
    main()
