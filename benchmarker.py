import asyncio
import aiohttp
import time
import json
import signal
import sys
import argparse
import logging
import colorama
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Custom formatter for colored logs
class ColoredLevelFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        msg = record.getMessage()

        color = ''
        if record.levelno >= logging.CRITICAL:
            color = Fore.MAGENTA
        elif record.levelno >= logging.ERROR:
            color = Style.BRIGHT + Fore.RED
        elif record.levelno >= logging.WARNING:
            color = Style.BRIGHT + Fore.YELLOW
        elif record.levelno >= logging.INFO:
            color = Style.BRIGHT + Fore.GREEN
        elif record.levelno >= logging.DEBUG:
            color = Fore.CYAN

        # Compose the message
        message = super().format(record)
        # Colorize entire message
        return f"{color}{message}{Style.RESET_ALL}"

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredLevelFormatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.setLevel(logging.DEBUG)

# Colorama Colors
CYAN = Style.BRIGHT + Fore.CYAN
MAGENTA = Style.BRIGHT + Fore.MAGENTA
GREEN = Style.BRIGHT + Fore.GREEN
YELLOW = Style.BRIGHT + Fore.YELLOW
RED = Style.BRIGHT + Fore.RED
RESET = Style.RESET_ALL

# Banner Title
banner_top = r"""
            ░████████   ░██████░██████████   ░███    ░██    ░██ ░██████████                ░██     ░██    ░███      ░██████   ░██     ░██ 
            ░██    ░██    ░██      ░██      ░██░██    ░██  ░██  ░██                        ░██     ░██   ░██░██    ░██   ░██  ░██     ░██ 
            ░██    ░██    ░██      ░██     ░██  ░██    ░██░██   ░██                        ░██     ░██  ░██  ░██  ░██         ░██     ░██ 
            ░████████     ░██      ░██    ░█████████    ░███    ░█████████                 ░██████████ ░█████████  ░████████  ░██████████ 
            ░██     ░██   ░██      ░██    ░██    ░██   ░██░██   ░██                        ░██     ░██ ░██    ░██         ░██ ░██     ░██ 
            ░██     ░██   ░██      ░██    ░██    ░██  ░██  ░██  ░██                        ░██     ░██ ░██    ░██  ░██   ░██  ░██     ░██ 
            ░█████████  ░██████    ░██    ░██    ░██ ░██    ░██ ░██████████                ░██     ░██ ░██    ░██   ░██████   ░██     ░██ 
"""
banner_bottom = r"""
                        ░████████   ░██████████ ░███    ░██   ░██████  ░██     ░██ ░███     ░███    ░███    ░█████████  ░██     ░██ ░██████████ ░█████████  
                        ░██    ░██  ░██         ░████   ░██  ░██   ░██ ░██     ░██ ░████   ░████   ░██░██   ░██     ░██ ░██    ░██  ░██         ░██     ░██ 
                        ░██    ░██  ░██         ░██░██  ░██ ░██        ░██     ░██ ░██░██ ░██░██  ░██  ░██  ░██     ░██ ░██   ░██   ░██         ░██     ░██ 
                        ░████████   ░█████████  ░██ ░██ ░██ ░██        ░██████████ ░██ ░████ ░██ ░█████████ ░█████████  ░███████    ░█████████  ░█████████  
                        ░██     ░██ ░██         ░██  ░██░██ ░██        ░██     ░██ ░██  ░██  ░██ ░██    ░██ ░██   ░██   ░██   ░██   ░██         ░██   ░██   
                        ░██     ░██ ░██         ░██   ░████  ░██   ░██ ░██     ░██ ░██       ░██ ░██    ░██ ░██    ░██  ░██    ░██  ░██         ░██    ░██  
                        ░█████████  ░██████████ ░██    ░███   ░██████  ░██     ░██ ░██       ░██ ░██    ░██ ░██     ░██ ░██     ░██ ░██████████ ░██     ░██ 
"""

print(Fore.CYAN + Style.BRIGHT + banner_top + RESET)
print(Fore.GREEN + banner_bottom + RESET)

# Help Formatter
class RawTextAndDefaultsHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_text = super()._get_help_string(action)
        if action.default is not argparse.SUPPRESS:
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                if "\n" in help_text:
                    help_text += f"{GREEN}\n(default: {action.default}{RESET})"
                else:
                    help_text += f"{GREEN} (default: {action.default}{RESET})"
        return help_text
# Help Arguments and Examples
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"{YELLOW}Bitaxe Benchmark Tool{RESET}\n"
                    "This script allows you to either benchmark your Bitaxe miner across various "
                    "voltage and frequency settings, or apply specific settings directly.\n",
        epilog=f"{GREEN}Examples:{RESET}\n"
               f"  {GREEN}1. Full Benchmark (Default at 1150mV, 500MHz ~ 1300mV, 900MHz(MAX set in configs)):{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* -v 1150 -f 500{RESET}\n\n"
               f"  {GREEN}2. Specific Settings eg; (1150mV, 780MHz) and exit:{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* --set-values -v 1150 -f 780{RESET}\n\n"
               f"  {GREEN}3.Display This Help Message:{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} --help{RESET}",
        formatter_class=RawTextAndDefaultsHelpFormatter
    )

    parser.add_argument('-i', '--bitaxe_ip', required=True, help=f"{YELLOW}IP of your Bitaxe miner (e.g., 192.168.0.26)\n  Required for benchmarking and setting.{RESET}")
    parser.add_argument('-v', '--voltage', type=int, default=1150, help=f"{YELLOW}Set Core Voltage in mV.{RESET}")
    parser.add_argument('-f', '--frequency', type=int, default=500, help=f"{YELLOW}Set Core Frequency in MHz.{RESET}")
    parser.add_argument('-s', '--set-values', action='store_true', help=f"{YELLOW}Set values to Bitaxe only; does NOT run Benchmark.{RESET}")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()



# Configuration
args = parse_arguments()
voltage_increment = 15
frequency_increment = 20
benchmark_time = 600
sample_interval = 15
max_temp = 68
min_allowed_voltage = 1060
max_allowed_voltage = 1300
min_allowed_frequency = 500
max_allowed_frequency = 1300
max_vr_temp = 86
min_input_voltage = 4650
max_input_voltage = 5500
max_power = 30
bitaxe_ip = f"http://{args.bitaxe_ip}"
initial_voltage = args.voltage
initial_frequency = args.frequency

small_core_count = None
asic_count = None

# Validate API inputs
if initial_voltage > max_allowed_voltage:
    raise ValueError(RED + f"Error: Initial voltage exceeds max {max_allowed_voltage}mV." + RESET)
if initial_voltage < min_allowed_voltage:
    raise ValueError(RED + f"Error: Initial voltage below min {min_allowed_voltage}mV." + RESET)
if initial_frequency > max_allowed_frequency:
    raise ValueError(RED + f"Error: Initial frequency exceeds max {max_allowed_frequency}MHz." + RESET)
if initial_frequency < min_allowed_frequency:
    raise ValueError(RED + f"Error: Initial frequency below min {min_allowed_frequency}MHz." + RESET)
if benchmark_time / sample_interval < 7:
    raise ValueError(RED + "Benchmark time too short." + RESET)

# Results
results = []

# Defaults
#default_voltage = 1100
#default_frequency = 500
system_reset_done = False
handling_interrupt = False

# Global session variable
session = None
# Create an asyncio.Event for shutdown signaling
shutdown_event = asyncio.Event()

# --- Async functions ---
async def fetch_default_settings(session):
    global default_voltage, default_frequency, small_core_count, asic_count
    url = f"{bitaxe_ip}/api/system/info"
    retries = 5
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=15) as resp:
                resp.raise_for_status()
                info = await resp.json()
                default_voltage = info.get("coreVoltage", 1100)
                if "smallCoreCount" not in info:
                    logger.error("Missing smallCoreCount in system info.")
                    sys.exit(1)
                default_frequency = info.get("frequency", 500)
                small_core_count = info.get("smallCoreCount", 0)
                asic_count = info.get("asicCount", 1)
                logger.info(f"User settings: Voltage={default_voltage}mV, Freq={default_frequency}MHz, Total Cores={small_core_count * asic_count}")
                return
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching system info. Attempt {attempt+1}")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching system info: {e}")
        await asyncio.sleep(5)
    logger.error("Failed to fetch system info after retries.")
    sys.exit(1)

async def set_system_settings(session, core_voltage, frequency):
    url = f"{bitaxe_ip}/api/system"
    payload = {
        "coreVoltage": core_voltage,
        "frequency": frequency
    }
    retries = 5
    for attempt in range(retries):
        try:
            async with session.patch(url, json=payload, timeout=15) as resp:
                resp.raise_for_status()
                logger.info(f"Set Voltage= {core_voltage}mV, Freq= {frequency}MHz")
                await asyncio.sleep(2)
                await restart_system(session)
                return
        except asyncio.TimeoutError:
            logger.warning(f"Timeout setting system. Attempt {attempt+1}")
        except aiohttp.ClientError as e:
            logger.error(f"Error setting system: {e}")
        await asyncio.sleep(3)
    logger.error("Failed to set system after retries.")

async def restart_system(session):
    global handling_interrupt
    url = f"{bitaxe_ip}/api/system/restart"
    try:
        if not handling_interrupt:
            logger.info("Applying new settings, waiting 90s for device restart...")
            async with session.post(url, timeout=15) as resp:
                resp.raise_for_status()
            await asyncio.sleep(100)
        else:
            logger.info("Applying final settings...")
            async with session.post(url, timeout=15) as resp:
                resp.raise_for_status()
    except asyncio.TimeoutError:
        logger.warning("Timeout during restart.")
    except aiohttp.ClientError as e:
        logger.error(f"Error during restart: {e}")

async def get_system_info(session):
    url = f"{bitaxe_ip}/api/system/info"
    retries = 5
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching system info. Attempt {attempt+1}")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching system info: {e}")
            break
        await asyncio.sleep(5)
    return None

async def benchmark_iteration(session, core_voltage, frequency):
    total_samples = benchmark_time // sample_interval
    expected_hashrate = frequency * ((small_core_count * asic_count) / 1000)
    hash_rates = []
    temperatures = []
    power_consumptions = []
    vr_temps = []
    fan_speeds = []

    for sample in range(total_samples):
        info = await get_system_info(session)
        if info is None:
            logger.error("Failed to fetch system info.")
            return None, None, None, False, None, None, None, "SYSTEM_INFO_FAILURE"

        temp = info.get("temp")
        vr_temp = info.get("vrTemp")
        voltage = info.get("voltage")
        hash_rate = info.get("hashRate")
        power_consumption = info.get("power")
        fan_speed = info.get("fanspeed")

        # Check limits
        if temp is None:
            logger.warning("Temperature data not available.")
            return None, None, None, False, None, None, None, "TEMPERATURE_DATA_FAILURE"
        if temp < 5:
            logger.warning("Temperature below 5°C.")
            return None, None, None, False, None, None, None, "TEMPERATURE_BELOW_5"
        if temp >= max_temp:
            logger.warning("Chip temp exceeded.")
            return None, None, None, False, None, None, None, "CHIP_TEMP_EXCEEDED"
        if vr_temp is not None and vr_temp >= max_vr_temp:
            logger.warning("VR temp exceeded.")
            return None, None, None, False, None, None, None, "VR_TEMP_EXCEEDED"
        if voltage is not None:
            if voltage < min_input_voltage:
                logger.warning("Input voltage below min.")
                return None, None, None, False, None, None, None, "INPUT_VOLTAGE_BELOW_MIN"
            if voltage > max_input_voltage:
                logger.warning("Input voltage above max.")
                return None, None, None, False, None, None, None, "INPUT_VOLTAGE_ABOVE_MAX"
        if hash_rate is None or power_consumption is None:
            logger.warning("Hashrate or power data missing.")
            return None, None, None, False, None, None, None, "HASHRATE_POWER_DATA_FAILURE"
        if power_consumption > max_power:
            logger.warning("Power exceeded.")
            return None, None, None, False, None, None, None, "POWER_CONSUMPTION_EXCEEDED"

        hash_rates.append(hash_rate)
        temperatures.append(temp)
        power_consumptions.append(power_consumption)
        if vr_temp is not None:
            vr_temps.append(vr_temp)
        if fan_speed is not None:
            fan_speeds.append(fan_speed)

        # Progress update
        percentage_progress = ((sample + 1) / total_samples) * 100
        logger.info(
            f"{CYAN}[{sample+1:2d}/{total_samples:2d}] {percentage_progress:5.1f}% | "
            f"CV:{core_voltage:4d}mV | FRQ:{frequency:4d}MHz | "
            f"HR:{int(hash_rate):4d} GH/s | IV:{int(voltage):4d}mV | TMP:{int(temp):2d}°C"
            + (f" | VR:{int(vr_temp):2d}°C" if vr_temp is not None and vr_temp > 0 else "")
            + (f" | PWR:{int(power_consumption):2d}W" if power_consumption is not None else "")
            + (f" | FAN:{int(fan_speed):2d}%" if fan_speed is not None else "")
        )

        if sample < total_samples - 1:
            await asyncio.sleep(sample_interval)

    # Process results
    if hash_rates and temperatures and power_consumptions:
        sorted_hashrates = sorted(hash_rates)
        trimmed_hashrates = sorted_hashrates[3:-3]
        avg_hashrate = sum(trimmed_hashrates) / len(trimmed_hashrates)

        sorted_temps = sorted(temperatures)
        trimmed_temps = sorted_temps[6:]
        avg_temp = sum(trimmed_temps) / len(trimmed_temps)

        avg_vr_temp = None
        if vr_temps:
            sorted_vr = sorted(vr_temps)
            trimmed_vr = sorted_vr[6:]
            avg_vr_temp = sum(trimmed_vr) / len(trimmed_vr)

        avg_power = sum(power_consumptions) / len(power_consumptions)

        avg_fan_speed = None
        if fan_speeds:
            avg_fan_speed = sum(fan_speeds) / len(fan_speeds)
            logger.info(f"{YELLOW}Avg Fan Speed: {avg_fan_speed:.2f}%{RESET}")

        if avg_hashrate > 0:
            efficiency = avg_power / (avg_hashrate / 1000)
        else:
            logger.warning("Zero Hashrate detected.")
            return None, None, None, False, None, None, None, "ZERO_HASHRATE"

        hashrate_ok = (avg_hashrate >= expected_hashrate * 0.94)

        logger.info(f"{YELLOW}Avg Hashrate: {avg_hashrate:.2f} GH/s (Expected: {expected_hashrate:.2f}){RESET}")
        logger.info(f"{YELLOW}Avg Temp: {avg_temp:.2f}°C{RESET}")
        if avg_vr_temp is not None:
            logger.info(f"{YELLOW}Avg VR Temp: {avg_vr_temp:.2f}°C{RESET}")
        logger.info(f"{YELLOW}Efficiency: {efficiency:.2f} J/TH{RESET}")

        return avg_hashrate, avg_temp, efficiency, hashrate_ok, avg_vr_temp, avg_power, avg_fan_speed, None
    else:
        logger.warning("Insufficient data collected.")
        return None, None, None, False, None, None, None, "NO_DATA"

# --- Main async flow ---
async def main():
    global session, results, system_reset_done
    async with aiohttp.ClientSession() as session:
        # Fetch default info
        await fetch_default_settings(session)

        # User only wants to set values
        if args.set_values:
            logger.info("Applying settings only...")
            await set_system_settings(session, initial_voltage, initial_frequency)
            logger.info("Settings applied. Check web interface.")
            return

        # Warn user
        logger.warning(RED + "DISCLAIMER: This program Benchmarks your BITAXE Device. ENSURE YOU HAVE PROPER COOLING TO YOUR BOARD AND PSU... USE AT OWN RISK!!!." + RESET)
        logger.warning(RED + "While safeguards are in place, running hardware outside of standard parameters carries inherent risks" + RESET)
        logger.warning(RED + "THE AUTHORS(s) ARE 'NOT' RESPONSIBLE FOR ANY DAMAGE TO YOUR DEVICE" + RESET)
        print(MAGENTA + "\nNOTE: Ambient temperature significantly affects these results. The optimal settings found may not work well if room temperature changes substantially. Re-run the benchmark if conditions change.\n" + RESET)

        current_voltage = initial_voltage
        current_frequency = initial_frequency

        while current_voltage <= max_allowed_voltage and current_frequency <= max_allowed_frequency:
            await set_system_settings(session, current_voltage, current_frequency)
            results_data = await benchmark_iteration(session, current_voltage, current_frequency)

            if results_data[0] is not None:
                (avg_hashrate, avg_temp, efficiency, hashrate_ok, avg_vr_temp, avg_power, avg_fan_speed, error_reason) = results_data
                result_entry = {
                    "coreVoltage": current_voltage,
                    "frequency": current_frequency,
                    "averageHashRate": avg_hashrate,
                    "averageTemperature": avg_temp,
                    "efficiencyJTH": efficiency,
                    "averagePower": avg_power,
                    "errorReason": error_reason
                }
                if avg_vr_temp is not None:
                    result_entry["averageVRTemp"] = avg_vr_temp
                if avg_fan_speed is not None:
                    result_entry["averageFanSpeed"] = avg_fan_speed
                results.append(result_entry)

                if hashrate_ok:
                    if current_frequency + frequency_increment <= max_allowed_frequency:
                        current_frequency += frequency_increment
                    else:
                        break
                else:
                    if current_voltage + voltage_increment <= max_allowed_voltage:
                        current_voltage += voltage_increment
                        current_frequency -= frequency_increment
                        logger.info(f"Hashrate low, decreasing frequency to {current_frequency}MHz, increasing voltage to {current_voltage}mV")
                    else:
                        break
            else:
                logger.info("Stopped due to Thermal or Settings limit issue, Reconfigure.")
                break

        # Save results
        await save_results()
        # Reset to best setting
        await reset_to_best_setting(session)

# --- Save results ---
async def save_results():
    ip_addr = args.bitaxe_ip
    filename = f"Benchmark@{ip_addr}.json"
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

# --- Reset to best setting ---
async def reset_to_best_setting(session):
    if not results:
        logger.info("No results, applying default settings.")
        await set_system_settings(session, default_voltage, default_frequency)
    else:
        best = max(results, key=lambda x: x["averageHashRate"])
        logger.info(f"Applying best settings: Voltage={best['coreVoltage']}mV, Freq={best['frequency']}MHz")
        await set_system_settings(session, best['coreVoltage'], best['frequency'])
    await restart_system(session)

# --- Restart system ---
async def restart_system(session):
    global handling_interrupt
    url = f"{bitaxe_ip}/api/system/restart"
    try:
        if not handling_interrupt:
            logger.info("Applying new settings, waiting for device restart...")
            async with session.post(url, timeout=15) as resp:
                resp.raise_for_status()
            await asyncio.sleep(100)
        else:
            logger.info("Applying final settings...")
            async with session.post(url, timeout=15) as resp:
                resp.raise_for_status()
    except asyncio.TimeoutError:
        logger.warning("Timeout during restart.")
    except aiohttp.ClientError as e:
        logger.error(f"Error during restart: {e}")

# --- Signal handler ---
def handle_sigint(signum, frame):
    global handling_interrupt, session, system_reset_done
    if handling_interrupt or system_reset_done:
        return
    handling_interrupt = True
    logger.info("Interrupted! Resetting system.")
    # Schedule cleanup
    asyncio.create_task(cleanup_and_exit(session, "SIGINT received"))

# --- Cleanup ---
async def cleanup_and_exit(session, reason=None):
    global system_reset_done
    if system_reset_done:
        return
    try:
        if results:
            await reset_to_best_setting(session)
            await save_results()
            logger.info(CYAN + "Bitaxe reset to best settings and results saved." + RESET)
        else:
            logger.info(GREEN + "No valid benchmarking results found. Applying predefined default settings." + RESET)
            await set_system_settings(session, default_voltage, default_frequency)
    finally:
        system_reset_done = True
        if reason:
            logger.error(RED + f"Benchmarking stopped: {reason}" + RESET)
        logger.info(CYAN + "Benchmarking completed." + RESET)
        # Instead of raising SystemExit, set shutdown event
        shutdown_event.set()

# Setup event loop with exception handler
def handle_exception(loop, context):
    exception = context.get('exception')
    if isinstance(exception, SystemExit):
        return
   
   # fallback handler
    loop.default_exception_handler(context)

if __name__ == '__main__':
    try:
        loop = asyncio.new_event_loop()
        loop.set_exception_handler(handle_exception)
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Shutdown requested by user.")
    finally:
        loop.close()
