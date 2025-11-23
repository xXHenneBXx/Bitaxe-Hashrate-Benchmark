import asyncio
import aiohttp
import json
import signal
import sys
import argparse
import logging
import colorama
from banner import banner_top, banner_bottom
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Custom formatter for colored logs
class ColoredLevelFormatter(logging.Formatter):
    def format(self, record):
        levelno = record.levelno
        if levelno >= logging.CRITICAL:
            color = Fore.MAGENTA
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
        
#Colorama Colors
CYAN = Style.BRIGHT + Fore.CYAN
GREEN = Style.BRIGHT + Fore.GREEN
YELLOW = Style.BRIGHT + Fore.YELLOW
RED = Fore.RED
MAGENTA = Style.BRIGHT + Fore.MAGENTA
RESET = Style.RESET_ALL

# Logo
print(Fore.CYAN + Style.BRIGHT + banner_top + RESET)
print(Fore.GREEN + banner_bottom + RESET)

# Setup logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredLevelFormatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.setLevel(logging.DEBUG)

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

# Arguments and Examples
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f"{YELLOW}Bitaxe Benchmark Tool{RESET}\n"
                    "This script allows you to either benchmark your Bitaxe miner across various "
                    "voltage and frequency settings, or apply specific settings directly.\n",
        epilog=f"{GREEN}Examples:{RESET}\n"
               f"  {GREEN}1. Full Benchmark (Default at 1100mV, 500MHz ~ 1300mV, 900MHz(MAX set in configs)):{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.*{RESET} default settings only\n\n"
               f"  {GREEN}2. User Specific Settings eg; (Default at 1100mV, 500MHz ~ 1300mV, 900MHz(MAX set in configs)):{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* -v 1150 -f 780{RESET}\n\n"
               f"  {GREEN}3. Mode Specific Settings eg;{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* -v 1150 -f 600 --mode single {RESET} only one iteration of the applied settings\n\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* -v 1150 -f 600 --mode normal {RESET} default increment settings applied\n\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* -v 1150 -f 600 --mode hybrid {RESET} hybrid mode is faster than normal with larger increments\n\n"
               f"  {GREEN}4.Device Specific Settings eg; (1150mV, 780MHz) and exit:{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} 192.168.0.* --set-values -v 1150 -f 780{RESET}\n\n"
               f"  {GREEN}5.Display This Help Message:{RESET}\n"
               f"     {YELLOW}python {sys.argv[0]} --help{RESET}",
        formatter_class=RawTextAndDefaultsHelpFormatter
    )
    
    # Arguments
    parser.add_argument('bitaxe_ips', nargs='+', help=f"{YELLOW}IP of your Bitaxe miner (e.g., 192.168.0.26)\n  Required for benchmarking and setting.{RESET}")
    parser.add_argument('-v', '--voltage', type=int, default=1100, help=f"{YELLOW}Set Core Voltage in mV.{RESET}")
    parser.add_argument('-f', '--frequency', type=int, default=500, help=f"{YELLOW}Set Core Frequency in MHz.{RESET}")
    parser.add_argument('-s', '--set-values', action='store_true', help=f"{YELLOW}Set values to Bitaxe only; does NOT run Benchmark.{RESET}")
    parser.add_argument('-m', '--mode', choices=['single','normal','hybrid'], default='normal', help=f"{YELLOW}Benchmark mode.{RESET}")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

# Configuration
args = parse_arguments()
voltage_increment = 5              # 5 mV incremtal for fine tuning
hybrid_voltage_increment = 10      # 10 mV increments for hybrid mode
frequency_increment = 5            # 5 Mhz incremntal changes for fine tuning
hybrid_frequency_increment = 15    # 15 Mhz increments for hybrid mode
benchmark_time = 600               # 10 minutes benchmark time
sample_interval = 15               # 15 seconds sample interval
max_temp = 68                      # Will stop if temperature reaches or exceeds this value
min_allowed_voltage = 1060         # Minimum allowed core voltage
max_allowed_voltage = 1300         # Maximum allowed core voltage
min_allowed_frequency = 500        # Minimum allowed frequency
max_allowed_frequency = 1200       # Maximum allowed core frequency
max_vr_temp = 86                   # Maximum allowed voltage regulator temperature
min_input_voltage = 4650           # Minimum allowed input voltage - Internal Voltage Below will start to fail
max_input_voltage = 5500           # Maximum allowed input voltage - ***DO NOT INCREASE YOU WILL BURN OUT YOUR DEVICE***
max_power = 60                     # Max of 30W because of DC plug ~ Increase if you have an upgraded PSU

# --- Globals ---
bitaxe_ips = [f"http://{ip}" for ip in args.bitaxe_ips]
initial_voltage = args.voltage
initial_frequency = args.frequency
# Dynamically determined default settings
asic_count = None
default_voltage = None
default_frequency = None
small_core_count = None
handling_interrupt = False
system_reset_done = False
# Session and event
session = None                    # Global variable to store the HTTP session
shutdown_event = asyncio.Event()  # Async event for graceful shutdown

# --- Generate list of voltages and frequencies based on mode ---
voltages_list = []
frequencies_list = []

if args.mode == 'single':
    voltages_list = [initial_voltage]
    frequencies_list = [initial_frequency]
elif args.mode == 'normal':
    voltages_list = list(range(initial_voltage, max_allowed_voltage + 1, voltage_increment))
    frequencies_list = list(range(initial_frequency, max_allowed_frequency + 1, frequency_increment))
elif args.mode == 'hybrid':
    voltages_list = list(range(initial_voltage, max_allowed_voltage + 1, hybrid_voltage_increment))
    frequencies_list = list(range(initial_frequency, max_allowed_frequency + 1, hybrid_frequency_increment)) 
else:
    # fallback to single if unknown
    voltages_list = [initial_voltage]
    frequencies_list = [initial_frequency]

# Validate initial API inputs
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

# Results storage
if len(bitaxe_ips) == 1:
    results = []  # Single device - simple list
else:
    device_results = {}  # Multiple devices - dictionary
    for ip in bitaxe_ips:
        device_results[ip] = []  # Separate tracking per device

# --- Async functions ---
async def fetch_default_settings(session):
    global default_voltage, default_frequency, small_core_count, asic_count
    # Take the first IP for single device mode
    url = f"{bitaxe_ips[0]}/api/system/info"
    for attempt in range(5):
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
                logger.info(f"Last Best Known settings: Voltage={default_voltage}mV, Freq={default_frequency}MHz, Total Cores={small_core_count * asic_count}")
                return
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching system info. Attempt {attempt+1}")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching system info: {e}")
        await asyncio.sleep(5)
    logger.error("Failed to fetch system info after retries.")
    sys.exit(1)

async def set_system_settings(session, core_voltage, frequency):
    # Use the first IP for single device mode
    url = f"{bitaxe_ips[0]}/api/system"
    payload = {
        "coreVoltage": core_voltage,
        "frequency": frequency
    }
    for attempt in range(5):
        try:
            async with session.patch(url, json=payload, timeout=15) as resp:
                resp.raise_for_status()
                logger.info(f"Starting.. Voltage= {core_voltage}mV, Freq= {frequency}MHz")
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
    # Use the first IP for single device mode
    url = f"{bitaxe_ips[0]}/api/system/restart"
    try:
        if not handling_interrupt:
            logger.info("Waiting 90s for device restart...")
            async with session.post(url, timeout=15) as resp:
                resp.raise_for_status()
            await asyncio.sleep(100)
        else:
            logger.info("Best settings applied")
            async with session.post(url, timeout=15) as resp:
                resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Restart failed: {e}")

async def get_system_info(session):
    # Use the first IP for single device mode
    url = f"{bitaxe_ips[0]}/api/system/info"
    for attempt in range(5):
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
    expected_hashrate = frequency * ((small_core_count * asic_count) / 1000) # Calculate expected hashrate based on frequency
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

        temp = info.get("temp") # Get ASIC temperature if available
        vr_temp = info.get("vrTemp") # Get VR temperature if available
        voltage = info.get("voltage") # Get Internal Voltage if available
        hash_rate = info.get("hashRate") # Get Hahs Rate if available
        power_consumption = info.get("power") # Get Power Consumption if available
        fan_speed = info.get("fanspeed") # Get Fan Speed if available

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

        # Only sleep if it's not the last iteration
        if sample < total_samples - 1:
            await asyncio.sleep(sample_interval)

    # Process results
    if hash_rates and temperatures and power_consumptions:
        sorted_hashrates = sorted(hash_rates)
        trimmed_hashrates = sorted_hashrates[3:-3] # Remove first 3 and last 3 elements
        avg_hashrate = sum(trimmed_hashrates) / len(trimmed_hashrates)

        # Sort and trim temperatures (remove lowest 6 readings during warmup)
        sorted_temps = sorted(temperatures)
        trimmed_temps = sorted_temps[6:] # Remove first 6 elements only
        avg_temp = sum(trimmed_temps) / len(trimmed_temps)

        # Only process VR temps if we have valid readings
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

# Main async flow
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

        # Disclaimer
        logger.warning(RED + "DISCLAIMER: This program Benchmarks your BITAXE Device. ENSURE YOU HAVE PROPER COOLING TO YOUR BOARD AND PSU... USE AT OWN RISK!!!." + RESET)
        logger.warning(RED + "While safeguards are in place, running hardware outside of standard parameters carries inherent risks" + RESET)
        logger.warning(RED + "THE AUTHORS ARE 'NOT' RESPONSIBLE FOR ANY DAMAGE TO YOUR DEVICE" + RESET)
        
        print(MAGENTA + f"\nNOTE: Ambient temperature significantly affects these results. The optimal settings found may not work well if room temperature changes substantially. Re-run the benchmark if conditions change.\n" + RESET)

        current_voltage = initial_voltage
        current_frequency = initial_frequency

        # Main benchmarking loop with shutdown check
        while (
            current_voltage <= max_allowed_voltage and
            current_frequency <= max_allowed_frequency and
            not shutdown_event.is_set()
        ):
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
                        logger.info(f"Hashrate low, decreasing freq to {current_frequency}MHz, increasing voltage to {current_voltage}mV")
                    else:
                        break
            else:
                logger.info("Stopped due to Thermal or Settings limit issue, Reconfigure.")
                break
        await cleanup_and_exit(session)

        # Save results
        await save_results()

        # Results
        if results:
            top_results = sorted(results, key=lambda x: x['averageHashRate'], reverse=True)[:5]
            top_efficient_results = sorted(results, key=lambda x: x["efficiencyJTH"], reverse=False)[:5]
            print("\nTop 5 results by Hashrate:")
            for res in top_results + top_efficient_results[:5]:
                print(f"Voltage: {res['coreVoltage']}mV, Freq: {res['frequency']}MHz, Hashrate: {res['averageHashRate']} GH/s")
        # Reset to best setting
        await reset_to_best_setting(session)

# --- Save results ---
async def save_results():
    ip_addr = args.bitaxe_ips[0]
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

# --- Cleanup ---
async def cleanup_and_exit(reason=None):
    global system_reset_done
    if system_reset_done:
        return
    try:
        if results:
            await reset_to_best_setting(session)
            await save_results()
            print(Fore.GREEN + "Bitaxe reset to best settings and results saved." + RESET)
        else:
            print(Fore.MAGENTA + "No valid benchmarking results found. Applying default settings." + RESET)
            await set_system_settings(session, default_voltage, default_frequency)
            await reset_to_best_setting(session)
            await save_results()
    finally:
        system_reset_done = True
        if reason:
            print(Fore.RED + f"Benchmarking stopped: {reason}" + RESET)
        print(Fore.CYAN + "Benchmarking completed/interrupted. Finishing Iteration, Please Wait Until Completed..." + RESET)
        # Set shutdown event to exit main loop
        shutdown_event.set()

# --- Signal handler ---
def handle_sigint(signum, frame):
    global handling_interrupt, session, system_reset_done
    if handling_interrupt or system_reset_done:
        return
    handling_interrupt = True
    # Schedule cleanup
    asyncio.create_task(cleanup_and_exit("SIGINT received"))
    logger.info("Interrupted! Resetting system.")

signal.signal(signal.SIGINT, handle_sigint)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Benchmarking interrupted by user")
        shutdown_event.set()
