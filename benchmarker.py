import asyncio
import aiohttp
import time
import json
import signal
import sys
import argparse
import colorama
import logging
from banner import banner_top, banner_bottom
from colorama import Fore, Style, init

# Preserve original init
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

#Colorama Colors
CYAN = Style.BRIGHT + Fore.CYAN
GREEN = Style.BRIGHT + Fore.GREEN
YELLOW = Style.BRIGHT + Fore.YELLOW
RED = Fore.RED
MAGENTA = Style.BRIGHT + Fore.MAGENTA
RESET = Style.RESET_ALL

# Banner
print(Fore.CYAN + Style.BRIGHT + banner_top + RESET)
print(Fore.GREEN + banner_bottom + RESET)

# This formatter allows for multi-line descriptions in help messages and adds default values
class RawTextAndDefaultsHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_text = super()._get_help_string(action)
        if action.default is not argparse.SUPPRESS:
            # Append default value to help text if available
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                if "\n" in help_text:
                    help_text += f"\n(default: {action.default})"
                else:
                    help_text += f" (default: {action.default})"
        return help_text

# Modify the parse_arguments function
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=
        f"{CYAN}Bitaxe Benchmark Tool{RESET}\n"
        "This script allows you to either benchmark your Bitaxe miner across various "
        "voltage and frequency settings, or apply specific settings directly.\n",
        epilog=
        f"{GREEN}Examples:{RESET}\n"
        f"  {GREEN}1. Run a full benchmark (starting at 1150mV, 500MHz):{RESET}\n"
        f"     {CYAN}python bitaxe_hasrate_benchmark_async.py 192.168.1.136{RESET}\n\n"
        f"  {GREEN}2. Apply specific settings (1150mV, 780MHz) and exit:{RESET}\n"
        f"     {CYAN}python bitaxe_hasrate_benchmark_async.py 192.168.1.136 --set-values -v 1150 -f 780{RESET}\n\n"
        f"  {GREEN}3. Benchmark multiple devices:{RESET}\n"
        f"     {CYAN}python bitaxe_hasrate_benchmark_async.py 192.168.1.136 192.168.1.137 -v 1150 -f 500{RESET}\n\n"
        f"  {GREEN}4. Get help (this message):{RESET}\n"
        f"     {CYAN}python bitaxe_hasrate_benchmark_async.py --help{RESET}",
        formatter_class=RawTextAndDefaultsHelpFormatter # <--- USE THE CUSTOM FORMATTER
    )

    # Positional and Optional Arguments (now supports multiple IPs)
    parser.add_argument('bitaxe_ips', nargs='*', help=f"{GREEN}IP address(es) miner(s) (e.g., 192.168.2.26){RESET}\n")
    parser.add_argument('-v', '--voltage', type=int, default=1150, help=f"{GREEN}Core voltage in mV.{RESET}\n")
    parser.add_argument('-f', '--frequency', type=int, default=500, help=f"{GREEN}Core frequency in MHz.{RESET}\n")
    parser.add_argument('-s', '--set-values', action='store_true', help=f"{GREEN}Apply Defaults or Set settings and Exit. No Benchmark!{RESET}\n")

    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Ensure at least one IP provided unless user only asked for help
#    if not parsed.bitaxe_ips:
#        parser.print_help()
#        sys.exit(1)

    return  parser.parse_args()

# Configuration
args = parse_arguments()
voltage_increment = 15
frequency_increment = 20
benchmark_time = 600          # 10 minutes benchmark time
sample_interval = 15          # 15 seconds sample interval
max_temp = 68                 # Will stop if temperature reaches or exceeds this value
min_allowed_voltage = 1000   # Minimum allowed core voltage
max_allowed_voltage = 1400    # Maximum allowed core voltage
min_allowed_frequency = 400   # Minimum allowed frequency
max_allowed_frequency = 1300  # Maximum allowed core frequency
max_vr_temp = 86              # Maximum allowed voltage regulator temperature
min_input_voltage = 4600      # Minimum allowed input voltage
max_input_voltage = 5500      # Maximum allowed input voltage
max_power = 40                # Max of 30W because of DC plug ~ Increase if you have an upgraded PSU

# bitaxe_ip variable used per-device now
initial_voltage = args.voltage
initial_frequency = args.frequency

# Add these variables to the global configuration section
small_core_count = None
asic_count = None

# Validate max Core Voltage
if initial_voltage > max_allowed_voltage:
    raise ValueError(RED + f"Error: Initial voltage exceeds the maximum allowed value of {max_allowed_voltage}mV. Please check the input and try again." + RESET)
# Validate max Frequency
if initial_frequency > max_allowed_frequency:
    raise ValueError(RED + f"Error: Initial frequency exceeds the maximum allowed value of {max_allowed_frequency}Mhz. Please check the input and try again." + RESET)
# Validate minimum Core Voltage
if initial_voltage < min_allowed_voltage:
    raise ValueError(RED + f"Error: Initial voltage is below the minimum allowed value of {min_allowed_voltage}mV." + RESET)
# Validate minimal frequency
if initial_frequency < min_allowed_frequency:
    raise ValueError(RED + f"Error: Initial frequency is below the minimum allowed value of {min_allowed_frequency}MHz." + RESET)
# Validate benchmark time
if benchmark_time / sample_interval < 7:
    raise ValueError(RED + f"Error: Benchmark time is too short. Please increase the benchmark time or decrease the sample interval. At least 7 samples are required." + RESET)

# Results storage (keep name; append results with "ip" entry)
results = []

# Dynamically determined default settings
default_voltage = 1150
default_frequency = 550

# Check if we're handling an interrupt (Ctrl+C)
handling_interrupt = False

# Add a global flag to track whether the system has already been reset
system_reset_done = False

# We will track running tasks to cancel them on signal
running_tasks = []

# --- Get Info from ASIC and Configs---

async def fetch_default_settings(bitaxe_ip, session):
    """
    Fetch /api/system/info and set default_voltage, default_frequency,
    small_core_count, asic_count (global vars) based on device info.
    """
    global default_voltage, default_frequency, small_core_count, asic_count
    try:
        async with session.get(f"{bitaxe_ip}/api/system/info", timeout=10) as response:
            if response.status != 200:
                logger.error(f"Error fetching default system settings from {bitaxe_ip}: HTTP {response.status}. Using fallback defaults.")
                default_voltage = 1100
                default_frequency = 500
                return False
            system_info = await response.json()
            default_voltage = system_info.get("coreVoltage", 1100)  # Fallback to 1150 if not found
            # Always get small_core_count from /system/info since it's always available there
            if "smallCoreCount" not in system_info:
                logger.error("Error: smallCoreCount field missing from /api/system/info response.")
                logger.critical("Cannot proceed without core count information for hashrate calculations.")
                return False
            default_frequency = system_info.get("frequency", 500)  # Fallback to 500 if not found
            small_core_count = system_info.get("smallCoreCount", 0)
            asic_count = system_info.get("asicCount", 1)
            logger.info(f"Current settings determined for {bitaxe_ip}:\n"
                          f"  Core Voltage: {default_voltage}mV\n"
                          f"  Frequency: {default_frequency}MHz\n"
                          f"  ASIC Configuration: {small_core_count * asic_count} total cores")
            return True
    except asyncio.TimeoutError:
        logger.error(f"Timeout while fetching default system settings from {bitaxe_ip}. Using fallback defaults.")
        default_voltage = 1100
        default_frequency = 500
        return False
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching default system settings from {bitaxe_ip}: {e}. Using fallback defaults.")
        default_voltage = 1100
        default_frequency = 500
        return False

async def get_system_info(bitaxe_ip, session):
    retries = 5 
    for attempt in range(retries):
        try:
            async with session.get(f"{bitaxe_ip}/api/system/info", timeout=10) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} while fetching system info from {bitaxe_ip}. Attempt {attempt + 1} of {retries}.")
                else:
                    return await response.json()
        except asyncio.TimeoutError:
            logger.info(f"Timeout while fetching system info from {bitaxe_ip}. Attempt {attempt + 1} of {retries}.")
        except aiohttp.ClientConnectionError:
            logger.error(f"Connection error while fetching system info from {bitaxe_ip}. Attempt {attempt + 1} of {retries}.")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching system info from {bitaxe_ip}: {e}")
            break
        await asyncio.sleep(5)  # Wait before retrying
    return None

async def set_system_settings(bitaxe_ip, core_voltage, frequency, session):
    settings = {
        "coreVoltage": core_voltage,
        "frequency": frequency
    }
    try:
        async with session.patch(f"{bitaxe_ip}/api/system", json=settings, timeout=10) as response:
            if response.status != 200:
                logger.error(f"Error setting system settings on {bitaxe_ip}: HTTP {response.status}")
                return False
            logger.info(f"Applying settings... Voltage = {core_voltage}mV, Frequency = {frequency}MHz")
            await asyncio.sleep(2)
            await restart_system(bitaxe_ip, session)
            return True
    except asyncio.TimeoutError:
        logger.error(f"Timeout when setting system settings on {bitaxe_ip}")
    except aiohttp.ClientError as e:
        logger.error(f"Error setting system settings on {bitaxe_ip}: {e}")
    return False

async def restart_system(bitaxe_ip, session):

    try:
        # Check if we're being called from handle_sigint
        is_interrupt = handling_interrupt

        # Restart here as some bitaxes get unstable with bad settings
        # If not an interrupt, wait 90s for system stabilization as some bitaxes are slow to ramp up
        if not is_interrupt:
            logger.info(f"Applying new settings and waiting 90s for system stabilization...")
            async with session.post(f"{bitaxe_ip}/api/system/restart", timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Error restarting the system at {bitaxe_ip}: HTTP {response.status}")
            await asyncio.sleep(90)  # Allow 90s time for the system to restart and start hashing
        else:
            logger.info(f"Applying final settings...")
            async with session.post(f"{bitaxe_ip}/api/system/restart", timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Error restarting the system at {bitaxe_ip}: HTTP {response.status}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout restarting system")
    except aiohttp.ClientError as e:
        logger.error(f"Error restarting the system {e}")

async def benchmark_iteration(bitaxe_ip, core_voltage, frequency, session):
    current_time = time.strftime("%H:%M:%S")
    print(CYAN + f"[{current_time}] Starting benchmark | Core Voltage: {core_voltage}mV, Frequency: {frequency}MHz" + RESET)
    hash_rates = []
    temperatures = []
    power_consumptions = [] 
    vr_temps = []
    fan_speeds = []
    total_samples = benchmark_time // sample_interval
    expected_hashrate = frequency * ((small_core_count * asic_count) / 1000)  # Calculate expected hashrate based on frequency

    for sample in range(total_samples):
        info = await get_system_info(bitaxe_ip, session)
        if info is None:
            logger.info(f"Skipping this iteration due to failure in fetching system info.")
            return None, None, None, False, None, None, None, "SYSTEM_INFO_FAILURE"

        temp = info.get("temp")
        vr_temp = info.get("vrTemp")  # Get VR temperature if available
        voltage = info.get("voltage")
        if temp is None:
            logger.info(f"Temperature data not available from {bitaxe_ip}.")
            return None, None, None, False, None, None, None, "TEMPERATURE_DATA_FAILURE"

        if temp < 5:
            logger.info(f"Temperature is below 5°C for {bitaxe_ip}. This is unexpected. Please check the system.")
            return None, None, None, False, None, None, None, "TEMPERATURE_BELOW_5"

        # Check both chip and VR temperatures
        if temp >= max_temp:
            logger.error(f"Chip temperature exceeded {max_temp}°C on {bitaxe_ip}! Stopping current benchmark.")
            return None, None, None, False, None, None, None, "CHIP_TEMP_EXCEEDED"

        if vr_temp is not None and vr_temp >= max_vr_temp:
            logger.error(f"Voltage regulator temperature exceeded {max_vr_temp}°C on {bitaxe_ip}! Stopping current benchmark.")
            return None, None, None, False, None, None, None, "VR_TEMP_EXCEEDED"

        if voltage is not None:
            if voltage < min_input_voltage:
                logger.error(f"Input voltage is below the minimum allowed value of {min_input_voltage}mV on {bitaxe_ip}! Stopping current benchmark.")
                return None, None, None, False, None, None, None, "INPUT_VOLTAGE_BELOW_MIN"
            if voltage > max_input_voltage:
                logger.error(f"Input voltage is above the maximum allowed value of {max_input_voltage}mV on {bitaxe_ip}! Stopping current benchmark.")
                return None, None, None, False, None, None, None, "INPUT_VOLTAGE_ABOVE_MAX"

        hash_rate = info.get("hashRate")
        power_consumption = info.get("power")
        fan_speed = info.get("fanspeed")    

        if hash_rate is None or power_consumption is None:
            logger.info(f"Hashrate or Watts data not available.")
            return None, None, None, False, None, None, None, "HASHRATE_POWER_DATA_FAILURE"

        if power_consumption > max_power:
            logger.error(f"Power consumption exceeded {max_power}W! Stopping current benchmark.")
            return None, None, None, False, None, None, None, "POWER_CONSUMPTION_EXCEEDED"

        hash_rates.append(hash_rate)
        temperatures.append(temp)
        power_consumptions.append(power_consumption)
        if vr_temp is not None and vr_temp > 0:
            vr_temps.append(vr_temp)
        if fan_speed is not None:
            fan_speeds.append(fan_speed)

        # Calculate percentage progress
        percentage_progress = ((sample + 1) / total_samples) * 100
        status_line = (
            f"[{sample + 1:2d}/{total_samples:2d}] "
            f"{percentage_progress:5.1f}% | "
            f"CV: {core_voltage:4d}mV | "
            f"F: {frequency:4d}MHz | "
            f"H: {int(hash_rate):4d} GH/s | "
            f"IV: {int(voltage):4d}mV | "
            f"T: {int(temp):2d}°C"
        )
        if vr_temp is not None and vr_temp > 0:
            status_line += f" | VR: {int(vr_temp):2d}°C"

        # Add Power (Watts) to the status line if available
        if power_consumption is not None:
            status_line += f" | P: {int(power_consumption):2d} W"

        # Add Fan Speed to the status line if available
        if fan_speed is not None:
            status_line += f" | FAN: {int(fan_speed):2d}%"

        logger.info(status_line)

        # Only sleep if it's not the last iteration
        if sample < total_samples - 1:
            await asyncio.sleep(sample_interval)

    if hash_rates and temperatures and power_consumptions:
        # Remove 3 highest and 3 lowest hashrates in case of outliers
        sorted_hashrates = sorted(hash_rates)
        # Protect against short lists
        if len(sorted_hashrates) > 6:
            trimmed_hashrates = sorted_hashrates[3:-3]  # Remove first 3 and last 3 elements
        else:
            trimmed_hashrates = sorted_hashrates
        average_hashrate = sum(trimmed_hashrates) / len(trimmed_hashrates)

        # Sort and trim temperatures (remove lowest 6 readings during warmup)
        sorted_temps = sorted(temperatures)
        if len(sorted_temps) > 6:
            trimmed_temps = sorted_temps[6:]  # Remove first 6 elements only
        else:
            trimmed_temps = sorted_temps
        average_temperature = sum(trimmed_temps) / len(trimmed_temps)

        # Only process VR temps if we have valid readings
        average_vr_temp = None
        if vr_temps:
            sorted_vr_temps = sorted(vr_temps)
            if len(sorted_vr_temps) > 6:
                trimmed_vr_temps = sorted_vr_temps[6:]  # Remove first 6 elements only
            else:
                trimmed_vr_temps = sorted_vr_temps
            average_vr_temp = sum(trimmed_vr_temps) / len(trimmed_vr_temps)

        average_power = sum(power_consumptions) / len(power_consumptions)

        average_fan_speed = None
        if fan_speeds:
            average_fan_speed = sum(fan_speeds) / len(fan_speeds)
            logger.debug(f"Average Fan Speed   {average_fan_speed:.2f}%")

        # Add protection against zero hashrate
        if average_hashrate > 0:
            efficiency_jth = average_power / (average_hashrate / 1_000)
        else:
            logger.error(f"Warning: Zero hashrate detected on {bitaxe_ip}, skipping efficiency calculation")
            return None, None, None, False, None, None, None, "ZERO_HASHRATE"

        # Calculate if hashrate is within 6% of expected
        hashrate_within_tolerance = (average_hashrate >= expected_hashrate * 0.94)

        logger.debug(f"Average Hashrate   {average_hashrate:.2f} GH/s (Expected: {expected_hashrate:.2f} GH/s)")
        logger.debug(f"Average Temperature   {average_temperature:.2f}°C")
        if average_vr_temp is not None:
            logger.debug(f"Average VR Temperature   {average_vr_temp:.2f}°C")
        logger.debug(f"Efficiency   {efficiency_jth:.2f} J/TH")

        return average_hashrate, average_temperature, efficiency_jth, hashrate_within_tolerance, average_vr_temp, average_power, average_fan_speed, None
    else:
        logger.info(f"No Hashrate or Temperature or Watts data collected for {bitaxe_ip}.")
        return None, None, None, False, None, None, None, "NO_DATA_COLLECTED"

def save_results():
    try:
        # If multiple devices, create multiple saves or a single combined file — original used one file per IP
        # Here we will save a single file with all results, naming by first IP for legacy compatibility
        if results:
            # Use the first result's ip if available
            ip_address = results[0].get("ip", "multi") if results else "multi"
        else:
            ip_address = "no_results"
        filename = f"Benchmark@{ip_address}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
        logger.debug(f"Results saved to {filename}")
        print()  # Add empty line

    except IOError as e:
        logger.error(f"Error saving results to file: {e}")

async def reset_to_best_setting(ip, session):
    """
    Apply the best or default settings for a given IP using the same selection logic,
    but keep the result structure compatible with original reset_to_best_setting.
    """
    if not results:
        logger.info(f"No valid benchmarking results found. Applying predefined default settings to {ip}.")
        await set_system_settings(ip, default_voltage, default_frequency, session)
    else:
        # Filter results for this IP
        ip_results = [r for r in results if r.get("ip") == ip and r.get("averageHashRate") is not None]
        if not ip_results:
            logger.info(f"No valid benchmarking results found for {ip}. Applying predefined default settings.")
            await set_system_settings(ip, default_voltage, default_frequency, session)
        else:
            best_result = sorted(ip_results, key=lambda x: x["averageHashRate"], reverse=True)[0]
            best_voltage = best_result["coreVoltage"]
            best_frequency = best_result["frequency"]

            logger.debug(f"Applying the best settings from benchmarking to {ip}:\n"
                          f"  Core Voltage: {best_voltage}mV\n"
                          f"  Frequency: {best_frequency}MHz")
            await set_system_settings(ip, best_voltage, best_frequency, session)
    await restart_system(ip, session)

# --- Signal handling adapted for asyncio ---
def handle_sigint(signum, frame):
    global system_reset_done, handling_interrupt
    if handling_interrupt or system_reset_done:
        return

    handling_interrupt = True
    logger.error("Benchmarking interrupted by user.")

    # Cancel all tasks
    for t in running_tasks:
        if not t.done():
            t.cancel()

# Register the signal handler for the main process
signal.signal(signal.SIGINT, handle_sigint)

# --- Per-device worker orchestrator ---
async def device_worker(raw_ip):
    global results, system_reset_done, handling_interrupt
    
    bitaxe_ip = f"http://{raw_ip}"

    # For each device open its own ClientSession
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Fetch defaults from the device
        ok = await fetch_default_settings(bitaxe_ip, session)
        if not ok:
            logger.error(f"Failed to fetch defaults for {bitaxe_ip}. Skipping device.")
            return

        # If we're in set-values mode, apply settings and exit for this device
        if args.set_values:
            logger.debug(f"\n--- Applying Settings Only to {bitaxe_ip} ---")
            logger.debug(f"Applying Core Voltage: {initial_voltage}mV, Frequency: {initial_frequency}MHz to Bitaxe at {bitaxe_ip}.")
            await set_system_settings(bitaxe_ip, initial_voltage, initial_frequency, session)
            logger.debug(f"Settings applied to {bitaxe_ip}. Check your Bitaxe web interface to confirm.")
            return

        # Main benchmarking process for this device
        try:
            # Add disclaimer per device
            print(RED + "\nDISCLAIMER:" + RESET)
            print(YELLOW + "This tool will stress test your Bitaxe by running it at various voltages and frequencies." + RESET)
            print(YELLOW + "While safeguards are in place, running hardware outside of standard parameters carries inherent risks." + RESET)
            print(YELLOW + "Use this tool at your own risk. The author(s) are not responsible for any damage to your hardware." + RESET)
            print(MAGENTA + "\nNOTE: Ambient temperature significantly affects these results. The optimal settings found may not" + RESET)
            print(MAGENTA + "work well if room temperature changes substantially. Re-run the benchmark if conditions change.\n" + RESET)

            current_voltage = initial_voltage
            current_frequency = initial_frequency

            while current_voltage <= max_allowed_voltage and current_frequency <= max_allowed_frequency:
                # Apply settings (async)
                await set_system_settings(bitaxe_ip, current_voltage, current_frequency, session)
                avg_hashrate, avg_temp, efficiency_jth, hashrate_ok, avg_vr_temp, avg_power, avg_fan_speed, error_reason = await benchmark_iteration(bitaxe_ip, current_voltage, current_frequency, session)

                if avg_hashrate is not None and avg_temp is not None and efficiency_jth is not None:
                    result = {
                        "ip": raw_ip,
                        "coreVoltage": current_voltage,
                        "frequency": current_frequency,
                        "averageHashRate": avg_hashrate,
                        "averageTemperature": avg_temp,
                        "efficiencyJTH": efficiency_jth,
                        "averagePower": avg_power,
                        "errorReason": error_reason
                    }

                    # Only add VR temp if it exists
                    if avg_vr_temp is not None:
                        result["averageVRTemp"] = avg_vr_temp

                    # Only add Fan Speed if it exists (assuming it's not None)
                    if avg_fan_speed is not None:
                        result["averageFanSpeed"] = avg_fan_speed

                    results.append(result)

                    if hashrate_ok:
                        # If hashrate is good, try increasing frequency
                        if current_frequency + frequency_increment <= max_allowed_frequency:
                            current_frequency += frequency_increment
                        else:
                            break  # We've reached max frequency with good results
                    else:
                        # If hashrate is not good, go back one frequency step and increase voltage
                        if current_voltage + voltage_increment <= max_allowed_voltage:
                            current_voltage += voltage_increment
                            current_frequency -= frequency_increment  # Go back to one frequency step and retry
                            logger.info(f"Hashrate to low compared to expected on {bitaxe_ip}. Decreasing frequency to {current_frequency}MHz and increasing voltage to {current_voltage}mV")
                        else:
                            break  # We've reached max voltage without good results
                else:
                    # If we hit thermal limits or other issues, we've found the highest safe settings
                    logger.error(f"Reached thermal or stability limits. Stopping further testing.")
                    break  # Stop testing higher values

                # Save intermittent results for the device
                save_results()

                # If global interrupt flag is set, break
                if handling_interrupt:
                    logger.critical(f"Interrupt detected; stopping benchmarking loop.")
                    break

        except asyncio.CancelledError:
            logger.error(f"Device worker cancelled.")
        except Exception as e:
            print(RED + f"An unexpected error occurred   {e}" + RESET)
            if results:
                await reset_to_best_setting(raw_ip, session)
                save_results()
            else:
                logger.info(GREEN + "No valid benchmarking results found. Applying predefined default settings." + RESET)
                await set_system_settings(bitaxe_ip, default_voltage, default_frequency, session)
                await restart_system(bitaxe_ip, session)
        finally:
            # Final cleanup per device
            if not system_reset_done:
                if results:
                    await reset_to_best_setting(raw_ip, session)
                    save_results()
                    logger.debug(f"Bitaxe reset to best or default settings and results saved.")
                else:
                    logger.info(f"No valid benchmarking results found for {bitaxe_ip}. Applying predefined default settings.")
                    await set_system_settings(bitaxe_ip, default_voltage, default_frequency, session)
                    await restart_system(bitaxe_ip, session)
                # Do not set system_reset_done True here globally; each device will be handled but flag remains for overall shutdown

# --- Main entrypoint ---
async def main():
    global running_tasks, system_reset_done

    # Launch a worker per IP
    ips = args.bitaxe_ips  # raw IPs like "192.168.1.136"
    tasks = []
    for ip in ips:
        task = asyncio.create_task(device_worker(ip))
        running_tasks.append(task)
        tasks.append(task)

    # Wait for all tasks to finish; handle cancellations gracefully
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Main: tasks cancelled.")
    finally:
        # Once all tasks done or cancelled, perform final combined save/summary
        if results:
            # Sort results by averageHashRate in descending order and get the top 5
            top_5_results = sorted(results, key=lambda x: x.get("averageHashRate", 0), reverse=True)[:5]
            top_5_efficient_results = sorted(results, key=lambda x: x.get("efficiencyJTH", float('inf')), reverse=False)[:5]

            # Create a dictionary containing all results and top performers
            final_data = {
                "all_results": results,
                "top_performers": [
                    {
                        "rank": i,
                        "ip": result["ip"],
                        "coreVoltage": result["coreVoltage"],
                        "frequency": result["frequency"],
                        "averageHashRate": result["averageHashRate"],
                        "averageTemperature": result["averageTemperature"],
                        "efficiencyJTH": result["efficiencyJTH"],
                        "averagePower": result["averagePower"],
                        **({"averageVRTemp": result["averageVRTemp"]} if "averageVRTemp" in result else {}),
                        **({"averageFanSpeed": result["averageFanSpeed"]} if "averageFanSpeed" in result else {})
                    }
                    for i, result in enumerate(top_5_results, 1)
                ],
                "most_efficient": [
                    {
                        "rank": i,
                        "ip": result["ip"],
                        "coreVoltage": result["coreVoltage"],
                        "frequency": result["frequency"],
                        "averageHashRate": result["averageHashRate"],
                        "averageTemperature": result["averageTemperature"],
                        "efficiencyJTH": result["efficiencyJTH"],
                        "averagePower": result["averagePower"],
                        **({"averageVRTemp": result["averageVRTemp"]} if "averageVRTemp" in result else {}),
                        **({"averageFanSpeed": result["averageFanSpeed"]} if "averageFanSpeed" in result else {})
                    }
                    for i, result in enumerate(top_5_efficient_results, 1)
                ]
            }

            # Save the final data to JSON with a consolidated name
            first_ip = results[0].get("ip", "multi")
            filename = f"bitaxe_benchmark_results_{first_ip}.json"
            with open(filename, "w") as f:
                json.dump(final_data, f, indent=4)

            logger.debug("Benchmarking completed.")
            if top_5_results:
                logger.debug("\nTop 5 Highest Hashrate Settings:")
                for i, result in enumerate(top_5_results, 1):
                    logger.debug(f"\nRank {i}:")
                    logger.debug(f"  IP: {result['ip']}")
                    logger.debug(f"  Core Voltage: {result['coreVoltage']}mV")
                    logger.debug(f"  Frequency: {result['frequency']}MHz")
                    logger.debug(f"  Average Hashrate: {result['averageHashRate']:.2f} GH/s")
                    logger.debug(f"  Average Temperature: {result['averageTemperature']:.2f}°C")
                    logger.debug(f"  Efficiency: {result['efficiencyJTH']:.2f} J/TH")
                    logger.debug(f"  Average Power: {result['averagePower']:.2f} W")
                    if "averageFanSpeed" in result:
                        logger.debug(f"  Average Fan Speed: {result['averageFanSpeed']:.2f}%")
                    if "averageVRTemp" in result:
                        logger.debug(f"  Average VR Temperature: {result['averageVRTemp']:.2f}°C")

                logger.debug("\nTop 5 Most Efficient Settings:")
                for i, result in enumerate(top_5_efficient_results, 1):
                    logger.debug(f"\nRank {i}:")
                    logger.debug(f"  IP: {result['ip']}")
                    logger.debug(f"  Core Voltage: {result['coreVoltage']}mV")
                    logger.debug(f"  Frequency: {result['frequency']}MHz")
                    logger.debug(f"  Average Hashrate: {result['averageHashRate']:.2f} GH/s")
                    logger.debug(f"  Average Temperature: {result['averageTemperature']:.2f}°C")
                    logger.debug(f"  Efficiency: {result['efficiencyJTH']:.2f} J/TH")
                    logger.debug(f"  Average Power: {result['averagePower']:.2f} W")
                    if "averageFanSpeed" in result:
                        logger.debug(f"  Average Fan Speed: {result['averageFanSpeed']:.2f}%")
                    if "averageVRTemp" in result:
                        logger.debug(f"  Average VR Temperature: {result['averageVRTemp']:.2f}°C")
            else:
                logger.error("No valid results were found during benchmarking.")
        else:
            logger.error("No results were collected from any device.")
        system_reset_done = True

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; exiting.")
