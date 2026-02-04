#!/usr/bin/env python3
"""
COINFT Interface for Jetson Orin Nano
Connects to COINFT device via UART1 at 1 Mbps
Handles initialization, data streaming, and packet parsing
"""

import serial
import time
import struct
import sys
from datetime import datetime


class CoinfTInterface:
    """Interface for COINFT sensor device"""
    
    # Protocol constants
    STX = 0x02  # Start of packet
    ETX = 0x03  # End of packet
    CMD_INIT = 0x69  # 'i' - Initialize
    CMD_START = 0x73  # 's' - Start streaming
    
    PACKET_SIZE = 26  # 1 (STX) + 1 (header) + 24 (data) + 1 (ETX)
    NUM_CHANNELS = 12
    BYTES_PER_CHANNEL = 2
    
    def __init__(self, port='/dev/ttyTHS0', baudrate=1000000, timeout=1):
        """
        Initialize COINFT interface
        
        Args:
            port: Serial port (default: /dev/ttyTHS0 = UART1)
            baudrate: Baud rate (default: 1000000 = 1 Mbps)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_streaming = False
        self.packet_count = 0
        self.error_count = 0
        
    def connect(self):
        """Establish connection to COINFT device"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
            
            # Clear any stale data
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            print(f"✓ Connected to COINFT on {self.port} @ {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            print(f"✗ Error connecting to {self.port}: {e}")
            return False
    
    def send_command(self, command):
        """
        Send command to COINFT
        
        Args:
            command: Command byte (CMD_INIT or CMD_START)
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected")
            return False
        
        try:
            self.serial_conn.write(bytes([command]))
            cmd_char = chr(command) if 32 <= command < 127 else f"0x{command:02X}"
            print(f"→ Sent command: {cmd_char} (0x{command:02X})")
            return True
            
        except serial.SerialException as e:
            print(f"✗ Error sending command: {e}")
            return False
    
    def initialize(self):
        """Initialize COINFT device"""
        print("\n=== Initializing COINFT ===")
        if self.send_command(self.CMD_INIT):
            time.sleep(0.5)  # Wait for initialization
            print("✓ COINFT initialized")
            return True
        return False
    
    def start_streaming(self):
        """Start data streaming from COINFT"""
        print("\n=== Starting Data Stream ===")
        if self.send_command(self.CMD_START):
            self.is_streaming = True
            print("✓ Data streaming started")
            print(f"\nPacket format: STX(0x02) + Header(1B) + Data(24B) + ETX(0x03)")
            print(f"Data: 12 channels × 2 bytes each\n")
            print("=" * 80)
            return True
        return False
    
    def find_packet_start(self):
        """Search for packet start marker (STX)"""
        while self.serial_conn.in_waiting > 0:
            byte = self.serial_conn.read(1)
            if len(byte) > 0 and byte[0] == self.STX:
                return True
        return False
    
    def read_packet(self):
        """
        Read and parse one COINFT packet
        
        Returns:
            tuple: (header, channels, valid) where channels is list of 12 values
        """
        try:
            # Wait for STX
            if not self.find_packet_start():
                return None, None, False
            
            # Read remaining bytes (header + data + ETX)
            remaining = self.PACKET_SIZE - 1  # Already read STX
            packet_data = self.serial_conn.read(remaining)
            
            if len(packet_data) != remaining:
                self.error_count += 1
                return None, None, False
            
            # Verify ETX
            if packet_data[-1] != self.ETX:
                self.error_count += 1
                print(f"✗ Invalid packet end: 0x{packet_data[-1]:02X} (expected 0x{self.ETX:02X})")
                return None, None, False
            
            # Extract header
            header = packet_data[0]
            
            # Extract channel data (24 bytes = 12 channels × 2 bytes)
            channel_data = packet_data[1:25]
            
            # Parse channels as 16-bit unsigned integers (big-endian)
            channels = []
            for i in range(self.NUM_CHANNELS):
                offset = i * self.BYTES_PER_CHANNEL
                # Try big-endian first (most common)
                value = struct.unpack('>H', channel_data[offset:offset+2])[0]
                channels.append(value)
            
            self.packet_count += 1
            return header, channels, True
            
        except Exception as e:
            self.error_count += 1
            print(f"✗ Packet parsing error: {e}")
            return None, None, False
    
    def display_packet(self, header, channels, verbose=False):
        """
        Display packet data
        
        Args:
            header: Header byte
            channels: List of 12 channel values
            verbose: Show detailed output
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if verbose:
            print(f"\n[{timestamp}] Packet #{self.packet_count}")
            print(f"  Header: 0x{header:02X}")
            print(f"  Channels:")
            for i, value in enumerate(channels):
                print(f"    Ch{i+1:2d}: {value:5d} (0x{value:04X})")
        else:
            # Compact format - one line per packet
            channel_str = " ".join(f"{v:5d}" for v in channels)
            print(f"[{timestamp}] #{self.packet_count:5d} | H:0x{header:02X} | {channel_str}")
    
    def stream_data(self, duration=None, verbose=False, display_rate=1):
        """
        Stream and display data from COINFT
        
        Args:
            duration: Stream duration in seconds (None = infinite)
            verbose: Show detailed packet info
            display_rate: Display every Nth packet (1 = all packets)
        """
        if not self.is_streaming:
            print("Error: Streaming not started. Call start_streaming() first.")
            return
        
        print(f"\nStreaming data... (Press Ctrl+C to stop)")
        if not verbose:
            print(f"\n{'Timestamp':<15} {'Packet':<8} {'Header':<8} {'Channels (12 values)'}")
            print("=" * 80)
        
        start_time = time.time()
        display_counter = 0
        
        try:
            while True:
                # Check duration limit
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Read packet
                header, channels, valid = self.read_packet()
                
                if valid:
                    display_counter += 1
                    if display_counter >= display_rate:
                        self.display_packet(header, channels, verbose)
                        display_counter = 0
                
                time.sleep(0.001)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print(f"\n\n{'=' * 80}")
            elapsed = time.time() - start_time
            print(f"Stream stopped after {elapsed:.1f} seconds")
            self._print_statistics(elapsed)
    
    def _print_statistics(self, elapsed_time):
        """Print streaming statistics"""
        print(f"\nStatistics:")
        print(f"  Total packets: {self.packet_count}")
        print(f"  Errors: {self.error_count}")
        print(f"  Duration: {elapsed_time:.2f} seconds")
        if elapsed_time > 0:
            rate = self.packet_count / elapsed_time
            data_rate = (self.packet_count * self.PACKET_SIZE) / elapsed_time / 1024
            print(f"  Packet rate: {rate:.1f} packets/sec")
            print(f"  Data rate: {data_rate:.2f} KB/sec")
        if self.packet_count > 0:
            error_rate = (self.error_count / (self.packet_count + self.error_count)) * 100
            print(f"  Error rate: {error_rate:.2f}%")
    
    def close(self):
        """Close connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print(f"\n✓ Closed connection to {self.port}")


def main():
    """Main function with interactive menu"""
    print("=" * 80)
    print("COINFT Interface - Jetson Orin Nano")
    print("=" * 80)
    
    # Create interface
    coinft = CoinfTInterface(port='/dev/ttyTHS0', baudrate=1000000)
    
    # Connect
    if not coinft.connect():
        print("\nConnection failed! Check:")
        print("  1. UART1 is enabled (sudo /opt/nvidia/jetson-io/jetson-io.py)")
        print("  2. Device is connected to pins 8 (TX), 10 (RX)")
        print("  3. You have permissions (sudo usermod -a -G dialout $USER)")
        return
    
    try:
        # Initialize
        if not coinft.initialize():
            print("Initialization failed!")
            return
        
        # Wait a moment
        time.sleep(1)
        
        # Start streaming
        if not coinft.start_streaming():
            print("Failed to start streaming!")
            return
        
        # Stream data
        print("\nOptions:")
        print("  1. Stream all packets (verbose)")
        print("  2. Stream compact (one line per packet)")
        print("  3. Stream compact (every 10th packet)")
        
        choice = input("\nSelect option (1-3, default: 2): ").strip()
        
        if choice == '1':
            coinft.stream_data(verbose=True)
        elif choice == '3':
            coinft.stream_data(verbose=False, display_rate=10)
        else:
            coinft.stream_data(verbose=False, display_rate=1)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
    finally:
        coinft.close()


if __name__ == "__main__":
    """
    Usage:
    
    1. Interactive mode (recommended):
       python3 coinft_interface.py
    
    2. Quick test (in Python):
       from coinft_interface import CoinfTInterface
       coinft = CoinfTInterface()
       coinft.connect()
       coinft.initialize()
       coinft.start_streaming()
       coinft.stream_data(duration=10)  # Stream for 10 seconds
       coinft.close()
    """
    main()
