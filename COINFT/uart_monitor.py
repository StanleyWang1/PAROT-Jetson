#!/usr/bin/env python3
"""
UART Bus Monitor for Jetson Orin Nano
Monitors and displays data from custom UART device connected to RX/TX
"""

import serial
import serial.tools.list_ports
import time
import sys
from datetime import datetime


class UARTMonitor:
    """Monitor UART bus and display incoming data streams"""
    
    def __init__(self, port=None, baudrate=9600, timeout=1):
        """
        Initialize UART monitor
        
        Args:
            port: Serial port (e.g., '/dev/ttyTHS0' for Jetson UART)
            baudrate: Baud rate (default: 9600)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        
    @staticmethod
    def list_available_ports():
        """List all available serial ports"""
        ports = serial.tools.list_ports.comports()
        print("\n=== Available Serial Ports ===")
        if not ports:
            print("No serial ports found!")
            return []
        
        for i, port in enumerate(ports):
            print(f"{i+1}. {port.device}")
            print(f"   Description: {port.description}")
            print(f"   Hardware ID: {port.hwid}")
            print()
        
        return [port.device for port in ports]
    
    def connect(self):
        """Establish connection to UART device"""
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
            
            print(f"\n✓ Connected to {self.port}")
            print(f"  Baudrate: {self.baudrate}")
            print(f"  Timeout: {self.timeout}s")
            print(f"  Data bits: 8, Parity: None, Stop bits: 1")
            print("\nMonitoring... (Press Ctrl+C to stop)\n")
            print("=" * 70)
            
            return True
            
        except serial.SerialException as e:
            print(f"✗ Error connecting to {self.port}: {e}")
            return False
    
    def monitor(self, display_mode='ascii', show_hex=False, show_timestamp=True):
        """
        Monitor incoming data from UART
        
        Args:
            display_mode: 'ascii', 'hex', or 'both'
            show_hex: Show hex values alongside ASCII
            show_timestamp: Add timestamp to each line
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected to serial port")
            return
        
        byte_count = 0
        line_buffer = bytearray()
        
        try:
            while True:
                if self.serial_conn.in_waiting > 0:
                    # Read available data
                    data = self.serial_conn.read(self.serial_conn.in_waiting)
                    byte_count += len(data)
                    
                    # Process data based on display mode
                    if display_mode == 'hex':
                        self._display_hex(data, show_timestamp)
                    elif display_mode == 'both':
                        self._display_both(data, show_timestamp)
                    else:  # ASCII mode
                        self._display_ascii(data, show_timestamp, line_buffer)
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
                
        except KeyboardInterrupt:
            print(f"\n\n{'=' * 70}")
            print(f"Monitor stopped. Total bytes received: {byte_count}")
            
        except serial.SerialException as e:
            print(f"\n✗ Serial connection error: {e}")
    
    def _get_timestamp(self):
        """Get formatted timestamp"""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def _display_ascii(self, data, show_timestamp, line_buffer):
        """Display data as ASCII text"""
        for byte in data:
            if byte == 0x0A:  # Newline
                if show_timestamp:
                    print(f"[{self._get_timestamp()}] ", end='')
                try:
                    print(line_buffer.decode('ascii', errors='replace'))
                except:
                    print(line_buffer.hex())
                line_buffer.clear()
            elif byte == 0x0D:  # Carriage return
                continue
            else:
                line_buffer.append(byte)
    
    def _display_hex(self, data, show_timestamp):
        """Display data as hexadecimal"""
        hex_str = ' '.join(f'{b:02X}' for b in data)
        if show_timestamp:
            print(f"[{self._get_timestamp()}] {hex_str}")
        else:
            print(hex_str)
    
    def _display_both(self, data, show_timestamp):
        """Display data as both hex and ASCII"""
        hex_str = ' '.join(f'{b:02X}' for b in data)
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
        
        if show_timestamp:
            print(f"[{self._get_timestamp()}]")
        print(f"  HEX:   {hex_str}")
        print(f"  ASCII: {ascii_str}")
        print()
    
    def send_data(self, data):
        """
        Send data to UART device
        
        Args:
            data: String or bytes to send
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Error: Not connected to serial port")
            return
        
        try:
            if isinstance(data, str):
                data = data.encode('ascii')
            
            bytes_sent = self.serial_conn.write(data)
            print(f"Sent {bytes_sent} bytes: {data}")
            
        except serial.SerialException as e:
            print(f"✗ Error sending data: {e}")
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print(f"\n✓ Closed connection to {self.port}")


def interactive_mode():
    """Run monitor in interactive mode with menu"""
    print("=" * 70)
    print("UART Bus Monitor - Jetson Orin Nano")
    print("=" * 70)
    
    # List available ports
    ports = UARTMonitor.list_available_ports()
    
    if not ports:
        print("\nCommon Jetson Orin Nano UART ports:")
        print("  /dev/ttyTHS0 - UART1 (40-pin header)")
        print("  /dev/ttyTHS1 - UART2")
        print("  /dev/ttyUSB0 - USB-to-Serial adapter")
        print("\nPlease specify the port manually.")
        ports = ['/dev/ttyTHS0']
    
    # Select port
    port_input = input(f"\nEnter port name (or press Enter for {ports[0]}): ").strip()
    port = port_input if port_input else ports[0]
    
    # Select baudrate
    print("\nCommon baud rates: 9600, 19200, 38400, 57600, 115200")
    baudrate_input = input("Enter baud rate (default: 9600): ").strip()
    baudrate = int(baudrate_input) if baudrate_input else 9600
    
    # Select display mode
    print("\nDisplay modes:")
    print("  1. ASCII (default)")
    print("  2. Hexadecimal")
    print("  3. Both")
    mode_input = input("Select display mode (1-3, default: 1): ").strip()
    
    display_modes = {
        '1': 'ascii',
        '2': 'hex',
        '3': 'both',
        '': 'ascii'
    }
    display_mode = display_modes.get(mode_input, 'ascii')
    
    # Create and run monitor
    monitor = UARTMonitor(port=port, baudrate=baudrate)
    
    if monitor.connect():
        try:
            monitor.monitor(display_mode=display_mode)
        finally:
            monitor.close()
    else:
        print("\nFailed to connect. Please check:")
        print("  1. Port name is correct")
        print("  2. Device is connected")
        print("  3. You have permissions (may need sudo)")
        print("  4. Port is not already in use")


def quick_test(port='/dev/ttyTHS0', baudrate=9600):
    """Quick test mode with default settings"""
    print(f"Quick test mode: {port} @ {baudrate} baud")
    
    monitor = UARTMonitor(port=port, baudrate=baudrate)
    
    if monitor.connect():
        try:
            monitor.monitor(display_mode='ascii')
        finally:
            monitor.close()


if __name__ == "__main__":
    """
    Usage examples:
    
    1. Interactive mode (recommended):
       python3 uart_monitor.py
    
    2. Quick test with defaults:
       python3 uart_monitor.py quick
    
    3. Specify port and baudrate:
       python3 uart_monitor.py /dev/ttyTHS0 115200
    """
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'quick':
            quick_test()
        elif len(sys.argv) >= 3:
            port = sys.argv[1]
            baudrate = int(sys.argv[2])
            quick_test(port, baudrate)
        else:
            port = sys.argv[1]
            quick_test(port)
    else:
        interactive_mode()
