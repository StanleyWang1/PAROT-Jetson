import time
import numpy as np
from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite

from control_table import *
from dynamixel_controller import DynamixelController

PORT_NAME = "/dev/ttyUSB0"
# PORT_NAME = "COM8"

# ----------------------
# Motor IDs
# ----------------------
MOTOR24 = 24
MOTOR25 = 25
MOTORS = (MOTOR24, MOTOR25)

# ----------------------
# Home & Soft Limits (adjust to your robot)
# ----------------------
OFFSET = 0
MOTOR24_HOME -= OFFSET   # gripper "open"
MOTOR25_HOME += OFFSET   # gripper "open"

SOFT_MIN = -2_147_483_648      # signed 32-bit ticks (Extended Position)
SOFT_MAX =  2_147_483_647
JOG_AMOUNT = 100  # ticks per jog movement

def clamp(v, vmin=SOFT_MIN, vmax=SOFT_MAX):
    return max(vmin, min(vmax, int(v)))

# ----------------------
# Operating Mode values (X-series)
# 4  = Extended Position
# ----------------------
MODE_EXTENDED_POSITION = 4

def dynamixel_connect():
    controller = DynamixelController(PORT_NAME, 57600, 2.0)
    # controller = DynamixelController("COM8", 57600, 2.0)

    # GroupSyncWrite for GOAL_POSITION (used in Extended Position mode)
    gsw_pos = GroupSyncWrite(
        controller.port_handler,
        controller.packet_handler,
        GOAL_POSITION[0],
        GOAL_POSITION[1],
    )

    # Reboot motors to a clean state
    for motor_id in MOTORS:
        dxl_comm_result, dxl_error = controller.packet_handler.reboot(
            controller.port_handler, motor_id
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[WARN] Reboot {motor_id} failed: "
                  f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"[WARN] Reboot {motor_id} error: "
                  f"{controller.packet_handler.getRxPacketError(dxl_error)}")
        else:
            print(f"[OK] Motor {motor_id} rebooted")
    time.sleep(2.0)

    # Start in Extended Position mode with torque ON
    set_operating_mode(controller, MODE_EXTENDED_POSITION, torque_cycle=True)
    set_motion_profiles(controller, vel=200, acc=50)

    return controller, gsw_pos

def set_motion_profiles(controller, vel=200, acc=50):
    """Set velocity/accel profiles (ticks/s and ticks/s^2)."""
    for m in MOTORS:
        controller.write(m, PROFILE_VELOCITY, int(vel))
        controller.write(m, PROFILE_ACCELERATION, int(acc))
    time.sleep(0.05)

def set_operating_mode(controller, mode, torque_cycle=True):
    """
    Safely switch operating mode:
    - torque off (if torque_cycle)
    - write operating mode
    - torque on (if torque_cycle)
    """
    if torque_cycle:
        for m in MOTORS:
            controller.write(m, TORQUE_ENABLE, 0)
        time.sleep(0.05)

    for m in MOTORS:
        controller.write(m, OPERATING_MODE, int(mode))
    time.sleep(0.05)

    if torque_cycle:
        for m in MOTORS:
            controller.write(m, TORQUE_ENABLE, 1)
        time.sleep(0.05)

def go_home(controller, gsw_pos):
    """Drive both motors to their home (open) positions in Extended Position mode."""
    pos24 = clamp(MOTOR24_HOME)
    pos25 = clamp(MOTOR25_HOME)

    ok = gsw_pos.addParam(MOTOR24, int(pos24).to_bytes(4, "little", signed=True))
    ok &= gsw_pos.addParam(MOTOR25, int(pos25).to_bytes(4, "little", signed=True))
    if not ok:
        print("[ERROR] GroupSyncWrite addParam failed (home)")
        gsw_pos.clearParam()
        return False

    dxl_comm_result = gsw_pos.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ERROR] GroupSyncWrite txPacket (home): "
              f"{controller.packet_handler.getTxRxResult(dxl_comm_result)}")
        gsw_pos.clearParam()
        return False

    gsw_pos.clearParam()
    return True

def move_motor(controller, motor_id, position):
    """Move a single motor to the specified position."""
    controller.write(motor_id, GOAL_POSITION, clamp(position))
    time.sleep(0.05)

def get_current_position(controller, motor_id):
    """Get the current position of a motor."""
    return controller.read(motor_id, PRESENT_POSITION)

def control_motor(controller, motor_id):
    """Interactive control loop for a single motor."""
    print(f"\nControlling Motor {motor_id}")
    print("W = jog positive")
    print("S = jog negative")
    print("Q = return to motor selection")
    
    while True:
        current_pos = get_current_position(controller, motor_id)
        print(f"\rCurrent Position: {current_pos}", end="")
        
        cmd = input("\nEnter command [W/S/Q]: ").strip().upper()
        
        if cmd == 'W':
            new_pos = current_pos + JOG_AMOUNT
            move_motor(controller, motor_id, new_pos)
        elif cmd == 'S':
            new_pos = current_pos - JOG_AMOUNT
            move_motor(controller, motor_id, new_pos)
        elif cmd == 'Q':
            return
        else:
            print("Invalid command. Use W/S/Q.")

def torque_off(controller):
    for m in MOTORS:
        controller.write(m, TORQUE_ENABLE, 0)
    time.sleep(0.05)

def main():
    controller, gsw_pos = dynamixel_connect()
    print("[INFO] Motors initialized in Extended Position mode. Homing...")
    go_home(controller, gsw_pos)
    time.sleep(1.0)
    print("[OK] At home (open).")

    try:
        while True:
            print("\nMotor Selection Menu:")
            print("24 = Control Motor 24")
            print("25 = Control Motor 25")
            print("Q  = Quit")
            
            cmd = input("Enter selection [24/25/Q]: ").strip().upper()
            
            if cmd == 'Q':
                print("[INFO] Quitting...")
                break
            elif cmd in ['24', '25']:
                motor_id = int(cmd)
                control_motor(controller, motor_id)
            else:
                print("[WARN] Invalid selection. Use 24/25/Q.")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt — stopping.")
    finally:
        try:
            torque_off(controller)
        except Exception as e:
            print(f"[WARN] Failed to disable torque cleanly: {e}")
        print("[DONE] Motors stopped and torque off.")

if __name__ == "__main__":
    main()